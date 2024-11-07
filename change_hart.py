"""This file contains code for text-to-image HART Transformer.

This file is adopted and modified from https://github.com/FoundationVision/VAR/blob/main/models/var.py
"""

import math
import types
from typing import Optional, Union
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I, mask_by_order
from hart.modules.networks.basic_hart import AdaLNSelfAttn, LlamaAttention, context_pooling, get_position_ids, apply_rotary_pos_emb
from hart.modules.networks.utils import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from xformers.ops import memory_efficient_attention
# from torch.nn.functional import scaled_dot_product_attention
    
def change_hart_infer(model: HARTForT2I):
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        final_stage=0,
        num_maskgit_iters=1,
        pag_scale:int = 0.0,
        cfg_scale:int = 1.5,
        cd_alpha: float = 0.0,
        cd_beta: float = 0.0,
        dynamic_scale: str = "linear",
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        assert label_B is not None
        assert label_B.shape[1] == self.context_token

        # Cond, Uncond, PAG | Cond, PAG | Cond, Uncond | Cond
        B = label_B.size(0)
        if pag_scale > 0.0 and cfg_scale > 1.0:
            uncond_label_B = torch.full_like(label_B, fill_value=0.0)
            sos = cond_BD = self.context_embed(self.context_norm(torch.cat([label_B, uncond_label_B, label_B], dim=0)))
            uncond_position_ids = torch.full_like(context_position_ids, fill_value=0)
            context_position_ids = torch.cat([context_position_ids, uncond_position_ids, context_position_ids])
            uncond_context_mask = torch.full_like(context_mask, fill_value=0)
            uncond_context_mask[:, 0] = 1
            context_mask = torch.cat([context_mask, uncond_context_mask, context_mask], dim=0)
            bsz = 3 * B
            
        elif pag_scale > 0.0:
            sos = cond_BD = self.context_embed(self.context_norm(torch.cat([label_B, label_B], dim=0)))
            context_position_ids = torch.cat([context_position_ids, context_position_ids])
            context_mask = torch.cat([context_mask, context_mask], dim=0)
            bsz = 2 * B
            
        elif cfg_scale > 1.0:
            uncond_label_B = torch.full_like(label_B, fill_value=0.0)
            sos = cond_BD = self.context_embed(self.context_norm(torch.cat([label_B, uncond_label_B], dim=0)))
            uncond_position_ids = torch.full_like(context_position_ids, fill_value=0)
            context_position_ids = torch.cat([context_position_ids, uncond_position_ids])
            uncond_context_mask = torch.full_like(context_mask, fill_value=0)
            uncond_context_mask[:, 0] = 1
            context_mask = torch.cat([context_mask, uncond_context_mask], dim=0)
            bsz = 2 * B

        else:
            sos = cond_BD = self.context_embed(self.context_norm(torch.cat([label_B], dim=0)))
            bsz = B
        
        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L) # lvl_1L = total generation length

        if self.pos_start is not None:
            next_token_map = (
                sos.expand(bsz, self.first_l, -1) # self.first_l = n_prefix
                + self.pos_start.expand(bsz, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )
        else:
            next_token_map = (
                sos.expand(bsz, self.first_l, -1) + lvl_pos[:, : self.first_l] # bsz, n_prefix, d
            )

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]) # self.Cvae = 32 (codebook) | patch_nums = 1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64

        for b in self.blocks:
            b.attn.kv_caching(True)

        extras = defaultdict(list)
        N = len(self.patch_nums)
        for si, pn in enumerate(self.patch_nums[:-1]):  # si: i-th segment
            num_stages_minus_1 = len(self.patch_nums) - 1
            
            if dynamic_scale == "linear":
                ratio = si / num_stages_minus_1
            elif dynamic_scale == "cosine":
                ratio = np.cos(math.pi / 2.0 * si / num_stages_minus_1)
            elif dynamic_scale == "reverse_linear":
                ratio = 1 - si / num_stages_minus_1
            else:
                dynamic_scale = 1.0
            if si > 0:
                cur_L += pn * pn
            else:
                cur_L += self.context_token # n_prefix = 300
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD) # bsz, n_prefix, d
            x = next_token_map # bsz, n_prefix, d
            # AdaLNSelfAttn.forward
            for b in self.blocks:
                # Haotian: si used for position embed
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                    context_position_ids=context_position_ids, # cond: [0, ..., 39, 40, 40, ..., 40] uncond: [0, 0, ..., 0]
                    context_mask=context_mask, # cond: [True] * 40, [False] * 260 | uncond: [True] + [False] * 299
                    cfg_scale = cfg_scale,
                    pag_scale = pag_scale,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            if si == self.num_stages_minus_1:
                last_layer_cond = x

            #* Apply CFG or PAG
            gamma = cfg_scale * ratio
            omega = pag_scale * ratio
            psi = cd_beta * ratio
            # Cond, Uncond, PAG | Cond, PAG | Cond, Uncond | Cond
            if cfg_scale > 1.0 and pag_scale > 0.0:
                cond, uncond, pag = logits_BlV.chunk(3)
                logits_BlV = (1 + gamma) * cond - gamma * uncond # + omega * pag
                if cd_beta > 0.0:
                    cutoff = math.log(cd_alpha) + logits_BlV.max(dim=-1, keepdim=True).values
                    diffs = (1 + psi) * logits_BlV - psi * pag
                    logits_BlV = diffs.masked_fill(logits_BlV < cutoff, -float('inf'))

            elif cfg_scale > 1.0:
                cond, uncond = logits_BlV.chunk(2)
                
                if cd_beta > 0.0:
                    cutoff = math.log(cd_alpha) + cond.max(dim=-1, keepdim=True).values
                    diffs = (1 + psi) * cond - psi * uncond
                    logits_BlV = diffs.masked_fill(cond < cutoff, -float('inf'))
                else:
                    logits_BlV = (1 + gamma) * cond - gamma * uncond
            elif pag_scale > 0.0:
                cond, pag = logits_BlV.chunk(2)
                logits_BlV = (1 - omega) * cond + omega * pag
            else:
                pass

            # Haotian: Added for text-conditioned generation
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
            )

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(
                bsz // B, 1, 1
            )  # double the batch sizes due to CFG

        ################ last stage maskgit ################
        si = len(self.patch_nums) - 1
        mask = torch.ones(B, self.last_level_pns).cuda()
        tokens = torch.zeros(B, self.last_level_pns, self.Cvae).cuda()
        orders = self.sample_orders(B)

        num_iter = num_maskgit_iters
        indices = list(range(num_iter))
        
        
        # generate latents with maskgit
        for step in indices:
            # mask_ratio = 1 - (step + 1) / num_iter
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.last_level_pns * mask_ratio)]).cuda()
            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, B, self.last_level_pns)
            if step >= num_iter - 1:
                mask_to_pred = mask[:B].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:B].bool(), mask_next.bool())
            mask = mask_next
            cur_mask = torch.cat([mask_to_pred] * (bsz // B), dim=0)
            cur_mask = cur_mask.nonzero(as_tuple=True)
            x = next_token_map[cur_mask].reshape(bsz, -1, self.C)
            for b in self.blocks:
                # Haotian: si used for position embed
                # note: m_maskgit makes sure that PEs are correct.
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=len(self.patch_nums) - 1,
                    m_maskgit=cur_mask,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    cfg_scale = cfg_scale,
                    pag_scale = pag_scale,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            last_layer_cond = x

            #* Apply CFG or PAG
            gamma = cfg_scale * ratio
            omega = pag_scale * ratio
            # Cond, Uncond, PAG | Cond, PAG | Cond, Uncond | Cond
            if cfg_scale > 1.0 and pag_scale > 0.0:
                cond, uncond, pag = logits_BlV.chunk(3)
                logits_BlV = (1 + gamma) * cond - gamma * uncond # + omega * pag
                if cd_beta > 0.0:
                    cutoff = math.log(cd_alpha) + logits_BlV.max(dim=-1, keepdim=True).values
                    diffs = (1 + psi) * logits_BlV - psi * pag
                    logits_BlV = diffs.masked_fill(logits_BlV < cutoff, -float('inf'))

            elif cfg_scale > 1.0:
                cond, uncond = logits_BlV.chunk(2)
                
                if cd_beta > 0.0:
                    cutoff = math.log(cd_alpha) + cond.max(dim=-1, keepdim=True).values
                    diffs = (1 + psi) * cond - psi * uncond
                    logits_BlV = diffs.masked_fill(cond < cutoff, -float('inf'))
                else:
                    logits_BlV = (1 + gamma) * cond - gamma * uncond
            elif pag_scale > 0.0:
                cond, pag = logits_BlV.chunk(2)
                logits_BlV = (1 - omega) * cond + omega * pag
            else:
                pass

            si = len(self.patch_nums) - 1
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            if final_stage == 0:
                # sample with diffusion model
                last_stage_discrete_cond = self.vae_quant_proxy[0].embedding(idx_Bl)
                last_stage_discrete_cond = self.word_embed(last_stage_discrete_cond)
                last_stage_discrete_cond = torch.cat(
                    [last_stage_discrete_cond] * (bsz // B), dim=0
                )
                last_stage_cond = self.decoder_norm(
                    last_layer_cond + last_stage_discrete_cond
                )
                bs, cur_seq_len, _ = last_stage_cond.shape
                ##### begin baseline sampling #####
                last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
                h_BChw_diff = self.diffloss.sample(
                    z=last_stage_cond, temperature=1.0, cfg=gamma
                )
                ##### end baseline sampling #####
                h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
                # [B, L, Cvae]
                h_BChw_diff, *_ = h_BChw_diff.chunk(bsz // B, dim=0)
                # update feature map
                tokens[mask_to_pred] = (h_BChw + h_BChw_diff).reshape(-1, self.Cvae)
            else:
                tokens[mask_to_pred] = h_BChw.reshape(-1, self.Cvae)
            topk_logits = logits_BlV.topk(100, dim=-1)
            extras["logit_hist_vals"].append(topk_logits.values)
            extras["logit_hist_inds"].append(topk_logits.indices)
        h_BChw_final = tokens.transpose(1, 2).reshape(
            B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]
        )
        f_hat += h_BChw_final

        ################ last stage maskgit ################

        for b in self.blocks:
            b.attn.kv_caching(False)
        generated = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        
        return generated, logits_BlV, idx_Bl
    model.autoregressive_infer_cfg = types.MethodType(autoregressive_infer_cfg, model)
    return model



def change_hart_block(model: AdaLNSelfAttn):
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self,
        x,
        cond_BD,
        attn_bias,
        si=-1,
        context_position_ids=None,
        context_mask=None,
        m_maskgit=None,
        cfg_scale : float = 1.5,
        pag_scale : float = 0.0,
    ):  # C: embed_dim, D: cond_dim
        # We achieve multi-token conditioning through LLM attention mask.
        if not self.disable_aln:
            # if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            #     cond_BD = cond_BD.max(1, keepdims=True).values
            condition = context_pooling(
                cond_BD, context_mask=context_mask, mode=self.sep_aln_pooling_mode
            ).unsqueeze(1)

            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_lin(condition).view(-1, 1, 6, self.C).unbind(2)
            )
            x = x + self.drop_path(
                self.attn(
                    self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                    attn_bias=attn_bias,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    si=si,
                    m_maskgit=m_maskgit,
                    cfg_scale = cfg_scale,
                    pag_scale = pag_scale,
                ).mul_(gamma1)
            )
            if self.use_cross_attn:
                # xattn_mask = get_xattn_mask(context_mask)
                x[:, cond_BD.size(1) :] += self.cross_attn(
                    x[:, cond_BD.size(1) :], cond_BD, None
                )
            x = x + self.drop_path(
                self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
            )  # this mul(gamma2) cannot be in-placed when FusedMLP is used
        else:
            if not self.shared_aln:
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x),
                        attn_bias=attn_bias,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                        si=si,
                        m_maskgit=m_maskgit,
                        cfg_scale = cfg_scale,
                        pag_scale = pag_scale,
                    )
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )
                x = x + self.drop_path(self.ffn(self.ln_wo_grad(x)))
            else:
                # cond_BD: [batch, 1, embed_dim]
                condition = context_pooling(cond_BD, context_mask, mode="avg")
                # [batch, 6, embed_dim]
                adaln_modulator = self.scale_shift_table[None] + condition.unsqueeze(1)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = adaln_modulator.chunk(
                    6, dim=1
                )
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                        attn_bias=attn_bias,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                        si=si,
                        m_maskgit=m_maskgit,
                        cfg_scale = cfg_scale,
                        pag_scale = pag_scale,
                    ).mul_(gamma1)
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )
                x = x + self.drop_path(
                            self.ffn(
                                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(
                                    gamma2
                                )
                        )
        return x
    model.forward = types.MethodType(forward, model)
    return model

try:
    from torch.nn.functional import (
        scaled_dot_product_attention as slow_attn,  # q, k, v: BHLc
    )
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None:
            attn.add_(attn_mask)
        return (
            F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
            if dropout_p > 0
            else attn.softmax(dim=-1)
        ) @ value
    
def change_hart_attn(model: LlamaAttention):
    def forward(
        self,
        x,
        attn_bias,
        si=-1,
        context_position_ids=None,
        context_mask=None,
        m_maskgit=None,
        cfg_scale: float = 1.5,
        pag_scale: float = 0.0,
    ):
        
        # cond, uncond, pag | cond, uncond | cond, pag | cond
        pm = None
        if cfg_scale > 1.0 and pag_scale > 0.0:
            cond, uncond, pag = x.chunk(3, dim=0)
            
            if m_maskgit:
                m_maskgit = tuple(torch.cat([m.chunk(3, dim=0)[0], m.chunk(3, dim=0)[1]], dim=0) for m in m_maskgit)
                
            cond_context_position_ids, uncond_context_position_ids, pag_context_position_ids = context_position_ids.chunk(3, dim=0)
            context_position_ids = torch.cat([cond_context_position_ids, uncond_context_position_ids], dim=0)
            # x = (2 + cfg_scale - pag_scale) * cond - cfg_scale * uncond + pag_scale * pag
            x = torch.cat([cond, uncond], dim=0)
        elif cfg_scale > 1.0:
            cond, uncond = x.chunk(2, dim=0)
            # x = (1 + cfg_scale) * cond - cfg_scale * uncond
            pag = None
            
        elif pag_scale > 0.0:
            cond, pag = x.chunk(2, dim=0)
            context_position_ids, pag_context_position_ids = context_position_ids.chunk(2, dim=0)
            if m_maskgit:
                m_maskgit = tuple(torch.cat([m.chunk(2, dim=0)[0], m.chunk(2, dim=0)[1]], dim=0) for m in m_maskgit)
            # x = (1 - pag_scale) * cond + pag_scale *pag
            uncond = None
            x = cond
        else:
            uncond = pag = None

        B, L, C = x.shape
        # [B, L, 2]
        if self.context_token == 0:
            position_ids = get_position_ids(
                B, self.patch_nums, x.device, si=si, m_maskgit=m_maskgit
            )
        else:
            # text to image
            # level 0 does not appear in the position_ids
            # since it is included in context tokens
            # should be 679 tokens for 16x16 latent w/ default 10-stage VAR
            if si == -1:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], x.device, si=si, m_maskgit=m_maskgit
                )
                # largest position_id
                position_ids = _position_ids + context_position_ids[:, -1].unsqueeze(
                    -1
                ).unsqueeze(-1)
            elif si > 0:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], x.device, si=si - 1, m_maskgit=m_maskgit
                )
                # largest position_id
                position_ids = _position_ids + context_position_ids[:, -1].unsqueeze(
                    -1
                ).unsqueeze(-1)
            else:
                pass
        qkv = F.linear(
            input = x,
            weight = self.qkv_proj.weight,
            bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc

        using_flash = (
            self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        )
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1  # q or k or v: BLHc
            dim_unsqueeze = 2
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2  # q or k or v: BHLc
            dim_unsqueeze = 1

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        ################## Use naive rotary embedding ##################
        # apply position embedding to visual tokens
        if self.context_token == 0:
            cos, sin = self.rotary_emb(v, position_ids)
        elif self.context_token > 0:
            if si == -1:
                # training, all tokens
                cos, sin = self.rotary_emb(v, position_ids)
                cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
                cos, sin = torch.cat([cos_c, cos], 1), torch.cat([sin_c, sin], 1)
            elif si == 0:
                # inference step 1, only context tokens
                cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
                cos, sin = cos_c, sin_c
            else:
                # si > 0, no need to add rotary emb for context
                # inference step > 1, only new tokens
                cos, sin = self.rotary_emb(v, position_ids) # 4, 6144, 2
        else:
            print("Context token cannot be negative", self.context_token)
            raise NotImplementedError
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=dim_unsqueeze)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0

        oup = (
            slow_attn(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
            )
            .transpose(1, 2)
            .reshape(B, L, C)
        )
        #! PAG Path
        if pag is not None:
            attention_mask_pag = None
            B, L, C = pag.shape
            # [B, L, 2]
            if self.context_token == 0:
                position_ids = get_position_ids(
                    B, self.patch_nums, x.device, si=si, m_maskgit=pm
                )
            else:
                # text to image
                # level 0 does not appear in the position_ids
                # since it is included in context tokens
                # should be 679 tokens for 16x16 latent w/ default 10-stage VAR
                if si == -1:
                    _position_ids = get_position_ids(
                        B, self.patch_nums[1:], x.device, si=si, m_maskgit=pm
                    )
                    # largest position_id
                    position_ids = _position_ids + pag_context_position_ids[:, -1].unsqueeze(
                        -1
                    ).unsqueeze(-1)
                elif si > 0:
                    _position_ids = get_position_ids(
                        B, self.patch_nums[1:], x.device, si=si - 1, m_maskgit=pm
                    )
                    # largest position_id
                    position_ids = _position_ids + pag_context_position_ids[:, -1].unsqueeze(
                        -1
                    ).unsqueeze(-1)
                else:
                    pass

            qkv = F.linear(
                input = pag,
                weight = self.qkv_proj.weight,
                bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
            ).view(B, L, 3, self.num_heads, self.head_dim)
            
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2  # q or k or v: BHLc
            dim_unsqueeze = 1

            if self.attn_l2_norm:
                scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
                if using_flash or self.using_xform:
                    scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
                q = F.normalize(q, dim=-1).mul(scale_mul)
                k = F.normalize(k, dim=-1)

            ################## Use naive rotary embedding ##################
            # apply position embedding to visual tokens
            if self.context_token == 0:
                cos, sin = self.rotary_emb(v, position_ids)
            elif self.context_token > 0:
                if si == -1:
                    # training, all tokens
                    cos, sin = self.rotary_emb(v, position_ids)
                    cos_c, sin_c = self.context_rotary_emb(v, pag_context_position_ids)
                    cos, sin = torch.cat([cos_c, cos], 1), torch.cat([sin_c, sin], 1)
                elif si == 0:
                    # inference step 1, only context tokens
                    cos_c, sin_c = self.context_rotary_emb(v, pag_context_position_ids)
                    cos, sin = cos_c, sin_c
                else:
                    # si > 0, no need to add rotary emb for context
                    # inference step > 1, only new tokens
                    cos, sin = self.rotary_emb(v, position_ids)
            else:
                print("Context token cannot be negative", self.context_token)
                raise NotImplementedError
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=dim_unsqueeze)

            if self.caching:
                k = torch.cat((self.cached_k[:B], k), dim=dim_cat) # cond | uncond
                v = torch.cat((self.cached_v[:B], v), dim=dim_cat) # # cond | uncond

            if si != 0 and self.context_token > 0:
                attention_mask_pag = torch.zeros(q.size(2), k.size(2), device=x.device)
                attention_mask_pag[ : , self.context_token : -q.size(2)] = float("-inf")

            dropout_p = self.attn_drop if self.training else 0.0

            oup_pag = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attention_mask_pag,
                    dropout_p=dropout_p,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )
            oup = torch.cat([oup, oup_pag], dim=0)
        return self.proj_drop(self.proj(oup))
    model.forward = types.MethodType(forward, model)
    return model