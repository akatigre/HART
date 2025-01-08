
from PIL import Image
from typing import List
from generate import residual_stage
import torch
import torch.nn.functional as F
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
from hart.modules.models.autoencoder import HARTHybridQuantizer, HARTAutoEncoder

from decode import cfg_decode
import types


def change_quant(quantizer: HARTHybridQuantizer):
    def idxBl_to_var_input(
        self, gt_ms_idx_Bl: List[torch.Tensor], logits_list: List[torch.Tensor], patch_nums=None
    ) -> torch.Tensor:
        if patch_nums is None:
            patch_nums = self.v_patch_nums
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = patch_nums[-1]
        SN = len(patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (0 <= self.prog_si - 1 < si):
                break  # progressive training: not supported yet, prog_si always -1
            idx_onehot = torch.nn.functional.one_hot(gt_ms_idx_Bl[si], num_classes=logits_list[si].shape[-1]).float() + logits_list[si].detach() - logits_list[si] # logits_BlV_ is a perturbed logits with epsilon applied
            h_BChw = F.interpolate(
                # last stage might be continuous
                (
                    idx_onehot @ self.embedding.weight
                    if len(gt_ms_idx_Bl[si].shape) == 2
                    else gt_ms_idx_Bl[si]
                )
                .transpose_(1, 2)
                .view(B, C, pn_next, pn_next),
                size=(H, W),
                mode="bicubic",
            )
            
            f_hat = f_hat + self.quant_resi[si / (SN - 1)](h_BChw)
            pn_next = patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
                .view(B, C, -1)
                .transpose(1, 2)
            )
        return (
            torch.cat(next_scales, dim=1) if len(next_scales) else None
        )  # cat BlCs to BLC, this should be float32
    quantizer.idxBl_to_var_input = types.MethodType(idxBl_to_var_input, quantizer)
    return quantizer
   
def change_get_logits(model: HARTForT2I):
    def _rms_norm(x, weight, eps=1e-06):
        x = x.to(weight.dtype)
        variance = (x * x).mean(-1, keepdim=True)
        s_variance = torch.rsqrt(variance + eps)
        return weight * (x * s_variance)
    
    def get_logits(self, x, cond_BD):
        norm_hidden_states = _rms_norm(x.float(), self.head_nm.ln_wo_grad.weight).float()
        if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            cond_BD = cond_BD.max(1, keepdims=True).values
        scale, shift = model.head_nm.ada_lin(cond_BD).view(-1, 1, 2, model.head_nm.C).unbind(2)
        adain_states = norm_hidden_states.mul(scale.add(1))
        adain_states = adain_states + shift.float()
        logits_BlV = self.head(adain_states) # 2B, 9450, V
        return logits_BlV
    
    model.get_logits = types.MethodType(get_logits, model)
    return model


sample_fn = lambda logits: torch.argmax(logits, dim=-1).view(-1, logits.shape[1])
    
# torch.multinomial(logits.softmax(dim=-1).view(-1, logits.shape[-1]), 
    # num_samples=1,
    # replacement=True,
    # generator=None,
    # ).view(-1, logits.shape[1], 1)[:, :, 0]

def generate_img_from_idxs(idxs_list: List[torch.tensor], auto_encoder):
    idxBl_to_img = auto_encoder.idxBl_to_img(idxs_list, same_shape=True, last_one=False)
    soft_img = convert_to_pil(idxBl_to_img[-1])[-1]
    return soft_img
   
def split_into_patches(logits: torch.tensor, patch_nums: List[int]):
    
    start_ends = [(sum([pn * pn for pn in patch_nums[:i]]), sum([pn * pn for pn in patch_nums[:i+1]])) for i in range(len(patch_nums))]
    logits_list = [logits[:, st:ed] for st, ed in start_ends]
    return logits_list

def get_topk_mask(logits_init, topk):
    
    _, topk_indices = torch.topk(logits_init, k=topk, largest=True, sorted=False, dim=-1) # Find the top-k largest values along the last dimension
    mask = torch.zeros_like(logits_init, dtype=torch.float)
    mask.scatter_(dim=2, index=topk_indices, value=1) # Create the final mask and scatter the value 1 for the top-k indices
    return mask

def soft_nll(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)

def convert_to_pil(images):
    images = images.add_(1).mul_(0.5)
    images = (images.clone().cpu() * 255.0)
    np_images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")
    pil_images = [Image.fromarray(image) for image in np_images]
    return pil_images

def soft_forward(
    B, 
    ema_model: HARTForT2I, 
    quantizer: HARTHybridQuantizer, 
    context_position_ids, 
    context_mask, 
    cond_BD,
    sos, 
    cfg_scale:int = 5,
    soft: bool = True,
    logits_list: List[torch.Tensor] = None,
    ):
    #! predict discrete tokens into f_hat
    if soft:
        quantizer = change_quant(quantizer)
        ema_model = change_get_logits(ema_model)
    f_hat = sos.new_zeros(B, ema_model.Cvae, ema_model.patch_nums[-1], ema_model.patch_nums[-1]) # tokens to be predicted
    
    for b in ema_model.blocks:
        b.attn.kv_caching(True)
    cur_L = 0
    next_token_map = sos
    lvl_pos = ema_model.lvl_embed(ema_model.lvl_1L) # 1, 9750, 1536
    output_logits = []
    for si, pn in enumerate(ema_model.patch_nums):
        #* predict discrete tokens
        # 1. Count the number of tokens at si
        cur_tokens = pn ** 2 if si else ema_model.context_token
        cur_L += cur_tokens
        ratio = si / (len(ema_model.patch_nums) - 1)
        
        x = next_token_map
        for b in ema_model.blocks:
            
            x = b(
                x=x, # B, 300, 1536
                cond_BD=ema_model.shared_ada_lin(cond_BD), # 2, 300, 1536 => 2, 4, 1536 => 2, 9, 1536
                attn_bias=None,
                si=si, # for positional embedding of different resolutions
                context_position_ids=context_position_ids, # 2, 300: 0 for paddings 1 for context
                context_mask=context_mask, # 2, 300
            )
        last_hidden_state = x
        logits_BlV = ema_model.get_logits(last_hidden_state, cond_BD)
        logits_cond, logits_uncond = logits_BlV.chunk(2, dim=0)
        logits_BlV = cfg_decode(logit_cond=logits_cond, logit_uncond=logits_uncond, scale=cfg_scale * ratio)
        
        if si == 0:
            logits_BlV = logits_BlV[:, [-1], :] # only use the last prediction
            
        output_logits.append(logits_BlV)
        
        if logits_list is not None:
            logits = logits_list[si]
        else:
            logits = logits_BlV
        idx_Bl = logits.argmax(dim=-1).view(B, pn ** 2)
        
        if soft:
            onehot_idxs = torch.nn.functional.one_hot(idx_Bl, num_classes=logits.shape[-1]) + logits - logits.detach()
            h_BChw = onehot_idxs @ quantizer.embedding.weight
        else:
            h_BChw = quantizer.embedding(idx_Bl)  # B, l, Cvae
            
        if si < len(ema_model.patch_nums) - 1:
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, ema_model.Cvae, pn, pn)

            SN = len(ema_model.patch_nums)
            HW = ema_model.patch_nums[-1]

            h = quantizer.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            ) # conv after upsample
            f_hat = f_hat + h
            
            next_size = ema_model.patch_nums[si + 1]
            next_token_map = F.interpolate(
                    f_hat,
                    size=(next_size, next_size),
                    mode="area",
                )
            pos_embeds = lvl_pos[:, cur_L : cur_L + next_size ** 2]
            next_token_map = next_token_map.view(B, ema_model.Cvae, -1).transpose(1, 2) # interpolated to next scale
           
            word_embeds = ema_model.word_embed(next_token_map)
            
            next_token_map = word_embeds + pos_embeds
            
            next_token_map = next_token_map.repeat(
                2, 1, 1
            ) # double the batch size for CFG
        else:
            f_hat = f_hat + torch.transpose(h_BChw, 1, 2).reshape(B, ema_model.Cvae, pn, pn)
            if logits_list is not None:
                word_embeds = ema_model.word_embed(f_hat.view(B, ema_model.Cvae, -1).transpose(1, 2))
                next_token_map = word_embeds + pos_embeds
                next_token_map = next_token_map.repeat(
                    2, 1, 1
                )
                ratio = (si - 1) / (len(ema_model.patch_nums) - 1)
            
                x = next_token_map
                for b in ema_model.blocks:
                    x = b(
                        x=x, # B, 300, 1536
                        cond_BD=ema_model.shared_ada_lin(cond_BD), # 2, 300, 1536 => 2, 4, 1536 => 2, 9, 1536
                        attn_bias=None,
                        si=si, # for positional embedding of different resolutions
                        context_position_ids=context_position_ids, # 2, 300: 0 for paddings 1 for context
                        context_mask=context_mask, # 2, 300
                    )
                logits_BlV = ema_model.get_logits(x, cond_BD)
                logits_cond, logits_uncond = logits_BlV.chunk(2, dim=0)
                logits_BlV = cfg_decode(logit_cond=logits_cond, logit_uncond=logits_uncond, scale=cfg_scale * ratio)
                output_logits.append(logits_BlV)

    h_BChw_diff = residual_stage(
        ema_model=ema_model, 
        last_hidden_state=last_hidden_state.detach(),
        h_BChw=h_BChw.detach(), # B, patch_nums[-1] x patch_nums[-1], Cvae
        cfg_scale=cfg_scale
    )

    h_BChw_final = h_BChw + h_BChw_diff
    h_BChw_final = h_BChw_final.transpose(1, 2).reshape(
        B, ema_model.Cvae, ema_model.patch_nums[-1], ema_model.patch_nums[-1]
    )
    f_hat_final = f_hat + h_BChw_final
    
    for b in ema_model.blocks:
        b.attn.kv_caching(False)
    auto_encoder: HARTAutoEncoder = ema_model.vae_proxy[0]
    images = auto_encoder.decoder(auto_encoder.post_quant_conv(f_hat_final)).clamp_(-1, 1)
    return images, output_logits

def soft_forward_all(
    pert_logits_list, 
    cond_BD, 
    sos,
    model: HARTForT2I, 
    auto_encoder: HARTAutoEncoder,
    quantizer: HARTHybridQuantizer, 
    context_position_ids, 
    context_mask
    ):
    quantizer = change_quant(quantizer)
    model = change_get_logits(model)
    pert_idx_Bl = [logits.argmax(dim=-1).view(-1, logits.size(1)) for logits in pert_logits_list]
    x_BLCv_wo_first_l = quantizer.idxBl_to_var_input(pert_idx_Bl, pert_logits_list) # B, 9450, 32
    
    B = cond_BD.shape[0] # 2 x original batch size for CFG
    sos = cond_BD.clone()
    sos = sos.expand(-1, model.first_l, -1)
    x_BLCv_wo_first_l = x_BLCv_wo_first_l.expand(B, -1, -1)
    dtype = x_BLCv_wo_first_l.dtype
    lvl_pos = model.lvl_embed(model.lvl_1L) # 1, 9750, 1536
    with torch.cuda.amp.autocast(enabled=False):
        
        x_BLC = torch.cat((sos, model.word_embed(x_BLCv_wo_first_l.float())), dim=1) # 2B, 9750, 1536
        x_BLC += lvl_pos.expand(B, -1, -1) # 2B, 9750, 1536
        
    ed = model.L + model.context_token - 1 # 9451 + 300 - 1 = 9750
    attn_bias = model.attn_bias_for_masking[:, :, :ed, :ed]
    cond_BD_or_gss = model.shared_ada_lin(cond_BD)
    
    dtype = x_BLC.dtype
    x_BLC = x_BLC.to(dtype=dtype)
    cond_BD_or_gss = cond_BD_or_gss.to(dtype=dtype)
    attn_bias = attn_bias.to(dtype=dtype)
    
    for i, b in enumerate(model.blocks):
        x_BLC = b(
            x=x_BLC, 
            cond_BD=cond_BD_or_gss, 
            attn_bias=attn_bias, 
            context_position_ids=context_position_ids, 
            context_mask=context_mask
            )
    x_BLC_logits, last_layer_cond = (
            x_BLC,
            x_BLC[:, -model.last_level_pns :, :],
        )
    x_BLC_logits = model.get_logits(x_BLC_logits.float(), cond_BD) # 2B, 9750, 4096
    
    logits_cond, logits_uncond = x_BLC_logits[:, -model.L :].chunk(2, dim=0) # 2B, 9451, 4096
    logits_cfg = cfg_decode(logits_cond, logits_uncond, scale=5)
    idx_BL_sampled = sample_fn(logits_cfg)
    start_ends = [(sum([pn * pn for pn in model.patch_nums[:i]]), sum([pn * pn for pn in model.patch_nums[:i+1]])) for i in range(len(model.patch_nums))]
    assert len(start_ends) == len(model.patch_nums) and start_ends[-1][1] == model.L
    idx_Bl_list = [idx_BL_sampled[:, st:ed] for st, ed in start_ends]
    f_hat = auto_encoder.idxBl_to_fhat(idx_Bl_list, same_shape=True, last_one=True)
    
    h_BChw = quantizer.embedding(idx_BL_sampled[:, -model.last_level_pns:])
    
    last_stage_discrete_cond = model.word_embed(h_BChw.clone())
    last_stage_discrete_cond = torch.cat(
                    [last_stage_discrete_cond, last_stage_discrete_cond], dim=0
                )
    last_layer_cond = model.decoder_norm(last_layer_cond + last_stage_discrete_cond)
    bs, cur_seq_len, _ = last_layer_cond.shape
    last_layer_cond = last_layer_cond.reshape(bs * cur_seq_len, -1)
    h_BChw_diff = model.diffloss.sample(
                    z=last_layer_cond, temperature=1.0, cfg=5
                )
    
    h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
    h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
    tokens = (h_BChw + h_BChw_diff).reshape(-1, model.last_level_pns, model.Cvae) # B, 4096, 32
    h_BChw_final = tokens.transpose(1, 2).reshape(
            -1, model.Cvae, model.patch_nums[-1], model.patch_nums[-1]
        )
    # f_hat = f_hat[-1]
    f_hat += h_BChw_final
    images = auto_encoder.fhat_to_img(f_hat)
    images = convert_to_pil(images)
    images[0].save("test1.png")
    return logits_cfg, images