import types
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
from hart.modules.models.transformer.hart_transformer_t2i import HARTAutoEncoder
from decode import cfg_decode
from yjk import residual_stage
from transformers import Trainer
from einops import rearrange
from torch import optim

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

class SoftModel(nn.Module):
    def __init__(self, 
                 logits_list: List[torch.Tensor], 
                 start_noise: float,
                 model: HARTForT2I,
                ):
        super().__init__()
        self.biases_list = nn.ParameterList([
            nn.Parameter(torch.randn_like(logits) * start_noise)
            for logits in logits_list
        ])
        self.logits_list = logits_list
        self.model = model

    def logits_to_embed(self, logits, model, quantizer, si, f_hat):
        pn = model.patch_nums[si]
        
        SN = len(model.patch_nums)
        HW = model.patch_nums[-1]
        
        idx_Bl = logits.argmax(dim=-1).view(-1, pn ** 2)
        onehot_idxs = torch.nn.functional.one_hot(idx_Bl, num_classes=logits.shape[-1]) + logits - logits.detach()
        h_BChw = onehot_idxs @ quantizer.embedding.weight
        h_BChw = h_BChw.transpose_(1, 2).reshape(-1, model.Cvae, pn, pn)

        if pn != HW:
            next_size = model.patch_nums[si + 1]
            h = quantizer.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            ) # conv after upsample
            f_hat = f_hat + h
        
            next_token_map = F.interpolate(
                    f_hat,
                    size=(next_size, next_size),
                    mode="area",
                )
            
            
        else:
            f_hat = f_hat + h_BChw
            next_token_map = f_hat
        next_token_map = next_token_map.view(-1, model.Cvae, next_size * next_size).transpose(1, 2) # interpolated to next scale
        word_embeds = model.word_embed(next_token_map)
        return word_embeds, h_BChw
    
    def get_logits(self, model, x, cond_BD):
        def _rms_norm(x, weight, eps=1e-06):
            x = x.to(weight.dtype)
            variance = (x * x).mean(-1, keepdim=True)
            s_variance = torch.rsqrt(variance + eps)
            return weight * (x * s_variance)

        norm_hidden_states = _rms_norm(x.float(), self.head_nm.ln_wo_grad.weight).float()
        if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            cond_BD = cond_BD.max(1, keepdims=True).values
        scale, shift = model.head_nm.ada_lin(cond_BD).view(-1, 1, 2, model.head_nm.C).unbind(2)
        adain_states = norm_hidden_states.mul(scale.add(1))
        adain_states = adain_states + shift.float()
        logits_BlV = self.head(adain_states) # 2B, 9450, V
        return logits_BlV
        
    def forward(self, 
        context_position_ids, 
        context_mask, 
        cond_BD,
        sos, 
        cfg_scale:int = 5,
        **kwargs
        ):
        quantizer = self.model.vae_quant_proxy[0]
        B = cond_BD.shape[0]
        f_hat = sos.new_zeros(B, self.model.Cvae, self.model.patch_nums[-1], self.model.patch_nums[-1]) # tokens to be predicted
        
        for b in self.model.blocks:
            b.attn.kv_caching(True)
        
        cur_L = 0
        next_token_map = sos
        lvl_pos = self.model.lvl_embed(self.model.lvl_1L) # 1, 9750, 1536
        output_logits = []
        for si, pn in enumerate(self.model.patch_nums):
            #* predict discrete tokens
            # 1. Count the number of tokens at si
            cur_tokens = pn ** 2 if si else self.model.context_token
            cur_L += cur_tokens
            
            ratio = si / (len(self.model.patch_nums) - 1)
            
            x = next_token_map
            for b in self.model.blocks:
                
                x = b(
                    x=x, # B, 300, 1536
                    cond_BD=self.model.shared_ada_lin(cond_BD), # 2, 300, 1536 => 2, 4, 1536 => 2, 9, 1536
                    attn_bias=None,
                    si=si, # for positional embedding of different resolutions
                    context_position_ids=context_position_ids, # 2, 300: 0 for paddings 1 for context
                    context_mask=context_mask, # 2, 300
                )
            last_hidden_state = x
            logits_BlV = self.get_logits(self.model, last_hidden_state, cond_BD)
            logits_cond, logits_uncond = logits_BlV.chunk(2, dim=0)
            logits_BlV = cfg_decode(logit_cond=logits_cond, logit_uncond=logits_uncond, scale=cfg_scale * ratio)
            
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :] # only use the last prediction
                
            output_logits.append(logits_BlV)
            logits = self.logits_list[si] + self.biases_list[si]
            assert self.logits_list[si].requires_grad == False and self.biases_list[si].requires_grad == True, f"logits_list[{si}] and biases_list[{si}] must be frozen and trainable respectively"
            next_size = self.model.patch_nums[si + 1]
            word_embeds, h_BChw = self.logits_to_embed(logits, self.model, quantizer, si, f_hat)
            pos_embeds = lvl_pos[:, cur_L : cur_L + next_size ** 2]
            next_token_map = word_embeds + pos_embeds
            next_token_map = next_token_map.repeat(
                2, 1, 1
            ) # double the batch size for CFG

        h_BChw_diff = residual_stage(
            model=self.model, 
            last_hidden_state=last_hidden_state.detach(),
            h_BChw=h_BChw.detach(), # B, patch_nums[-1] x patch_nums[-1], Cvae
            cfg_scale=cfg_scale
        )

        h_BChw_final = h_BChw + h_BChw_diff
        h_BChw_final = h_BChw_final.transpose(1, 2).reshape(
            -1, self.model.Cvae, self.model.patch_nums[-1], self.model.patch_nums[-1]
        )
        f_hat_final = f_hat + h_BChw_final
        for b in self.model.blocks:
            b.attn.kv_caching(False)
        auto_encoder: HARTAutoEncoder = self.model.vae_proxy[0]
        images = auto_encoder.decoder(auto_encoder.post_quant_conv(f_hat_final)).clamp_(-1, 1)
        return {'images': images, 'logits': output_logits}
    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits_cfg_list = outputs.get("logits")
        logits_cfg = [
            F.interpolate(
                logits.permute(0, 2, 1).reshape(-1, model.V, model.patch_nums[si+1], model.patch_nums[si+1]), 
                size=(model.patch_nums[si], model.patch_nums[si]), 
                mode="bicubic",
                ) 
            for si, logits in enumerate(logits_cfg_list[1:-1])
            ] + [
                logits_cfg_list[-1].permute(0, 2, 1).reshape(-1, model.V, model.patch_nums[-1], model.patch_nums[-1])
                ]
        logits_cfg = [rearrange(logits, "B C h w -> B (h w) C") for logits in logits_cfg]
        logits_cfg = torch.cat(logits_cfg, dim=1) # B, 9451, 4096
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits_cfg.view(-1, model.V), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    