import torch
import torch.nn.functional as F
from typing import Callable
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
from hart.modules.networks.utils import (
    sample_with_top_k_top_p_,
)
from hart.modules.models.autoencoder import HARTHybridQuantizer, HARTAutoEncoder



# @torch.inference_mode()
@torch.no_grad()
def generate_images(
    sos, 
    B, 
    ema_model: HARTForT2I, 
    quantizer: HARTHybridQuantizer,
    context_tensor, 
    context_position_ids, 
    context_mask, 
    cond_BD, # prefix conditional tokens
    lvl_pos, 
    next_token_map, 
    decode_func: Callable,
    cfg_scale:int = 5, 
    top_k = 0, 
    top_p = 0.0,
    num_samples: int = 1,
    rng = None
    ):
    """
        epsilon [torch.Tensor]: parameters to be optimized, shape: B, Cvae, patch_nums[-1], patch_nums[-1], same shape as f_hat
    """
    #! predict discrete tokens into f_hat
    f_hat = sos.new_zeros(B, ema_model.Cvae, ema_model.patch_nums[-1], ema_model.patch_nums[-1]) # tokens to be predicted
    B = context_tensor.size(0)
    
    for b in ema_model.blocks:
        b.attn.kv_caching(True)
    cur_L = 0
    hidden_state_list = []
    logits_list = []
    filter_list = []
    # for si, pn in enumerate(ema_model.patch_nums[:-1]):
    for si, pn in enumerate(ema_model.patch_nums):
        #* predict discrete tokens
        # 1. Count the number of tokens at si
        cur_L += pn ** 2 if si else ema_model.context_token
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
        hidden_state_list.append(last_hidden_state.detach().cpu().clone())
        logits_BlV = ema_model.get_logits(last_hidden_state, cond_BD)
        logits_cond, logits_uncond = logits_BlV.chunk(2, dim=0)
        logits_BlV = decode_func(logit_cond=logits_cond, logit_uncond=logits_uncond, scale=cfg_scale * ratio)
        
        # Added for text-conditioned generation
        if si == 0:
            logits_BlV = logits_BlV[:, [-1], :] # only use the last prediction
        # 3. Sample token index with top-k and top-p
        B, l, V = logits_BlV.shape
        logits_list.append(logits_BlV.detach().cpu().clone())
        topk_filter = logits_BlV > logits_BlV.topk(
            k=10, largest=True, sorted=False, dim=-1
        )[0].amin(dim=-1, keepdim=True)
        filter_list.append(topk_filter)
        
        logits_BlV = logits_BlV.detach().clone()
        logits_BlV = sample_with_top_k_top_p_(
            logits_BlV.detach().clone(),
            top_k = (600 if si < 7 else 300),
            top_p = top_p
        ) # mask vocabularies
        
        idx_Bl = torch.multinomial(
            logits_BlV.softmax(dim=-1).view(-1, V),
            num_samples=abs(num_samples),
            replacement=(num_samples >= 0),
            generator=rng,
        ).view(B, l, num_samples)[:, :, 0]
        h_BChw = quantizer.embedding(idx_Bl)  # B, l, Cvae
        
        if si < len(ema_model.patch_nums) - 1:
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, ema_model.Cvae, pn, pn)

            # 4. Add residual to predicted logits and interpolate to get next resolution
            SN = len(ema_model.patch_nums)
            HW = ema_model.patch_nums[-1]
            if si != SN - 1:
                h = quantizer.quant_resi[si / (SN - 1)](
                    F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
                )  # conv after upsample
                f_hat.add_(h)
            else:
                f_hat.add_(h_BChw)
            
            next_token_map = F.interpolate(
                    f_hat,
                    size=(ema_model.patch_nums[si + 1], ema_model.patch_nums[si + 1]),
                    mode="area",
                )

            next_token_map = next_token_map.view(B, ema_model.Cvae, -1).transpose(1, 2) # interpolated to next scale
            word_embeds = ema_model.word_embed(next_token_map)
            pos_embeds = lvl_pos[:, cur_L : cur_L + ema_model.patch_nums[si + 1] ** 2]
            next_token_map = word_embeds + pos_embeds
            
            next_token_map = next_token_map.repeat(
                2, 1, 1
            ) # double the batch size for CFG

    h_BChw_diff = residual_stage(
        ema_model=ema_model, 
        last_hidden_state=last_hidden_state, 
        h_BChw=h_BChw,
        cfg_scale=cfg_scale
    )

    h_BChw_final = h_BChw + h_BChw_diff
    h_BChw_final = h_BChw_final.transpose(1, 2).reshape(
        B, ema_model.Cvae, ema_model.patch_nums[-1], ema_model.patch_nums[-1]
    )
    f_hat += h_BChw_final

    for b in ema_model.blocks:
        b.attn.kv_caching(False)
    auto_encoder: HARTAutoEncoder = ema_model.vae_proxy[0]
    images = auto_encoder.decoder(auto_encoder.post_quant_conv(f_hat)).clamp_(-1, 1)
    images = images.add_(1).mul_(0.5) # scale between 0, 1
    return images, hidden_state_list, logits_list, filter_list

def residual_stage(
    ema_model: HARTForT2I, 
    last_hidden_state, 
    h_BChw,
    cfg_scale,
):
    last_stage_discrete_cond = h_BChw.detach().clone()
    last_stage_discrete_cond = ema_model.word_embed(last_stage_discrete_cond)
    last_stage_discrete_cond = torch.cat(
        [last_stage_discrete_cond, last_stage_discrete_cond],
        dim=0
    )
    last_stage_cond = ema_model.decoder_norm(
        last_hidden_state + last_stage_discrete_cond
    )
    bs, cur_seq_len, _ = last_stage_cond.shape
    last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
    
    h_BChw_diff = ema_model.diffloss.sample(
        z = last_stage_cond,
        temperature = 1.0,
        cfg = cfg_scale,
    )
    
    h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
    h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
    return h_BChw_diff