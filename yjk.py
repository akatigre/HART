import torch
import torch.nn.functional as F
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
from hart.modules.models.autoencoder import HARTHybridQuantizer, HARTAutoEncoder
from decode import cfg_decode

def soft_forward_cfg_all(ema_model: HARTForT2I, 
                     epsilon: torch.tensor, 
                     hidden_states: torch.tensor,
                     logits: torch.tensor, 
                     topk_mask: torch.tensor,
                     context_position_ids: torch.tensor,
                     context_mask: torch.tensor,
                     cond_BD: torch.tensor,
                     cfg_scale: float,
                     quantizer: HARTHybridQuantizer,
                     ):
    #! predict discrete tokens into f_hat
    # 1. add epsilon to logits
    logits = ema_model.get_logits(hidden_states + epsilon, cond_BD)
    # 2. mask out tokens with initial topk
    logits_ = logits.masked_fill(topk_mask, -torch.inf)
    B, l, V = logits.shape
    idx_Bl = torch.multinomial(
        logits_.softmax(dim=-1).view(-1, V),
        num_samples = 1,
        replacement = True,
        generator = None,
    ).view(B, l, -1)[:, :, 0] # get the top-1 logits
    # 2. do Straight-Through Estimator
    idx_onehot = torch.nn.functional.one_hot(idx_Bl, num_classes=V).float()
    idx_onehot = idx_onehot + logits.detach() - logits # logits_BlV_ is a perturbed logits with epsilon applied
    h_BChw = idx_onehot @ quantizer.embedding.weight
    h_BChw = h_BChw.reshape(B, -1, ema_model.Cvae) # B, 9450, 32
    # 300 + sum([pn * pn for pn in ema_model.patch_nums[1:]]) = 9750 -> context + 2nd patch
    attn_bias = ema_model.attn_bias_for_masking # 1, 1, 9750, 9750
    
    word_embeds = ema_model.word_embed(h_BChw) # B, 9450, 1536
    lvl_pos = ema_model.lvl_embed(ema_model.lvl_1L) # B, 9750, 1536
    
    # word_embeds = word_embeds.repeat(2, 1, 1) # 2B, 9750, 1536
    sos = cond_BD.clone()
    x = torch.cat((sos, word_embeds), dim=1) # B, 9750, 1536
    x = x + lvl_pos # B, 9750, 1536
    for b in ema_model.blocks:
        x = b(
            x = x, # B, 5355, 1536
            cond_BD = ema_model.shared_ada_lin(cond_BD), # B, 300, 1536
            attn_bias = attn_bias, 
            si = -1, # si = -1 for all tokens
            context_position_ids = context_position_ids,
            context_mask = context_mask,
        ) # 2B, 9750, 1536
        
    last_hidden_state = x
    logits_BlV = ema_model.get_logits(last_hidden_state, cond_BD) # 2B, 9750, V
    logits_cond, logits_uncond = logits_BlV.chunk(2, dim=0)
    logits_BlV = cfg_decode(
        logit_cond=logits_cond, 
        logit_uncond=logits_uncond, 
        scale=cfg_scale
        ) # B, 9750, V
    selected_tokens = logits_BlV.max(dim=-1).indices
    selected_tokens = selected_tokens.view(B, l, -1)[:, :, 0] # B, 9750
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(logits_BlV, selected_tokens)
    return loss, selected_tokens


def image_decode(selected_tokens, quantizer, ema_model, lvl_pos, context_position_ids, context_mask, cond_BD, cfg_scale, top_k, top_p, rng):
    B = selected_tokens.shape[0]
    pn = sum(ema_model.patch_nums)
    h_BChw = quantizer.embedding(selected_tokens)  # B, l, Cvae
    h_BChw = h_BChw.transpose_(1, 2).reshape(B, ema_model.Cvae, pn, pn)

    # 4. Add residual to predicted logits and interpolate to get next resolution
    SN = len(ema_model.patch_nums)
    HW = ema_model.patch_nums[-1]
    f_hat.add_(h_BChw)
    return
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

    tokens = last_stage(
        next_token_map=next_token_map,
        ema_model=ema_model, 
        ratio=ratio,

        context_position_ids=context_position_ids,
        context_mask=context_mask,
        cfg_scale=cfg_scale,
        B=B,
        cond_BD=cond_BD,
        top_k=top_k, 
        top_p=top_p,
        rng=rng,
        quantizer=quantizer,
    )

    h_BChw_final = tokens.transpose(1, 2).reshape(
        B, ema_model.Cvae, ema_model.patch_nums[-1], ema_model.patch_nums[-1]
    )
    f_hat += h_BChw_final

    for b in ema_model.blocks:
        b.attn.kv_caching(False)
    auto_encoder: HARTAutoEncoder = ema_model.vae_proxy[0]
    images = auto_encoder.decoder(auto_encoder.post_quant_conv(f_hat)).clamp_(-1, 1)
    images = images.add_(1).mul_(0.5) # scale between 0, 1
    return images, logits_list, filter_list