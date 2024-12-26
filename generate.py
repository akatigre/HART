import torch
from tqdm import trange
from typing import Callable
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
from hart.modules.networks.utils import (
    gumbel_softmax_with_rng,
    sample_with_top_k_top_p_,
)
from hart.modules.models.autoencoder import HARTHybridQuantizer
import decode
from decode import sample_ste
from hart.utils import encode_prompts, llm_system_prompt

def generate_images(sos, B, 
            ema_model: HARTForT2I, 
            quantizer: HARTHybridQuantizer,
            context_tensor, context_position_ids, context_mask, 
            cond_BD, # prefix conditional tokens
            lvl_pos, next_token_map, rng, cfg, cfg_scale, 
            decode_func: Callable, 
            more_smooth:bool = False,
            final_stage:int = 0,
            num_maskgit_iters:int = 1,
            top_k = 0, 
            top_p = 0.0,
            soft:bool = False,
            do_langevin_dynamics: bool = False,
            epsilon: torch.Tensor = None,
            ):
    #! predict discrete tokens into f_hat
    f_hat = sos.new_zeros(B, ema_model.Cvae, ema_model.patch_nums[-1], ema_model.patch_nums[-1]) # tokens to be predicted
    B = context_tensor.size(0)
    
    for b in ema_model.blocks:
        b.attn.kv_caching(True)
    cur_L = 0
    
    for si, pn in enumerate(ema_model.patch_nums[:-1]):
        #* predict discrete tokens
        # 1. Count the number of tokens at si
        cur_L += pn ** 2 if si else ema_model.context_token

        ratio = si / ema_model.num_stages_minus_1
        
        # 2. Get logits at si
        with torch.inference_mode():
            _, logits_BlV = ema_model.single_step(
                decode_func = decode_func, 
                si = si, 
                ratio = ratio,
                context_position_ids = context_position_ids, 
                context_mask = context_mask, 
                next_token_map = next_token_map, 
                cond_BD = cond_BD, 
                cfg = cfg_scale,
                epsilon = epsilon if do_langevin_dynamics else None
            )
        # 3. Sample token index with top-k and top-p
        logits_BlV, num_samples, replacement = sample_with_top_k_top_p_(
            logits_BlV.clone().detach(),
            top_k=(600 if si < 7 else 300),
            top_p=top_p,
            num_samples=1,
        )
        
        B, l, V = logits_BlV.shape
        idx_Bl = torch.multinomial(
            logits_BlV.softmax(dim=-1).view(-1, V),
            num_samples=num_samples,
            replacement=replacement,
            generator=rng,
        ).view(B, l, num_samples)[:, :, 0]
        if soft: # straight-through estimator
            idx_onehot = torch.nn.functional.one_hot(idx_Bl, num_classes=V)
            idx_onehot = idx_onehot + logits_BlV.detach() - logits_BlV
            h_BChw = idx_onehot @ quantizer.embedding.weight
        else:
            if not more_smooth:
                # this is the default case
                h_BChw = quantizer.embedding(idx_Bl)  # B, l, Cvae
            else:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ quantizer.embedding.weight.unsqueeze(0)

        h_BChw = h_BChw.transpose_(1, 2).reshape(B, ema_model.Cvae, pn, pn)

        # 4. Add residual to predicted logits and interpolate to get next resolution
        f_hat, next_token_map = quantizer.get_next_autoregressive_input(
            si, len(ema_model.patch_nums), f_hat, h_BChw, patch_nums=ema_model.patch_nums
        ) # f_hat: B, Cvae, patch_nums[si], patch_nums[si] | next_token_map: B, Cvae, patch_nums[si+1], patch_nums[si+1]

        next_token_map = next_token_map.view(B, ema_model.Cvae, -1).transpose(1, 2) # interpolated to next scale
        next_token_map = (
            ema_model.word_embed(next_token_map)
            + lvl_pos[:, cur_L : cur_L + ema_model.patch_nums[si + 1] ** 2]
        )
        next_token_map = next_token_map.repeat(
            2, 1, 1
        ) # double the batch size for CFG

    # Predict the last stage resolution with MaskGit sampling (iter = 1, not used)
    si = len(ema_model.patch_nums) - 1
    mask = torch.ones(B, ema_model.last_level_pns).cuda()
    tokens = torch.zeros(B, ema_model.last_level_pns, ema_model.Cvae).cuda()
    orders = ema_model.sample_orders(B)
    with torch.inference_mode():
        last_hidden_state, logits_BlV, mask_to_pred = ema_model.final_stage(
            mask = mask, 
            orders = orders, 
            step = 0, 
            num_iter = num_maskgit_iters, 
            context_position_ids = context_position_ids, 
            context_mask = context_mask, 
            ratio = ratio, 
            cfg = cfg_scale, 
            next_token_map = next_token_map, 
            cond_BD = cond_BD, 
            B = B, 
            cond_BD_or_gss = ema_model.shared_ada_lin(cond_BD), 
            decode_func = decode_func
            ) # maskgit tokens
        
        
    logits_BlV, num_samples, replacement = sample_with_top_k_top_p_(
        logits_BlV.clone().detach(),
        top_k = top_k,
        top_p = top_p,
        num_samples=1,
    )
    
    B, l, V = logits_BlV.shape
    idx_Bl = torch.multinomial(
        logits_BlV.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=rng,
    ).view(B, l, num_samples)[:, :, 0]
    if soft: # straight-through estimator
        idx_onehot = torch.nn.functional.one_hot(idx_Bl, num_classes=V)
        idx_onehot = idx_onehot + logits_BlV.detach() - logits_BlV
        h_BChw = idx_onehot @ quantizer.embedding.weight
    else:        
        if not more_smooth:  # this is the default case
            h_BChw = quantizer.embedding(idx_Bl)  # B, l, Cvae
        else:  # not used when evaluating FID/IS/Precision/Recall
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
            h_BChw = gumbel_softmax_with_rng(
                logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
            ) @ quantizer.embedding.weight.unsqueeze(0)
    
    if final_stage == 0:
        # sample with diffusion model
        last_stage_discrete_cond = quantizer.embedding(idx_Bl)
        last_stage_discrete_cond = ema_model.word_embed(last_stage_discrete_cond)
        last_stage_discrete_cond = torch.cat(
            [last_stage_discrete_cond, last_stage_discrete_cond], dim=0
        )
        last_stage_cond = ema_model.decoder_norm(
            last_hidden_state + last_stage_discrete_cond
        )
        bs, cur_seq_len, _ = last_stage_cond.shape
        ##### begin baseline sampling #####
        last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
        h_BChw_diff = ema_model.diffloss.sample(
            z=last_stage_cond, temperature = 1.0, cfg = cfg_scale,
        )
        ##### end baseline sampling #####
        h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
        # [B, L, Cvae]
        h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
        # update feature map
        tokens[mask_to_pred] = (h_BChw + h_BChw_diff).reshape(-1, ema_model.Cvae)
    else:
        tokens[mask_to_pred] = h_BChw.reshape(-1, ema_model.Cvae)

    h_BChw_final = tokens.transpose(1, 2).reshape(
        B, ema_model.Cvae, ema_model.patch_nums[-1], ema_model.patch_nums[-1]
    )
    f_hat += h_BChw_final

    for b in ema_model.blocks:
        b.attn.kv_caching(False)
        
    images = ema_model.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
    return images

def prepare_embeds(
    text_model,
    text_tokenizer,
    ema_model,
    prompts,
    cfg,
    device
):
    
    text_model.to(device).eval()

    _, context_mask, context_position_ids, context_tensor = encode_prompts(
        prompts,
        text_model,
        text_tokenizer,
        cfg.model_params.max_token_length,
        llm_system_prompt,
        cfg.model_params.use_llm_system_prompt,
    ) # tokenize prompts with text model

    text_model.to("cpu")
    
    cond_BD = ema_model.context_embed(
        ema_model.context_norm(
            torch.cat((context_tensor, torch.full_like(context_tensor, fill_value=0.0)), dim=0)
        )
    ) # embed the prompt tokens 
    sos = cond_BD.clone()
    context_position_ids = torch.cat(
        (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
        dim=0,
    ) # position ids for prompt tokens

    B = context_mask.shape[0]
    context_mask = torch.cat(
        (context_mask, torch.full_like(context_mask, fill_value=0))
    )
    context_mask[B:, 0] = 1 

    if ema_model.pos_1LC is not None:
        lvl_pos = ema_model.lvl_embed(ema_model.lvl_1L) + ema_model.pos_1LC
    else:
        lvl_pos = ema_model.lvl_embed(ema_model.lvl_1L)

    if ema_model.pos_start is not None:
        next_token_map = (
            sos.expand(2 * B, ema_model.first_l, -1)
            + ema_model.pos_start.expand(2 * B, ema_model.first_l, -1)
            + lvl_pos[:, : ema_model.first_l]
        )
    else:
        next_token_map = (
            sos.expand(2 * B, ema_model.first_l, -1) + lvl_pos[:, : ema_model.first_l]
        )

    return context_tensor, context_position_ids, context_mask, cond_BD, lvl_pos, next_token_map, sos, B