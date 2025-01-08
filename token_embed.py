from hart.utils import encode_prompts, llm_system_prompt
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
import torch

def prepare_embeds(
    text_model,
    text_tokenizer,
    ema_model: HARTForT2I,
    prompts,
    cfg
    ):
    

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
    )
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
        sos = (
            sos.expand(2 * B, ema_model.first_l, -1)
            + ema_model.pos_start.expand(2 * B, ema_model.first_l, -1)
            + lvl_pos[:, : ema_model.first_l]
        )
        
    else:
        sos = sos.expand(2 * B, ema_model.first_l, -1) + lvl_pos[:, : ema_model.first_l]

    return context_tensor, context_position_ids, context_mask, cond_BD, lvl_pos, B, sos
