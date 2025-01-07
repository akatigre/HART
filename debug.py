import os
import copy
import torch
from PIL import Image 
from torch.nn import functional as F

from utils import set_seed
from transformers import AutoTokenizer, AutoModel

from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I

import hydra
from decode import cfg_decode
from token_embed import prepare_embeds
from yjk import soft_forward, soft_forward_all, convert_to_pil
from omegaconf import DictConfig
from einops import rearrange
import logging
from rich.logging import RichHandler 
from yjk import generate_img_from_idxs, split_into_patches
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")
log.setLevel(logging.INFO)


@hydra.main(config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    log.info(f"Set seed {cfg.seed}")
    model_params = cfg.model_params
    assert model_params.model_name == "hart", "Model name should be hart"
    folder_name = "debug"
    device = torch.device("cuda")
    
    model = HARTForT2I.from_pretrained(cfg.model_params.model_path)
    model = model.to(device)
    model.eval()

    if cfg.model_params.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(cfg.model_params.model_path, "ema_model.bin"))
        )
        ema_model.eval()
        del model
        torch.cuda.empty_cache()
    quantizer = ema_model.vae_quant_proxy[0]

    text_tokenizer = AutoTokenizer.from_pretrained(cfg.model_params.text_model_path)
    text_model = AutoModel.from_pretrained(cfg.model_params.text_model_path)
    prompts = ["a cow", "a bicycle", "a clock"]
    idx = cfg.idx
    
    for idx in range(len(prompts)):
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            context_tensor, context_position_ids, context_mask, cond_BD, lvl_pos, B, sos = prepare_embeds(
                text_model,
                text_tokenizer, 
                ema_model, 
                [prompts[idx]], 
                cfg, 
                device
            )
            del text_tokenizer, text_model
            torch.cuda.empty_cache()
            with torch.no_grad():
                
                images, logits_list = soft_forward(
                    B, 
                    ema_model = ema_model, 
                    quantizer = quantizer, 
                    context_position_ids = context_position_ids, 
                    context_mask = context_mask, 
                    cond_BD = cond_BD,
                    sos = sos, 
                    cfg_scale=cfg.cfg_scale,
                    soft=False,
                    logits_list=None,
                )
                # logits, images = soft_forward_all(
                #         pert_logits_list=logits_list,
                #         cond_BD=cond_BD,
                #         sos=sos,
                #         model=ema_model,
                #         auto_encoder=ema_model.vae_proxy[0],
                #         quantizer=quantizer,
                #         context_position_ids=context_position_ids,
                #         context_mask=context_mask,
                #     )
            torch.cuda.empty_cache()            
            if cfg.yjk.do_langevin_dynamics:
            
                biases_list = [
                        torch.nn.Parameter(
                            torch.randn_like(
                                logits_list[i],
                                ) * cfg.yjk.start_noise
                            ) for i in range(len(logits_list))
                            ]     
                optim = torch.optim.AdamW(
                    biases_list,
                    lr = cfg.yjk.stepsize,
                    weight_decay = cfg.yjk.weight_decay,
                )
                best_loss = torch.inf
                for child_name, child_module in ema_model.named_children():
                    for name, param in child_module.named_parameters():
                        param.requires_grad = True
                
                target_logits = torch.cat(logits_list, dim=1)
                target_labels = torch.argmax(target_logits, dim=-1)
                target_labels = target_labels.view(target_logits.shape[0], target_logits.shape[1], -1)[:, :, 0] # B, 9451
                target_labels = target_labels.view(-1)
                cfg_scale = cfg.cfg_scale
                topk = cfg.yjk.k_filter
                for it in range(cfg.yjk.update_iters):
                    optim.zero_grad()
                    logits_list = [logits.detach() for logits in logits_list]
                    cond_BD = cond_BD.detach()
                    context_position_ids = context_position_ids.detach()
                    context_mask = context_mask.detach()
                    
                    # mask_list = [get_topk_mask(logits, topk) for logits in logits_list] # B, 9451, 4096: -torch.inf for small values, 0 for topk values
                    # mask = torch.cat(mask_list, dim=1)
                    weight_list = [(cur_len + 1) / len(ema_model.patch_nums) for cur_len in range(len(ema_model.patch_nums))]
                    pert_logits_list = [logits + bias * weight for logits, bias, weight in zip(logits_list, biases_list, weight_list)]

                    ppl_images, logits_cfg_list = soft_forward(
                        B = B,
                        ema_model = ema_model,
                        quantizer = quantizer,
                        context_position_ids = context_position_ids,
                        context_mask = context_mask,
                        cond_BD = cond_BD,
                        sos = sos,
                        cfg_scale = cfg_scale,
                        soft = True,
                        logits_list = pert_logits_list,
                    )
                    
                    """ 
                    Equivalent to 
                    shift_logits = lm_logits[..., :-1, :].contiguous() # 4, 9, ..., 4096
                    shift_labels = labels[..., 1:].contiguous() # 4, 9, ..., 4096
                    """
                    logits_cfg = [
                        F.interpolate(logits.permute(0, 2, 1).reshape(B, -1, ema_model.patch_nums[si+1], ema_model.patch_nums[si+1]), size=(ema_model.patch_nums[si], ema_model.patch_nums[si]), mode="bicubic",) 
                        for si, logits in enumerate(logits_cfg_list[1:-1])
                        ] + [
                            logits_cfg_list[-1].permute(0, 2, 1).reshape(B, -1, ema_model.patch_nums[-1], ema_model.patch_nums[-1])
                            ]
                    
                    logits_cfg = [rearrange(logits, "B C h w -> B (h w) C") for logits in logits_cfg]
                    logits_cfg = torch.cat(logits_cfg, dim=1) # B, 9451, 4096
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(logits_cfg.view(-1, logits_cfg.size(-1)), target_labels)
                    loss.backward()
                    optim.step()
                    
                    log.info(f"Iteration {it} loss: {loss.item()} bias grad: {sum([bias.grad.sum().item() for bias in biases_list])}") # biases_list[-1].grad = None
                    if loss < best_loss:
                        best_loss = loss
                        output_idxs = torch.argmax(logits_cfg, dim=-1)
                        output_idxs = output_idxs.view(logits_cfg.shape[0], logits_cfg.shape[1], -1)[:, :, 0] # B, 9751
                        output_idxs_list = split_into_patches(output_idxs, ema_model.patch_nums)
                    if cfg.yjk.add_noise:
                        noise = [torch.normal(mean=0.01, std=0.01, size=biases_list[idx].shape,
                                        device='cuda', requires_grad=False, dtype=torch.float16) for idx in range(len(biases_list))]
                        for i in range(len(biases_list)):
                            biases_list[i].data = biases_list[i].data + noise[i]
                pert_img = generate_img_from_idxs(output_idxs_list, ema_model.vae_proxy[0])
                save_as = f"{prompts[idx]}_topk_{cfg.yjk.k_filter}_lr{cfg.yjk.stepsize}_decay{cfg.yjk.weight_decay}_start_noise{cfg.yjk.start_noise}"
                save_as += "_noised" if cfg.yjk.add_noise else ""
                pert_img.save(f"debug/{save_as}.png")
                
def constraint_loss_by_ppl(logits, cs_onehot, cs_ids, logits_t):
    device = logits.device
    log_ps = logits.log_softmax(-1).unsqueeze(2)

    cs_onehot_ = cs_onehot.unsqueeze(1).type(torch.FloatTensor).to(device)
    ps_t = logits_t.softmax(-1).unsqueeze(2)
    ppl_max_idx = (ps_t * cs_onehot_).argmax(1)  # [batch_size, num_cs, vocab_size]
    ppl_max_idx_onehot = torch.zeros_like(log_ps * cs_onehot_).scatter_(1, ppl_max_idx.unsqueeze(1), cs_onehot_)

    constraint_max_log_ps_ = (log_ps * ppl_max_idx_onehot).sum(1).sum(-1)  # shape: [batch_size, num_cs]

    ## Mask
    log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
    cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_cs, length]
    mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_cs]
    mask = (mask < 1).type(torch.FloatTensor)
    mask = mask.to(device)

    loss = - (constraint_max_log_ps_ * mask).sum()

    if mask.sum() != 0:
        loss = loss / mask.sum()
    else:
        loss = 0

    return loss

def cross_entropy_loss(logits, targets):
    probs = F.softmax(logits, dim=-1)
        
    # Take log of probabilities
    log_probs = torch.log(probs + 1e-10)  # Added epsilon to avoid log(0)
    
    # Select the log-probabilities for the target classes
    # targets should be class indices (not one-hot encoded)
    log_probs_for_targets = log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
    
    # Compute the negative log-likelihood loss
    loss = -log_probs_for_targets.squeeze(-1)  # Squeeze removes the last dimension
    
    # Return the mean loss over the batch
    return loss.mean()
if __name__ == "__main__":
    try:
        main()
    except:
        log.exception("An error occurred")