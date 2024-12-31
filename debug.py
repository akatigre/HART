import os
import copy
import torch
from PIL import Image 


from utils import set_seed
from transformers import AutoTokenizer, AutoModel

from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I

import hydra
from decode import cfg_decode
from token_embed import prepare_embeds
from generate import generate_images
from yjk import image_decode, soft_forward_cfg
from omegaconf import DictConfig

import logging
from rich.logging import RichHandler 

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")
log.setLevel(logging.INFO)


@hydra.main(config_path="../configs", config_name="config")
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
    prompts = ["a bench", "a cow", "a bicycle", "a clock"]
    idx = 3
    with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
        context_tensor, context_position_ids, context_mask, cond_BD, lvl_pos, next_token_map, sos, B = prepare_embeds(
            text_model,
            text_tokenizer, 
            ema_model, 
            [prompts[idx]], 
            cfg, 
            device
        )
        del text_model, text_tokenizer
        torch.cuda.empty_cache()
        images, hidden_state_list, logits_list, filter_list = generate_images(
            sos, 
            B, 
            ema_model = ema_model, 
            quantizer = quantizer,
            context_tensor = context_tensor, 
            context_position_ids = context_position_ids, 
            context_mask = context_mask, 
            cond_BD = cond_BD, 
            lvl_pos = lvl_pos, 
            next_token_map = next_token_map, 
            rng = None, 
            decode_func = cfg_decode, 
            cfg_scale = cfg.cfg_scale,
            ) 
        images = (images.clone().cpu() * 255.0)
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        img_name = f"{prompts[idx]}_cfg_{cfg.cfg_scale}_original"
        pil_images[0].save(f"{folder_name}/{img_name}_testing.png")

        if cfg.yjk.do_langevin_dynamics:
            torch.cuda.empty_cache()
            hidden_state_init = torch.cat(hidden_state_list[1:], dim=1) # B, 9450 (ema_model.L), 1536 (ema_model.Cvae)
            # logits_init = torch.cat(logits_list[1:], dim=1) # B, 9450 (ema_model.L), 4096 (ema_model.V)
            topk_init = torch.cat(filter_list[1:], dim=1) # B, 9450 (ema_model.L), 4096 (ema_model.V)
            
            bias = torch.nn.Parameter(
                torch.randn(
                    2 * B, ema_model.L - 1, ema_model.C, device=device
                    ) * cfg.yjk.start_noise
                ) # patch_nums: 1, 2, 3, 4, 5, ..., 64
            optim = torch.optim.AdamW(
                [bias], 
                lr = cfg.yjk.stepsize,
                weight_decay = cfg.yjk.weight_decay,
            )
            for it in range(cfg.yjk.update_iters):
                optim.zero_grad()
                loss, selected_tokens = soft_forward_cfg(
                    ema_model, 
                    epsilon = bias, 
                    hidden_states = hidden_state_init.detach().clone().to(device),
                    logits = None, # logits_init.detach().clone().to(device), 
                    topk_mask = topk_init.detach().clone().to(device),
                    context_position_ids = context_position_ids.detach().clone().to(device),
                    context_mask = context_mask.detach().clone().to(device),
                    cond_BD = cond_BD.detach().clone().to(device),
                    cfg_scale = cfg.cfg_scale,
                    quantizer = quantizer,
                )
                loss.backward()
                optim.step()
                log.info(f"Iteration {it} loss: {loss.item()}")
            bias = bias.detach().cpu()
            
            images = (images.clone().cpu() * 255.0)
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            img_name = f"{prompts[idx]}_cfg_{cfg.cfg_scale}"
            img_name += "_energy" if cfg.yjk.do_langevin_dynamics else "_original"
            pil_images[0].save(f"{folder_name}/{img_name}.png")


if __name__ == "__main__":
    try:
        main()
    except:
        log.exception("An error occurred")