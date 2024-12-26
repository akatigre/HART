import os
import copy
import torch
from PIL import Image 


from utils import set_seed
from transformers import AutoTokenizer, AutoModel

from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I

import hydra
from decode import cfg_decode
from generate import generate_images, prepare_embeds
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
        
        if cfg.yjk.do_langevin_dynamics:
            bias = torch.nn.Parameter(torch.randn_like(lvl_pos) * 0.5)
            optimizer = torch.optim.Adam([bias], lr=0.025)
            max_iterations = cfg.yjk.update_iters
            for step in range(max_iterations):
                optimizer.zero_grad()
                images, onehot_global, logits_global = generate_images(
                    sos, 
                    B, 
                    ema_model, 
                    quantizer = ema_model.vae_quant_proxy[0],
                    context_tensor = context_tensor, 
                    context_position_ids = context_position_ids, 
                    context_mask = context_mask, 
                    cond_BD = cond_BD, 
                    lvl_pos = lvl_pos, 
                    next_token_map = next_token_map, 
                    rng = None, 
                    cfg = cfg.model_params, 
                    decode_func = cfg_decode, 
                    cfg_scale = cfg.cfg_scale,
                    soft = True,
                    epsilon = bias
                )
                loss_fnc = torch.nn.CrossEntropyLoss()  # logits_global vs onehot_global
                loss = loss_fnc(logits_global.detach(), onehot_global)
                loss.backward()
                optimizer.step()
                log.info(f"Step {step} loss: {loss.item()}")
        else:
            images = generate_images(
                sos, 
                B, 
                ema_model, 
                quantizer = ema_model.vae_quant_proxy[0],
                context_tensor = context_tensor, 
                context_position_ids = context_position_ids, 
                context_mask = context_mask, 
                cond_BD = cond_BD, 
                lvl_pos = lvl_pos, 
                next_token_map = next_token_map, 
                rng = None, 
                cfg = cfg.model_params, 
                decode_func = cfg_decode, 
                cfg_scale = cfg.cfg_scale,
                soft = True, # False # True to test soft sampling
                )

    images = (images.clone().cpu() * 255.0)
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(f"{folder_name}/{prompts[idx]}_cfg_{cfg.cfg_scale}_original_soft.png")


if __name__ == "__main__":
    try:
        main()
    except:
        log.exception("An error occurred")