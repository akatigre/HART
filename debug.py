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
from generate import generate_images
from yjk import teacher_forced_decoding
from omegaconf import DictConfig

import logging
from rich.logging import RichHandler 
from generate import residual_stage
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
    gen_images = True
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
        context_tensor, context_position_ids, context_mask, cond_BD, lvl_pos, B, sos= prepare_embeds(
            text_model,
            text_tokenizer, 
            ema_model, 
            [prompts[idx]], 
            cfg, 
            device
        )
        del text_model, text_tokenizer
        torch.cuda.empty_cache()
        with torch.no_grad():
            images, hidden_state_list, logits_list, indices_list = generate_images( 
                B, 
                ema_model = ema_model, 
                quantizer = quantizer, 
                context_position_ids = context_position_ids, 
                context_mask = context_mask, 
                cond_BD = cond_BD, 
                lvl_pos = lvl_pos, 
                sos = sos, 
                rng = None, 
                decode_func = cfg_decode, 
                cfg_scale = cfg.cfg_scale,
                )
        images = (images.clone().cpu() * 255.0)
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        img_name = f"{prompts[idx]}_cfg_{cfg.cfg_scale}_original"
        pil_images[0].save(f"{folder_name}/{img_name}.png")
        del images, pil_images
        torch.cuda.empty_cache()
        
        if cfg.yjk.do_langevin_dynamics:
            biases_list = [
                    torch.nn.Parameter(
                        torch.randn_like(
                            logits_list[i]
                            ) * cfg.yjk.start_noise
                        ) for i in range(len(logits_list))
                        ]     
            optim = torch.optim.AdamW(
                biases_list,
                lr = cfg.yjk.stepsize,
                weight_decay = cfg.yjk.weight_decay,
            )
            best_loss = torch.inf
            for it in range(cfg.yjk.update_iters):
                optim.zero_grad()
                loss = teacher_forced_decoding(
                    model = ema_model, 
                    quantizer = ema_model.vae_quant_proxy[0],
                    auto_encoder = ema_model.vae_proxy[0],
                    patch_nums = ema_model.patch_nums,
                    logits_list = [logits.detach() for logits in logits_list],
                    biases_list = biases_list,
                    gt_indices_list = [indices.detach() for indices in indices_list],
                    cond_BD = cond_BD.detach(),
                    cfg_scale = cfg.cfg_scale,
                    context_position_ids = context_position_ids.detach(),
                    context_mask = context_mask.detach(),
                    topk = cfg.yjk.topk,
                )
                if loss < best_loss:
                    best_loss = loss
                    
                loss.backward()
                log.info(f"Iteration {it} loss: {loss.item()} bias grad: {biases_list[0].grad.sum().item()}")
                torch.cuda.empty_cache()
                optim.step()


if __name__ == "__main__":
    try:
        main()
    except:
        log.exception("An error occurred")