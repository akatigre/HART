import os
import copy
import time
import torch
import numpy as np
from PIL import Image 
from tqdm import trange
from pathlib import Path
from torchvision.utils import make_grid

from utils import set_seed, load_metadata
from transformers import AutoTokenizer, AutoModel
from hart.utils import encode_prompts, llm_system_prompt
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I

import hydra
import decode
import json
from generate import generate
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
    # Create folder to save images
    if cfg.teacher_force:
        folder_name = f"reconstructed/{cfg.model_params.model_name}/{cfg.decode}{cfg.cfg_scale}_tf{cfg.teacher_force_upto*100}"
    else:
        folder_name = f"generated/{cfg.model_params.model_name}/{cfg.decode}{cfg.cfg_scale}"

    if cfg.update_logits:
        assert cfg.decode == "vanilla", "Update logits is only supported for vanilla decode"
        folder_name += f"_update_logits"

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

    decode_func = getattr(decode, f"{cfg.decode}_decode")
    val_prompts, metadatas = load_metadata(cfg)
    categories = val_prompts.get("categories", None)
    batch_size = 1
    N = len(val_prompts['prompts'])
    # load from users passed arguments

    for start_idx in trange(0, N, batch_size):
        gt_path = None
        if cfg.benchmark.name=="geneval":
            prompts = val_prompts['prompts'][start_idx: start_idx + batch_size]
            names = val_prompts['name'][start_idx: start_idx + batch_size]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]
            metas = metadatas[start_idx: start_idx + batch_size]
            for save, metadata in zip(save_path[::4], metas[::4]):
                os.makedirs(save.parent, exist_ok=True)
                with open(os.path.join(save.parent, "metadata.jsonl"), "w") as fp:
                    json.dump(metadata, fp)

        elif cfg.benchmark.name=="dpgbench":
            prompts = val_prompts['prompts'][start_idx: start_idx + batch_size]
            names = val_prompts['name'][start_idx: start_idx + batch_size]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / name for name in names if not (Path(cfg.benchmark.outdirs) / folder_name / name).exists()]

        elif cfg.benchmark.name=="mjhq":
            cats = categories[start_idx: start_idx + batch_size] if categories is not None else None
            gt_path = [Path(cfg.benchmark.outdirs).parent / 'root' / cat / name for cat, name in zip(cats, names)]
            save_path = [Path(cfg.benchmark.outdirs) / folder_name / cat / name for cat, name in zip(cats, names) if not (Path(cfg.benchmark.outdirs) / folder_name / cat / name).exists()]
            for save in save_path:
                os.makedirs(save.parent, exist_ok=True)
        else:
            raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        
        if not len(save_path):
            continue
        start_time = time.time()
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
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
                
                sos = cond_BD = ema_model.context_embed(
                    ema_model.context_norm(
                        torch.cat((context_tensor, torch.full_like(context_tensor, fill_value=0.0)), dim=0)
                    )
                ) # embed the prompt tokens 

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
            
                images = generate(sos, B, 
                                  ema_model, 
                                  quantizer = ema_model.vae_quant_proxy[0],
                                  context_tensor = context_tensor, 
                                  context_position_ids = context_position_ids, 
                                  context_mask = context_mask, 
                                  cond_BD = cond_BD, 
                                  lvl_pos = lvl_pos, 
                                  next_token_map = next_token_map, 
                                  rng = None, 
                                  nonmyopic = cfg.nonmyopic, 
                                  cfg = cfg.model_params, 
                                  decode_func = decode_func, 
                                  cfg_scale = cfg.cfg_scale,
                                  update_logits = cfg.update_logits,
                                  )
        end_time = time.time()
        images = (images.clone().cpu() * 255.0)

        for i in range(len(save_path)):
            if not save_path[i].parent.exists():
                save_path[i].parent.mkdir(parents=True, exist_ok=True)
                
        log.info(f"Time taken: {end_time - start_time} seconds")
        if cfg.benchmark.name=="dpgbench":
            per_prompt_images.extend([image for image in images])
            for img_idx in range(0, len(per_prompt_images), cfg.benchmark.batch):
                images = make_grid(per_prompt_images[img_idx: img_idx + cfg.benchmark.batch], nrow=2)
                images = images.permute(1, 2, 0).cpu().numpy().astype('uint8')
                images = Image.fromarray(images)
                save_path[img_idx].parent.mkdir(parents=True, exist_ok=True)
                images.save(save_path[img_idx])
            per_prompt_images = []
        else:
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            for save_at, image in zip(save_path, pil_images):
                save_at.parent.mkdir(parents=True, exist_ok=True)
                image.save(save_at)

        log.info(f"Generated {prompts}, saved into {save_path}")
        break

if __name__ == "__main__":
    try:
        main()
    except:
        log.exception("An error occurred")