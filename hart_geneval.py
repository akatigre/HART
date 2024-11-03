
import copy
import os
import time
import torch
from tqdm import tqdm, trange
from PIL import Image 
import numpy as np
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
os.system("export PYTHONPATH='$PYTHONPATH:/home/server08/yoonjeon_workspace/MMAR/hart'")

from utils import set_seed, save_images
from hart.utils import encode_prompts, llm_system_prompt
from change_hart import change_hart_infer, change_hart_block, change_hart_attn

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from rich.logging import RichHandler
from pathlib import Path
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")
log.setLevel(logging.INFO)

wandb.offline = True
@hydra.main(config_path=".", config_name="hart_config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    log.info(f"Set seed {cfg.seed}")
    enable_pag = cfg.pag_scale > 0.0
    enable_cfg = cfg.cfg > 1.0
    enable_cd = cfg.cd_beta < 1.0
    log.info(f"Enable PAG: {enable_pag}, Enable CFG: {enable_cfg}, Enable CD: {enable_cd}")

    # Create folder to save images
    folder_name = "generated"
    if enable_pag: 
        folder_name += f"_pag:{cfg.pag_scale}_layer:{cfg.layer_types}"
        cfg.outdir += f"_PAG_{cfg.pag_scale}"
    if enable_cfg: 
        folder_name += f"_cfg{cfg.cfg}"
        cfg.outdir += f"_CFG_{cfg.cfg}"
    if enable_cd: 
        folder_name += f"_cd{cfg.cd_beta}"
        cfg.outdir += f"_CD_{cfg.cd_beta}"

    import json
    with open(cfg.metadata_file) as f:
        metadatas = [json.loads(line) for line in f]

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(code_dir=os.getcwd())
    )
    wandb.run.log_code("/home/server08/yoonjeon_workspace/MMAR/hart", include_fn=lambda path: path.endswith(".py"))

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(cfg.model_path) # HARTForT2I
    model = model.to(device)
    model.eval()

    if cfg.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(cfg.model_path, "ema_model.bin"))
        )
        ema_model.eval()
        del model
        torch.cuda.empty_cache()

    text_tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_path)
    text_model = AutoModel.from_pretrained(cfg.text_model_path)

    #! Change layer forward functions to support PAG
    decoder_layers = ema_model.blocks
    
    cfg.batch_size = cfg.batch_size
    if cfg.layer_types=="all":
        layer_idxs = range(len(decoder_layers))
    elif cfg.layer_types=="early":
        layer_idxs = range(len(decoder_layers) // 3)
    elif cfg.layer_types=="middle":
        layer_idxs = range(len(decoder_layers) // 3, 2 * len(decoder_layers) // 3)
    elif cfg.layer_types=="late":
        layer_idxs = range(2 * len(decoder_layers) // 3, len(decoder_layers))
    log.info(f"Total layers : {len(decoder_layers)}, Changing layers: {layer_idxs}")

    ema_model = change_hart_infer(ema_model)
    for b_idx, block in enumerate(decoder_layers):
        block = change_hart_block(block)
        if b_idx in layer_idxs:
            block.attn = change_hart_attn(block.attn)
    log.info("Change HART model for PAG")
    
    mini_batch_size = 2
    assert cfg.batch_size % mini_batch_size == 0
    start_time = time.time()
    for p_idx, metadata in tqdm(enumerate(metadatas)):
        
        outpath = os.path.join(cfg.outdir, f"{p_idx:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        log.info(f"With Prompt '{prompt}' generating {cfg.batch_size} images")

        prompts = [prompt] * cfg.batch_size

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
                text_model.to(device).eval()

                _, context_mask, context_position_ids, context_tensor = encode_prompts(
                    prompts,
                    text_model,
                    text_tokenizer,
                    cfg.max_token_length,
                    llm_system_prompt,
                    cfg.use_llm_system_prompt,
                )

        text_model.to("cpu")
        torch.cuda.empty_cache()
        sample_count = 0
        per_prompt_images = []
        for idx in trange(0, cfg.batch_size, mini_batch_size):
            c_mask, c_id, c_tensor = context_mask[idx : idx + mini_batch_size], context_position_ids[idx : idx + mini_batch_size], context_tensor[idx : idx + mini_batch_size]

            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
                    
                    images, logits, idx_Bl = ema_model.autoregressive_infer_cfg(
                        B = c_tensor.size(0),
                        label_B = c_tensor, # 2, 300, d
                        cfg_scale = cfg.cfg,
                        pag_scale = cfg.pag_scale,
                        g_seed = cfg.seed + idx,
                        more_smooth = cfg.more_smooth,
                        context_position_ids = c_id, # 2, 300 padding on the right [0, ... 39, 40, 40, ..., 40]
                        context_mask = c_mask, # 2, 300 [T, ..., T, F, ...., F]
                    )
                    per_prompt_images.append(images.clone().permute(0, 2, 3, 1).cpu())

        images = torch.cat(per_prompt_images, dim=0)
        images *= 255.0
        images = images.numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        for sample in pil_images:
            sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
            sample_count += 1

        rows = [[cfg.cfg, cfg.pag_scale, wandb.Image(image), prompt] for i, image in enumerate(pil_images)]
        columns = ["cfg", "pag", "image", "text"]

        wandb.log({"images": wandb.Table(data=rows, columns=columns)}, step=p_idx)
        # Log extras and generated_tokens to wandb
        out_dir = Path(f"./outputs/{str(wandb.run.id)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(logits, out_dir / "extras.pt")
        torch.save(idx_Bl, out_dir / "generated_tokens.pt")
        wandb.save(str(out_dir) + "/*", policy="end")

    total_time = time.time() - start_time


if __name__ == "__main__":
    try:
        main()
    except:
        log.exception("An error occurred")