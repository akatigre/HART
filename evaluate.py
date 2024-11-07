
import copy
import json
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

from utils import set_seed, load_metadata
from hart.utils import encode_prompts, llm_system_prompt
from change_hart import change_hart_infer, change_hart_block, change_hart_attn
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


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    log.info(f"Set seed {cfg.seed}")
    model_params = cfg.model_params
    assert model_params.model_name == "hart", "Model name should be hart"
    
    enable_pag = cfg.pag_scale > 0.0
    enable_cfg = cfg.cfg_scale > 1.0
    enable_cd = cfg.cd_beta > 0.0 and cfg.cd_alpha < 1.0
    log.info(f"Enable PAG: {enable_pag}, Enable CFG: {enable_cfg}, Enable CD: {enable_cd}")

    # Create folder to save images
    folder_name = "generated"
    output_path = str(Path(cfg.benchmark.outdirs) / cfg.model_params.model_name)
    if enable_pag: 
        folder_name += f"_pag:{cfg.pag_scale}_layer:{cfg.layer_types}"
        output_path += f"_PAG_{cfg.pag_scale}"
    if enable_cfg: 
        folder_name += f"_cfg{cfg.cfg_scale}_{cfg.dynamic_scale}"
        output_path += f"_CFG_{cfg.cfg_scale}_{cfg.dynamic_scale}"
    if enable_cd: 
        folder_name += f"_cd{cfg.cd_alpha}:{cfg.cd_beta}"
        output_path += f"_CD_{cfg.cd_alpha}:{cfg.cd_beta}"

    device = torch.device("cuda")
    model = AutoModel.from_pretrained(cfg.model_params.model_path) # HARTForT2I
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

    #! Change layer forward functions to support PAG
    decoder_layers = ema_model.blocks
    
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

    batch_size = cfg.benchmark.batch
    val_prompts, metadatas = load_metadata(cfg)
    categories = val_prompts.get("categories", None)
    # load from users passed arguments
    start_time = time.time()
    for idx, (prompt, name) in tqdm(enumerate(zip(val_prompts['prompts'], val_prompts['name'])), desc="Generating images"):
        
        cat = categories[idx] if categories is not None else None
        outpath = Path(output_path) / name if cat is None else Path(output_path) / cat / name
        if cfg.benchmark.name=="geneval":
            os.makedirs(outpath, exist_ok=True)
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadatas[idx], fp)

            if len(os.listdir(outpath)) > batch_size:
                continue

        elif cfg.benchmark.name=="mjhq" or cfg.benchmark.name=="dpgbench":
            os.makedirs(outpath.parent, exist_ok=True)
            if os.path.exists(outpath):
                continue
        else:
            raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        
        log.info(f"With Prompt '{prompt}' generating {batch_size} images")
        prompts = [prompt] * batch_size
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
                )

                text_model.to("cpu")
                torch.cuda.empty_cache()

                mini_batch = 2
                batched_imgs =  []
                assert batch_size < mini_batch or batch_size % mini_batch == 0
                for mini in range(0, batch_size, mini_batch):
                    ctx = context_tensor[mini : mini + mini_batch]
                    ctx_mask = context_mask[mini : mini + mini_batch]
                    ctx_pos_ids = context_position_ids[mini : mini + mini_batch]
                    
                    images, extras, sampled_ids = ema_model.autoregressive_infer_cfg(
                        B = ctx.size(0),
                        label_B = ctx, # 2, 300, d
                        cfg_scale = cfg.cfg_scale,
                        pag_scale = cfg.pag_scale,
                        g_seed = cfg.seed+mini,
                        more_smooth = cfg.model_params.more_smooth,
                        context_position_ids = ctx_pos_ids, # 2, 300 padding on the right [0, ... 39, 40, 40, ..., 40]
                        context_mask = ctx_mask, # 2, 300 [T, ..., T, F, ...., F]
                        dynamic_scale = cfg.dynamic_scale,
                        cd_alpha = cfg.cd_alpha,
                        cd_beta = cfg.cd_beta,
                    )

                    images = (images.clone().cpu() * 255.0) 
                    batched_imgs.append(images)
                images = torch.cat(batched_imgs, dim=0)

        if cfg.benchmark.name=="geneval":
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]
            sample_count = 0
            for sample in pil_images:
                sample.save(os.path.join(outpath, f"{sample_count:05}.png"))
                sample_count += 1
            # (Path(outpath) / str(sample_count)).mkdir(exist_ok=True)
            # torch.save(extras, Path(outpath) / str(sample_count) / "extras.pt")
            # torch.save(sampled_ids, Path(outpath) / str(sample_count)/ "generated_tokens.pt")
        elif cfg.benchmark.name=="dpgbench":
            from torchvision.utils import make_grid
            images = make_grid(images, nrow=2) # BCHW
            images = images.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            images = Image.fromarray(images)
            images.save(outpath)
            # (Path(outpath.parent) / outpath.stem).mkdir(exist_ok=True)
            # torch.save(extras, Path(outpath.parent) / outpath.stem / "extras.pt")
            # torch.save(sampled_ids, Path(outpath.parent) / outpath.stem / "generated_tokens.pt")
        elif cfg.benchmark.name=="mjhq":
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            images = Image.fromarray(images[0])
            images.save(outpath)
            # (Path(outpath.parent) / outpath.stem).mkdir(exist_ok=True)
            # torch.save(extras, Path(outpath.parent) / outpath.stem / "extras.pt")
            # torch.save(sampled_ids, Path(outpath.parent) / outpath.stem / "generated_tokens.pt")
        else:
            raise ValueError(f"benchmark name {cfg.benchmark.name} not supported.")
        log.info(f"Generated image saved at {outpath}")

    total_time = time.time() - start_time
    log.info(f"Total time taken: {total_time} Prompt numbers: {len(val_prompts)}")

if __name__ == "__main__":
    try:
        main()
    except:
        log.exception("An error occurred")