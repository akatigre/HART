import os
import torchvision
from PIL import Image
from collections import defaultdict
import json

import random
import numpy as np
import torch
from transformers import set_seed as hf_set_seed

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)



def save_images(sample_imgs, sample_folder_dir, store_separately, prompts):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images.png"))
        print(f"Example images are saved to {sample_folder_dir}")
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(os.path.join(sample_folder_dir, f"{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


def load_metadata(cfg):
    val_prompts = defaultdict(list)
    
    prompt_path = cfg.benchmark.prompts
    if cfg.benchmark.name=="dpgbench":
        prompt_lists = sorted(os.listdir(prompt_path))
        for p in prompt_lists:
            full_path = os.path.join(prompt_path, p)
            with open(full_path, 'r') as f:
                line = f.read().splitlines()[0]
            val_prompts["name"].append(p.replace("txt", "png"))
            val_prompts["prompts"].append(line)
        metadatas = None
        
    elif cfg.benchmark.name=="geneval":
        with open(prompt_path) as f:
            metadatas = [json.loads(line) for line in f]
        val_prompts["prompts"] = [metadata['prompt'] for metadata in metadatas]
        val_prompts["name"] = [f"{idx:0>5}" for idx in range(len(val_prompts["prompts"]))]
        
    elif cfg.benchmark.name=="mjhq":
        with open(prompt_path, "r") as f:
            metadatas = json.load(f)
        file_names = sorted(list(metadatas.keys()))
        
        val_prompts["name"] = [file_name + ".png" for file_name in file_names]
        val_prompts["prompts"] = [metadatas[filename]["prompt"] for filename in file_names]
        val_prompts["categories"] = [metadatas[filename]["category"] for filename in file_names]
        
    else:
        raise NotImplementedError(f"Unknown benchmark name: {cfg.benchmark.name}")
    return val_prompts, metadatas