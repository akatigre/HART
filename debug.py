import os
import copy
import torch

from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import Dataset
from custom_model import SoftModel
import hydra
from token_embed import prepare_embeds
from yjk import soft_forward
from utils import set_seed
from omegaconf import DictConfig
import logging
from torch import optim
from custom_model import CustomTrainer
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger("rich")
log.setLevel(logging.INFO)


# @hydra.main(config_path="./configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    log.info(f"Set seed {cfg.seed}")
    model_params = cfg.model_params

    device = torch.device("cuda")
    
    model = AutoModel.from_pretrained(cfg.model_params.model_path)
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
                text_model.to(ema_model.device),
                text_tokenizer, 
                ema_model, 
                [prompts[idx]], 
                cfg
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
                generate_img_from_idxs(images)
            torch.cuda.empty_cache()            
            if cfg.yjk.do_langevin_dynamics:
                best_loss = torch.inf
                
                ema_model.train()
                for child_name, child_module in ema_model.named_children():
                    for name, param in child_module.named_parameters():
                        param.requires_grad = True
                        
                with torch.no_grad():
                    target_logits = torch.cat(logits_list, dim=1)
                    target_labels = torch.argmax(target_logits, dim=-1)
                    target_labels = target_labels.view(target_logits.shape[0], target_logits.shape[1], -1)[:, :, 0] # B, 9451
                    target_labels = target_labels.view(-1)
                
                training_args = TrainingArguments(
                    output_dir="./debug",
                    # learning_rate=1e-4,
                    num_train_epochs=10,
                    # weight_decay=0.001,
                    # gradient_checkpointing=True,
                    per_device_train_batch_size=1,
                    fp16=True, 
                    deepspeed="ds_config.json",

                )
                
                
                model = SoftModel(logits_list = logits_list, start_noise = cfg.yjk.start_noise, model = ema_model)
                
                
                dummy_data = Dataset.from_dict({
                    "context_position_ids": [context_position_ids],
                    "context_mask": [context_mask],
                    "cond_BD": [cond_BD],
                    "sos": [sos],
                    "cfg_scale": [cfg.cfg_scale],
                    "labels": [target_labels]
                })
                
                trainer = CustomTrainer( # 
                    model=model,
                    args=training_args,
                    train_dataset=dummy_data,
                    eval_dataset=dummy_data,  # You can use a validation dataset here
                )
                trainer.optimizer = optim.Adam(model.biases_list.parameters(), lr=1e-3)
                trainer.train()
                
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    hydra.initialize(config_path="./configs", version_base="1.1")
    cfg = hydra.compose(config_name="config")
    try:
        main(cfg)
    except Exception as e:
        log.error(f"Error: {e}")
        raise e
