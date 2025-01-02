
from PIL import Image
from typing import List
from generate import residual_stage
import torch
import torch.nn.functional as F
from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
from hart.modules.models.autoencoder import HARTHybridQuantizer, HARTAutoEncoder

from decode import cfg_decode
import types


def change_quant(quantizer: HARTHybridQuantizer):
    def idxBl_to_var_input(
        self, gt_ms_idx_Bl: List[torch.Tensor], logits_list: List[torch.Tensor], patch_nums=None
    ) -> torch.Tensor:
        if patch_nums is None:
            patch_nums = self.v_patch_nums
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = patch_nums[-1]
        SN = len(patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (0 <= self.prog_si - 1 < si):
                break  # progressive training: not supported yet, prog_si always -1
            idx_onehot = torch.nn.functional.one_hot(gt_ms_idx_Bl[si], num_classes=logits_list[si].shape[-1]).float() + logits_list[si].detach() - logits_list[si] # logits_BlV_ is a perturbed logits with epsilon applied
            h_BChw = F.interpolate(
                # last stage might be continuous
                (
                    idx_onehot @ self.embedding.weight
                    if len(gt_ms_idx_Bl[si].shape) == 2
                    else gt_ms_idx_Bl[si]
                )
                .transpose_(1, 2)
                .view(B, C, pn_next, pn_next),
                size=(H, W),
                mode="bicubic",
            )
            
            f_hat = f_hat + self.quant_resi[si / (SN - 1)](h_BChw)
            pn_next = patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
                .view(B, C, -1)
                .transpose(1, 2)
            )
        return (
            torch.cat(next_scales, dim=1) if len(next_scales) else None
        )  # cat BlCs to BLC, this should be float32
    quantizer.idxBl_to_var_input = types.MethodType(idxBl_to_var_input, quantizer)
    return quantizer
   
def change_get_logits(model: HARTForT2I):
    def _rms_norm(x, weight, eps=1e-06):
        x = x.to(weight.dtype)
        variance = (x * x).mean(-1, keepdim=True)
        s_variance = torch.rsqrt(variance + eps)
        return weight * (x * s_variance)
    
    def get_logits(self, x, cond_BD):
        norm_hidden_states = _rms_norm(x.float(), self.head_nm.ln_wo_grad.weight).float()
        if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            cond_BD = cond_BD.max(1, keepdims=True).values
        scale, shift = model.head_nm.ada_lin(cond_BD).view(-1, 1, 2, model.head_nm.C).unbind(2)
        adain_states = norm_hidden_states.mul(scale.add(1)).add_(shift).float()
        logits_BlV = self.head(adain_states) # 2B, 9450, V
        return logits_BlV
    
    model.get_logits = types.MethodType(get_logits, model)
    return model

def teacher_forced_decoding(
    model: HARTForT2I,
    quantizer: HARTHybridQuantizer,
    auto_encoder: HARTAutoEncoder, 
    patch_nums: List[int],
    logits_list: List[torch.tensor],
    biases_list: List[torch.tensor],
    gt_indices_list: List[torch.tensor],
    cond_BD: torch.tensor,
    cfg_scale: float,
    context_position_ids: torch.tensor,
    context_mask: torch.tensor,
    top_p: float = 0.0,
):
    quantizer = change_quant(quantizer)
    model = change_get_logits(model)
    """
    teacher forced decoding
    gt_idx_Bl: List[torch.Tensor]
        [B x (1*1), B x (2*2), B x (3*3), ..., B x (64*64)]
    model.L: total image tokens (9451)
    """
    #! Test idx to image
    # idxBl_to_img = auto_encoder.idxBl_to_img(gt_indices_list, same_shape=True, last_one=False)
    # pil_images = convert_to_pil(torch.cat(idxBl_to_img, dim=0))
    # for i in range(len(pil_images)):
    #     pil_images[i].save(f"teacher_forced_decoding_{i}.png")
        
    sample_fn = lambda logits: torch.multinomial(logits.softmax(dim=-1).view(-1, model.V), 
        num_samples=1,
        replacement=True,
        generator=None,
        ).view(-1, logits.shape[1], 1)[:, :, 0]
    
    # sampled_idx_Bl = [sample_fn(logits) for si, logits in enumerate(logits_list)]
    # idxBl_to_img = auto_encoder.idxBl_to_img(sampled_idx_Bl, same_shape=True, last_one=False)
    # pil_images = convert_to_pil(torch.cat(idxBl_to_img, dim=0))
    # for i in range(len(pil_images)):
    #     pil_images[i].save(f"teacher_forced_decoding_sampled_{i}.png")

    topk_list = [logits < logits.topk(k=10, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True) for logits in logits_list]
    pert_logits_list = [logits + bias for logits, bias in zip(logits_list, biases_list)]
    pert_idx_Bl = [sample_fn(
        torch.masked_fill(logits, topk, -torch.inf)
        ) for logits, topk in zip(pert_logits_list, topk_list)]
    idxBl_to_img = auto_encoder.idxBl_to_img(pert_idx_Bl, same_shape=True, last_one=False)
    pil_images = convert_to_pil(torch.cat(idxBl_to_img, dim=0))
    for i in range(len(pil_images)):
        pil_images[i].save(f"teacher_forced_decoding_sampled_perturbed_{i}.png")
    
    x_BLCv_wo_first_l = quantizer.idxBl_to_var_input(pert_idx_Bl, pert_logits_list) # B, 9450, 32
    B = cond_BD.shape[0] # 2 x original batch size for CFG
    sos = cond_BD.clone()
    sos = sos.expand(-1, model.first_l, -1)
    x_BLCv_wo_first_l = x_BLCv_wo_first_l.expand(B, -1, -1)
    dtype = x_BLCv_wo_first_l.dtype
    lvl_pos = model.lvl_embed(model.lvl_1L) # 1, 9750, 1536
    with torch.cuda.amp.autocast(enabled=False):
        
        x_BLC = torch.cat((sos, model.word_embed(x_BLCv_wo_first_l.float())), dim=1) # 2B, 9750, 1536
        x_BLC += lvl_pos.expand(B, -1, -1) # 2B, 9750, 1536
    
    attn_bias = model.attn_bias_for_masking
    cond_BD_or_gss = model.shared_ada_lin(cond_BD)
    x_BLC = x_BLC.to(dtype=dtype)
    cond_BD_or_gss = cond_BD_or_gss.to(dtype=dtype)
    attn_bias = attn_bias.to(dtype=dtype)
    
    for i, b in enumerate(model.blocks):
        x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias, si=-1, context_position_ids=context_position_ids, context_mask=context_mask)
    
    last_layer_cond = x_BLC
    #! Equivalent to x_BLC = model.get_logits(x_BLC.float(), cond_BD)
    
    logits_BlV = model.get_logits(last_layer_cond.float(), cond_BD)
    logits_cond, logits_uncond = logits_BlV.chunk(2, dim=0)
    logits_cfg = cfg_decode(
        logit_cond=logits_cond, 
        logit_uncond=logits_uncond, 
        scale=cfg_scale
        ) # B, 9450, V
    logits_cfg = logits_cfg[:, model.context_token - 1:]
    start_ends = [(sum([pn * pn for pn in patch_nums[:i]]), sum([pn * pn for pn in patch_nums[:i+1]])) for i in range(len(patch_nums))]
    logits_cfg_list = [logits_cfg[:, st:ed] for st, ed in start_ends]
    pert_idx_Bl = [sample_fn(
        torch.masked_fill(logits, topk, -torch.inf)
        ) for logits, topk in zip(pert_logits_list, topk_list)]
    pert_idx_Bl = [sample_fn(torch.masked_fill(logits, topk, -torch.inf)) for logits, topk in zip(logits_cfg_list, topk_list)]
    idxBl_to_img = auto_encoder.idxBl_to_img(pert_idx_Bl, same_shape=True, last_one=False)
    pil_images = convert_to_pil(torch.cat(idxBl_to_img, dim=0))
    for i in range(len(pil_images)):
        pil_images[i].save(f"teacher_forced_decoding_soft_sampled_perturbed_{i}.png")
    
    selected_tokens = logits_cfg.max(dim=-1).indices
    selected_tokens = selected_tokens.view(logits_cfg.shape[0], logits_cfg.shape[1], -1)[:, :, 0] # B, 9751
    
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits_cfg.view(-1, logits_cfg.size(-1)), selected_tokens.view(-1))
    return loss

def convert_to_pil(images):
    images = images.add_(1).mul_(0.5)
    images = (images.clone().cpu() * 255.0)
    np_images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")
    pil_images = [Image.fromarray(image) for image in np_images]
    return pil_images