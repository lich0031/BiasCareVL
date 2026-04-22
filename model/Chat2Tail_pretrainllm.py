from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from utils.utils import DEFAULT_SEG_Task_TOKEN
from .IMIS.utils import FocalDice_MSELoss
from .IMIS.segment_anything import build_sam_vit_h, build_sam_vit_b

import random

import os
import json
import gc

'''
def compute_mask_loss(pred_masks, gt_masks):
    """
    Compute the mask prediction loss by combining Dice loss and BCE loss.
    pred_masks: [B, n_masks, H, W] raw logits
    gt_masks: [B, n_masks, H, W] binarized ground-truth masks
    """
    # Convert logits to probability maps with sigmoid first.
    pred_probs = torch.sigmoid(pred_masks)
    
    # Dice Loss
    smooth = 1.0
    intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
    dice_loss = 1 - ((2 * intersection + smooth) / (union + smooth))
    dice_loss = dice_loss.mean()
    
    # BCE Loss
    bce_loss = F.binary_cross_entropy(pred_probs, gt_masks)
    
    return dice_loss + bce_loss

def compute_iou_loss(pred_iou, gt_masks, pred_masks):
    """
    Compute the MSE loss for IoU prediction.
    The ground-truth IoU is used as the supervision signal and compared with pred_iou.
    pred_iou: [B, n_masks]
    gt_masks and pred_masks are used to compute the target IoU
    """
    # Convert predicted masks into probability maps.
    pred_probs = torch.sigmoid(pred_masks)
    # Flatten the H and W dimensions to compute IoU for each mask.
    intersection = (pred_probs * gt_masks).sum(dim=(2,3))
    union = (pred_probs + gt_masks).sum(dim=(2,3)) - intersection
    gt_iou = intersection / (union + 1e-6)  # Avoid division by zero.
    # Assume gt_iou has the same shape as pred_iou.
    return F.mse_loss(pred_iou, gt_iou)

def compute_semantic_loss(pred_sem, gt_sem):
    """
    Compute cross-entropy loss for semantic prediction.
    pred_sem: [B, n_masks, num_classes] in logits form
    gt_sem: [B, n_masks] integer class labels for each mask
    """
    B, n_masks, num_classes = pred_sem.shape
    pred_sem_flat = pred_sem.view(-1, num_classes)
    gt_sem_flat = gt_sem.view(-1)
    return F.cross_entropy(pred_sem_flat, gt_sem_flat)
'''
class Chat2TailConfig(PretrainedConfig):
    model_type = "chat2tail"
    def __init__(self, seg_token_idx=None, det_token_idx=None, prompt_token_idx=None, vision_pretrained=None, version=None, image_size=224, out_dim=768, lm_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.seg_token_idx = seg_token_idx
        self.det_token_idx = det_token_idx
        self.prompt_token_idx = prompt_token_idx
        # self.ref_token_idx = ref_token_idx
        # self.end_ref_token_idx = end_ref_token_idx
        
        self.vision_pretrained = vision_pretrained
        self.version = version
        self.image_size = image_size
        self.out_dim = out_dim
        self.lm_weight = lm_weight

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask)
        union = torch.sum(pred) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_loss

class Chat2TailForCausalLMStandard(PreTrainedModel):
    config_class = Chat2TailConfig
    base_model_prefix = "chat2tail"
    def __init__(self, config, tokenizer=None):
        super(Chat2TailForCausalLMStandard, self).__init__(config)
        self.tokenizer = tokenizer
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.imis_loss = FocalDice_MSELoss()
        self.image_size = config.image_size
        self.lm_weight = config.lm_weight
        self.promptype = config.promptype
        
        # Initialize the vision module.
        # self.SEG = self.initialize_seg_modules(pretrain_path=self.config.vision_pretrained)

        # Initialize the language model.
        if "deepseek" in self.config.version:
            from .deepseek_vl2.models import DeepseekVLV2ForCausalLM
            self.VL = DeepseekVLV2ForCausalLM.from_pretrained(self.config.version)
            self.VL.language.resize_token_embeddings(len(tokenizer))
            self.VL.language.enable_input_require_grads()
            self.VL.language.lm_head.weight = self.VL.language.get_input_embeddings().weight

            self.VL.config.use_cache = True
            self.VL.config.output_hidden_states = True
        elif "Qwen" in self.config.version:
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.VL = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.config.version,
                                                                        attn_implementation="flash_attention_2",
                                                                        cache_dir='/mnt/disk2/hwj/Tails/Chat2Tail/cache_dir', torch_dtype="auto"
                                                                    )
            self.VL.language = self.VL.model
            self.VL.vision = self.VL.visual
            self.VL.config.use_cache = False
            if hasattr(self.VL, "enable_input_require_grads"):
                self.VL.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    self.VL.requires_grad_(True)
                self.VL.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        else:
            raise ValueError(f"Unsupported model version: {self.config.version}")

        self.seg_fc = nn.Sequential(
                    nn.Linear(self.VL.language.config.hidden_size, self.config.out_dim),
                    # nn.ReLU(),
                    )
        # self.det_fc = nn.Sequential(
        #             nn.Linear(self.VL.language.config.hidden_size, self.config.out_dim),
        #             # nn.ReLU(),
        #             )
        # self.fusion_fc = nn.Linear(self.config.out_dim, self.config.out_dim)
        
        
        '''newly added 1'''
        self.add11 = nn.Parameter(torch.randn(1, 1, self.VL.language.config.hidden_size))
        self.add12 = nn.MultiheadAttention(self.VL.language.config.hidden_size, 8, batch_first=True)
        ''''''
        # '''newly added 2'''
        # self.add21 = nn.Conv2d(self.SEG.image_encoder.encoder_embed_dim, config.out_dim, kernel_size=1)
        # self.add22 = nn.MultiheadAttention(config.out_dim, 4, batch_first=True)
        # ''''''

        self.post_init()
        print('Using standard pretrain.py')

    def initialize_seg_modules(self, pretrain_path):
        if 'pretrainseg' in pretrain_path:
            print(f"Loading stage 1 SEG weights from: {pretrain_path}")
            SEG = build_sam_vit_b(image_size=self.config.image_size, checkpoint=None) # Pass None initially
            SEG.text_model = nn.Identity() # Replace text model as before
            full_state_dict = torch.load(pretrain_path, map_location="cpu") # Load to CPU first
            
            # Create a new state dict for the SEG model by filtering and removing the 'SEG.' prefix
            seg_state_dict = {}
            for k, v in full_state_dict.items():
                if k.startswith("SEG."):
                    seg_state_dict[k[len("SEG."):]] = v # Remove the 'SEG.' prefix
            
            # Load the filtered state dict into the SEG model
            # Use strict=False to ignore missing keys (like text_model if not saved) or unexpected keys (like VL parts)
            missing_keys, unexpected_keys = SEG.load_state_dict(seg_state_dict, strict=False)
            
            if missing_keys:
                print(f"****** Warning: Missing keys when loading SEG state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"****** Warning: Unexpected keys when loading SEG state_dict: {unexpected_keys}")
                
            print(f'****** Successfully loaded SEG parameters from stage 1 checkpoint.')
            # Clean up memory
            del full_state_dict
            del seg_state_dict
            gc.collect() # Explicitly call garbage collector
            torch.cuda.empty_cache()
        else:
            SEG = build_sam_vit_b(image_size=self.config.image_size, checkpoint=pretrain_path)
            SEG.text_model = nn.Identity()
        return SEG

    def _sample_prompts(self, seg_text_prompt, points, bboxs):
        """
        Randomly drop some non-None inputs among seg_text_prompt, points, and bboxs.
        If more than one input is non-None, randomly set 0 to (count - 1) of them to None,
        while keeping at least one input.
        """
        vars_list = [seg_text_prompt, points, bboxs]
        non_none_indices = [i for i, v in enumerate(vars_list) if v is not None]
        if len(non_none_indices) > 1:
            
            num_to_remove = random.randint(0, len(non_none_indices) - 1)
            indices_to_set_none = random.sample(non_none_indices, num_to_remove)
            for idx in indices_to_set_none:
                if idx == 0:
                    seg_text_prompt = None
                elif idx == 1:
                    points = None
                elif idx == 2:
                    bboxs = None
        return seg_text_prompt, points, bboxs

    def _sample_promptstype(self, prompt_type):
        if not isinstance(prompt_type, str):
            raise TypeError("prompt_type must be a string.")
        if not prompt_type:
            return ""

        chars = list(prompt_type)
        
        if len(chars) <= 1:
            return prompt_type
            
        num_to_keep = random.randint(1, len(chars))
        kept_chars = random.sample(chars, num_to_keep)
        kept_chars.sort(key=lambda char: prompt_type.find(char))
        
        prompt_type_process = "".join(kept_chars)
        return prompt_type_process


    def forward(self, inference=False, **kwargs):

        points = (kwargs.get("point_coords"), kwargs.get("point_labels"))
        bboxs = kwargs.get("bboxs")
        
        images = kwargs.get("images")
        image_labels = kwargs.get("image_labels")
        prepare_vl_inputs = kwargs.get("prepare_vl")
        task_list = kwargs.get("task_list")
        
        
        batch_size = images.size(0)
        
        self.multimask_output = False
        self.inference = inference
        # Extract visual features.
        '''
        image_embeddings = self.SEG.image_encoder(images)
        '''
        if 'Qwen' in self.config.version:
            lm_outputs = self.VL(**prepare_vl_inputs, output_hidden_states=True)
        elif 'deepseek' in self.config.version:
            fm_labels = prepare_vl_inputs.labels.clone()
            seg_token_mask = (prepare_vl_inputs.input_ids == self.config.seg_token_idx)
            fm_labels[seg_token_mask] = -100
            inputs_embeds = self.VL.prepare_inputs_embeds(**prepare_vl_inputs)
            lm_outputs = self.VL(
                input_ids=prepare_vl_inputs.input_ids,
                attention_mask=prepare_vl_inputs.attention_mask,
                inputs_embeds=inputs_embeds,
                images=prepare_vl_inputs.images,
                images_seq_mask=prepare_vl_inputs.attention_mask,  # Assume attention_mask also serves as the image-sequence mask here.
                images_spatial_crop=prepare_vl_inputs.images_spatial_crop,
                labels=fm_labels,
            )
        else:
                raise ValueError(f"Unsupported model version: {self.config.version}")

        lm_loss = lm_outputs.loss  
        '''
        last_hidden_state = lm_outputs.hidden_states[-1]
        
        if self.inference:
            sample_promptype = self.promptype
        else:
            sample_promptype = self._sample_promptstype(self.promptype)

        if "b" not in sample_promptype:
            bboxs = None
        if "p" not in sample_promptype:
            points = None
        if "t" in sample_promptype:
            if 'Qwen' in self.config.version:
                current_attention_mask = prepare_vl_inputs['attention_mask']
            elif 'deepseek' in self.config.version:
                current_attention_mask = prepare_vl_inputs.attention_mask
            else:
                raise ValueError(f"Cannot determine attention_mask source for model version: {self.config.version}")

        #     expanded_attention_mask = current_attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        #     masked_hidden_state = last_hidden_state * expanded_attention_mask
        #     summed_hidden_state = masked_hidden_state.sum(dim=1) # Shape: (batch_size, hidden_size)
        #     num_non_padding_tokens = current_attention_mask.sum(dim=1, keepdim=True).to(dtype=last_hidden_state.dtype) # Shape: (batch_size, 1)
        #     global_text_embedding = torch.zeros_like(summed_hidden_state) 
        #     valid_batch_indices = (num_non_padding_tokens.squeeze(-1) > 1e-9) # Check against small epsilon to avoid issues with all-padding sequences
        #     if valid_batch_indices.any():
        #         global_text_embedding[valid_batch_indices] = summed_hidden_state[valid_batch_indices] / num_non_padding_tokens[valid_batch_indices]
        #     textfeature = self.seg_fc(global_text_embedding)
        
            # newly added 1
            B, L, H = last_hidden_state.shape
            query = self.add11.expand(B, -1, -1)     # [B,1,H]
            attn_mask = (current_attention_mask == 0)     # [B, L]
            pooled, _ = self.add12(query, last_hidden_state, last_hidden_state,
                                       key_padding_mask=attn_mask)
            textfeature = self.seg_fc(pooled.squeeze(1))
        else:
            textfeature = None
        # seg_text_prompt = self.fusion_fc(textfeature) if textfeature is not None else None
        
        
        
        # newly added 2
        if textfeature is not None:
            # image_embeddings: [B, C, H, W]
            img_feat = self.add21(image_embeddings)          # [B, D, H, W]
            B, D, H0, W0 = img_feat.shape
            img_seq = img_feat.flatten(2).permute(0, 2, 1)        # [B, H0*W0, D]
            # Build the text query.
            txt_q = textfeature.unsqueeze(1)                     # [B, 1, D]
            # query=text，key=value=img_seq
            fused, _ = self.add22(txt_q, img_seq, img_seq)
            seg_text_prompt = fused.squeeze(1)                   # [B, 1, D]
        else:
            seg_text_prompt = None

        if not self.inference:
            seg_text_prompt, points, bboxs = self._sample_prompts(seg_text_prompt, points, bboxs)

        sparse_embeddings, dense_embeddings = self.SEG.prompt_encoder(
            points=points,
            boxes=bboxs,
            masks=None,
            text=seg_text_prompt,
        )
        
        dense_pe = self.SEG.prompt_encoder.get_dense_pe()  # Get the global dense positional encoding.
        SEG_outputs = self.SEG.mask_decoder(
            image_embeddings=image_embeddings,      # [B, C, H, W]
            image_pe=dense_pe,                      # [B, ...] or broadcastable to [B, C, H, W]
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            # text_prompt_embeddings = None,
            text_prompt_embeddings=seg_text_prompt,  # [B, 1, hidden_dim]
            multimask_output=self.multimask_output,
        )

        if self.multimask_output:
            low_res_masks, iou_pred, semantic_pred = self.get_max_pred(SEG_outputs)
        else:
            low_res_masks, iou_pred, semantic_pred = SEG_outputs['low_res_masks'], SEG_outputs['iou_pred'], SEG_outputs['semantic_pred']
            
        pred_masks = F.interpolate(low_res_masks, size=self.image_size, mode='bilinear', align_corners=False)
        
        if self.inference:
            out_masks = torch.sigmoid(pred_masks)
            return {
                "pred_masks": out_masks,
                "gt_masks": image_labels,
                # "lm_loss": lm_loss,
            }
        '''
        if self.inference:
            return {
                "pred_masks": torch.zeros_like(image_labels),  # Placeholder for inference
                "gt_masks": image_labels,
                # "lm_loss": lm_loss,
            }
        else:
            # isseg = [float(self.config.seg_token_idx in ids) for ids in input_ids]
            isseg = [1.0 if 'seg' in task or 'det' in task else 0.0 for task in task_list]
            image_labels_exp = image_labels.unsqueeze(1)  # [B, 1, H, W]
            
        seg_loss = torch.zeros((), device=images.device, dtype=lm_outputs.loss.dtype)
        '''
        valid_seg = 0
        for i in range(batch_size):
            if isseg[i] > 0:
                seg_loss += self.dice_loss(pred_masks[i:i+1], image_labels_exp[i:i+1])
                valid_seg += 1
                
        if valid_seg > 0:
            seg_loss = seg_loss / valid_seg
        '''
        lm_loss = lm_outputs.loss
        lm_loss = lm_loss / batch_size
        total_loss = self.lm_weight * lm_loss + seg_loss
        # total_loss = seg_loss

        return {
            "loss": total_loss,   
            "lm_loss": lm_loss,   
            "seg_loss": seg_loss,
        }
