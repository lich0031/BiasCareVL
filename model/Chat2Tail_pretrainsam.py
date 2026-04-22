from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from utils.utils import DEFAULT_SEG_Task_TOKEN
from segment_anything import build_sam_vit_b


class Chat2TailConfig(PretrainedConfig):
    model_type = "chat2tail"
    def __init__(self, seg_token_idx=None, det_token_idx=None, prompt_token_idx=None, vision_pretrained=None, version=None, tail_loop=False, image_size=1024, out_dim=768, lm_weight=1.0, promptype="pbt", **kwargs):
        super().__init__(**kwargs)
        self.seg_token_idx = seg_token_idx
        self.det_token_idx = det_token_idx
        self.prompt_token_idx = prompt_token_idx
        self.tail_loop = tail_loop
        
        self.vision_pretrained = vision_pretrained
        self.version = version
        if image_size != 1024:
            print(f"Warning: image_size is set to {image_size}, but the model is trained with 1024. This may lead to unexpected results.")
        self.image_size = image_size
        self.out_dim = out_dim
        self.lm_weight = lm_weight
        self.promptype = promptype

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

def clear_model_weights(model):
    for param in model.parameters():
        param.data.zero_()
        
class Chat2TailForCausalLMStandard(PreTrainedModel):
    config_class = Chat2TailConfig
    base_model_prefix = "chat2tail"
    def __init__(self, config, processor=None, lineartype=None):
        super(Chat2TailForCausalLMStandard, self).__init__(config)
        self.processor = processor
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.image_size = config.image_size
        self.lm_weight = config.lm_weight
        self.promptype = config.promptype
        
        # Initialize the vision module.

        self.SEG = self.initialize_seg_modules(config.vision_pretrained)

        # self.SEG = self.SEG.to(torch.bfloat16)
        # mat = self.SEG.prompt_encoder.positional_encoding_gaussian_matrix
        # self.SEG.prompt_encoder.positional_encoding_gaussian_matrix = mat.to(torch.bfloat16)
        print('Using pretrainsam.py')

    def initialize_seg_modules(self, pretrain_path):
        """
        Load a local SAM-ViT-B checkpoint if it exists, otherwise
        fall back to the official public URL.
        """
        pretrain_path = "/mnt/disk2/hwj/Tails/weight/sam_vit_b_01ec64.pth"
        SEG = build_sam_vit_b(checkpoint=pretrain_path)
        SEG.text_model = nn.Identity()
        return SEG

    
    def forward(self, **kwargs):
        # points = (kwargs.get("point_coords"), kwargs.get("point_labels"))
        bboxs = kwargs.get("bboxs")
        images = kwargs.get("images")
        image_labels = kwargs.get("image_labels")
        self.multimask_output = False

        B = images.size(0)
        # images = torch.tensor(images, dtype=torch.float32)
        # image_labels = torch.tensor(image_labels, dtype=torch.float32)
        # point_coords = torch.tensor(kwargs.get("point_coords"), dtype=torch.float32)
        # point_labels = torch.tensor(kwargs.get("point_labels"), dtype=torch.int64)
        images = torch.as_tensor(images, dtype=torch.float32)
        image_labels = torch.as_tensor(image_labels, dtype=torch.float32)
        point_coords = torch.as_tensor(kwargs.get("point_coords"), dtype=torch.float32)
        point_labels = torch.as_tensor(kwargs.get("point_labels"), dtype=torch.int64)

        points = (point_coords, point_labels)
        self.SEG = self.SEG.to(torch.float32)

        image_embeddings = self.SEG.image_encoder(images)
        
        sparse_embeddings, dense_embeddings = self.SEG.prompt_encoder(
            points=points, 
            boxes=None, 
            masks=None, 
            )

        dense_pe = self.SEG.prompt_encoder.get_dense_pe()  # Get the global dense positional encoding.

        # self.SEG = self.SEG.to(torch.bfloat16)
        # sparse_embeddings = sparse_embeddings.to(torch.bfloat16)
        # dense_embeddings = dense_embeddings.to(torch.bfloat16) 
        SEG_outputs = self.SEG.mask_decoder(
            image_embeddings=image_embeddings,      # [B, C, H, W]
            image_pe=dense_pe,                      # [B, ...] or broadcastable to [B, C, H, W]
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,           # [B, 1, hidden_dim], or None if dense embeddings are absent
            multimask_output=self.multimask_output,
        )
        low_res_masks = SEG_outputs[0]
        # low_res_masks = SEG_outputs['low_res_masks']
        pred_masks = F.interpolate(low_res_masks, size=self.image_size, mode='bilinear', align_corners=False)
        out_masks = torch.sigmoid(pred_masks)
        return {
            "pred_masks": out_masks,
            "gt_masks": image_labels,
        }
