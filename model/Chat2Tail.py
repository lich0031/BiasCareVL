from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from utils.utils import DEFAULT_SEG_Task_TOKEN, GLOBAL_SEED, DEFAULT_PROMPT_TOKEN, DEFAULT_CLS_Task_TOKEN
from .IMIS.utils import FocalDice_MSELoss
from .IMIS.segment_anything import build_sam_vit_b
from model.IMIS.dataloaders.data_utils import get_points_from_mask, get_bboxes_from_mask, get_points_from_mask_batch
from torchvision.ops import sigmoid_focal_loss
from PIL import Image
import numpy as np
import random

import os
import json
import gc

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class ASLwithClassWeight(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(ASLwithClassWeight, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1-p)
        self.neg_weights = torch.exp(p)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, label):
        weight = label * self.pos_weights.to(pred.device) + (1 - label) * self.neg_weights.to(pred.device)

        # Calculating Probabilities
        xs_pos = torch.sigmoid(pred)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

       # Basic CE calculation
        los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - label) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss *= weight

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * label
            pt1 = xs_neg * (1 - label)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()
    
class Chat2TailConfig(PretrainedConfig):
    model_type = "chat2tail"
    def __init__(self, seg_token_idx=None, det_token_idx=None, prompt_token_idx=None, cls_token_idx=None, vision_pretrained=None, version=None,aware_loss=False, tail_loop=False, image_size=224, out_dim=768, lm_weight=1.0, lambda_uncertainty=1.0, cls_num = 0, promptype="pbt", clsloss_type=None, **kwargs):
        super().__init__(**kwargs)
        self.seg_token_idx = seg_token_idx
        self.det_token_idx = det_token_idx
        self.prompt_token_idx = prompt_token_idx
        self.cls_token_idx = cls_token_idx
        self.tail_loop = tail_loop
        self.clsloss_type = clsloss_type
        self.vision_pretrained = vision_pretrained
        self.version = version
        self.image_size = image_size
        self.out_dim = out_dim
        self.lm_weight = lm_weight
        self.promptype = promptype
        self.aware_loss = aware_loss
        self.lambda_uncertainty = lambda_uncertainty  # Hyperparameter; larger values apply more extreme weighting.
        self.cls_num = cls_num

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
    # base_model_prefix = "model"
    def __init__(self, config, processor=None, lineartype='combine'):
        super(Chat2TailForCausalLMStandard, self).__init__(config)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.dice_loss = DiceLoss()
        self.image_size = config.image_size
        self.lm_weight = config.lm_weight
        self.promptype = config.promptype
        self.cls_num = config.cls_num
        self.clsloss_type = config.clsloss_type
        self.lineartype = lineartype

        # Initialize the vision module.
        self.SEG = self.initialize_seg_modules(pretrain_path=self.config.vision_pretrained)

        # Initialize the language model.
        if "deepseek" in self.config.version:
            from .deepseek_vl2.models import DeepseekVLV2ForCausalLM
            self.VL = DeepseekVLV2ForCausalLM.from_pretrained(self.config.version)
            self.VL.language.resize_token_embeddings(len(self.tokenizer))
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

        
        embed_dim = self.SEG.image_encoder.encoder_embed_dim
        hidden_size = self.VL.language.config.hidden_size
        self.text_fc = nn.Linear(hidden_size, embed_dim)
        
        # fintune seg module so cut the langugae branch
        '''newly added 1'''
        self.add11 = nn.Parameter(torch.randn(1, 1, self.VL.language.config.hidden_size))
        self.add12 = nn.MultiheadAttention(self.VL.language.config.hidden_size, 8, batch_first=True)
        
        '''for classification'''
        self.config.num_classes = self.cls_num
        if self.config.num_classes and self.config.num_classes > 0:
            if self.lineartype == 'vlmcross':
                self.cls_head_cross_attn = nn.MultiheadAttention(embed_dim=self.VL.language.config.hidden_size, num_heads=8, batch_first=True)
                self.cls_head_ln = nn.LayerNorm(self.VL.language.config.hidden_size)
                self.cls_head_final = nn.Linear(self.VL.language.config.hidden_size, self.config.num_classes)
                
            elif self.lineartype == 'vlmcrosscombine':
                self.cls_head_cross_attn = nn.MultiheadAttention(embed_dim=self.VL.language.config.hidden_size, num_heads=8, batch_first=True)
                self.cls_head_ln = nn.LayerNorm(self.VL.language.config.hidden_size)
                img_dim = self.SEG.image_encoder.encoder_embed_dim
                self.cls_head_pro = nn.Linear(self.VL.language.config.hidden_size, img_dim)
                self.cls_head_final = nn.Linear(img_dim *2, self.config.num_classes)

            elif self.lineartype == 'twovisiondualcross':
                embed_dim = self.SEG.image_encoder.encoder_embed_dim
                self.cls_head_cross_attn1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
                self.cls_head_ln1 = nn.LayerNorm(embed_dim)
                self.cls_head_cross_attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
                self.cls_head_ln2 = nn.LayerNorm(embed_dim)
                self.cls_head_pro = nn.Linear(self.VL.language.config.hidden_size, embed_dim)
                self.cls_head_final = nn.Linear(embed_dim *2, self.config.num_classes)

            elif self.lineartype == 'twovisiondualcrosstoken':
                embed_dim = self.SEG.image_encoder.encoder_embed_dim
                self.cls_head_cross_attn1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
                self.cls_head_ln1 = nn.LayerNorm(embed_dim)
                self.cls_head_cross_attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)
                self.cls_head_ln2 = nn.LayerNorm(embed_dim)
                self.cls_head_pro = nn.Linear(self.VL.language.config.hidden_size, embed_dim)
                self.cls_head_pro_token = nn.Linear(self.VL.language.config.hidden_size, embed_dim)
                self.cls_head_Lrelu = nn.LeakyReLU(0.1)
                self.cls_head_final = nn.Linear(embed_dim *3, self.config.num_classes)
            
            elif self.lineartype == 'vlmcrossdualcrossplustext' or self.lineartype == 'vlmcrossdualcross':
                self.cls_head_cross_attn1 = nn.MultiheadAttention(embed_dim=self.VL.language.config.hidden_size, num_heads=8, batch_first=True)
                self.cls_head_ln1 = nn.LayerNorm(self.VL.language.config.hidden_size)
                img_dim = self.SEG.image_encoder.encoder_embed_dim
                self.cls_head_pro = nn.Linear(self.VL.language.config.hidden_size, img_dim)
                self.cls_head_cross_attn2 = nn.MultiheadAttention(embed_dim=img_dim, num_heads=8, batch_first=True)
                self.cls_head_ln2 = nn.LayerNorm(img_dim)
                self.cls_head_cross_attn3 = nn.MultiheadAttention(embed_dim=img_dim, num_heads=8, batch_first=True)
                self.cls_head_ln3 = nn.LayerNorm(img_dim)
                self.cls_head_final = nn.Linear(img_dim *2, self.config.num_classes)
                
            elif self.lineartype == 'vlmdualcrossdualcross':             
                self.cls_head_cross_attn1 = nn.MultiheadAttention(embed_dim=self.VL.language.config.hidden_size, num_heads=8, batch_first=True)
                self.cls_head_ln1 = nn.LayerNorm(self.VL.language.config.hidden_size)
                self.cls_head_cross_attn2 = nn.MultiheadAttention(embed_dim=self.VL.language.config.hidden_size, num_heads=8, batch_first=True)
                self.cls_head_ln2 = nn.LayerNorm(self.VL.language.config.hidden_size)
                img_dim = self.SEG.image_encoder.encoder_embed_dim
                self.cls_head_pro = nn.Linear(self.VL.language.config.hidden_size, img_dim)
                
                self.cls_head_cross_attn3 = nn.MultiheadAttention(embed_dim=img_dim, num_heads=8, batch_first=True)
                self.cls_head_ln3 = nn.LayerNorm(img_dim)
                self.cls_head_cross_attn4 = nn.MultiheadAttention(embed_dim=img_dim, num_heads=8, batch_first=True)
                self.cls_head_ln4 = nn.LayerNorm(img_dim)
                self.cls_head_final = nn.Linear(img_dim *2, self.config.num_classes)
                
            elif self.lineartype == 'vlmtokendualcrossdualcross1' or self.lineartype == 'vlmtokendualcrossdualcross2':             
                self.cls_head_cross_attn1 = nn.MultiheadAttention(embed_dim=self.VL.language.config.hidden_size, num_heads=8, batch_first=True)
                self.cls_head_ln1 = nn.LayerNorm(self.VL.language.config.hidden_size)
                img_dim = self.SEG.image_encoder.encoder_embed_dim
                self.cls_head_pro = nn.Linear(self.VL.language.config.hidden_size, img_dim)
                
                self.cls_head_cross_attn2 = nn.MultiheadAttention(embed_dim=img_dim, num_heads=8, batch_first=True)
                self.cls_head_ln2 = nn.LayerNorm(img_dim)
                self.cls_head_cross_attn3 = nn.MultiheadAttention(embed_dim=img_dim, num_heads=8, batch_first=True)
                self.cls_head_ln3 = nn.LayerNorm(img_dim)
                self.cls_head_final = nn.Linear(img_dim *2, self.config.num_classes)
                
            else:
                raise ValueError(f"Unsupported lineartype: {self.lineartype}")

            if self.config.num_classes == 40:
                
                if self.clsloss_type is None or 'focalonly' in self.clsloss_type:
                    self.cls_loss_fn = nn.BCEWithLogitsLoss().cuda()
                elif self.clsloss_type == 'asl':
                    total_instance_num = 298164
                    class_instance_nums = [3886, 75507, 219, 4833, 86321, 187, 17750, 43202,
                                           4402, 33872, 1332, 3154, 13144, 3348, 4660, 774,
                                           823, 11593, 890, 155, 2652, 89213, 6077, 8650, 39380,
                                           76831, 696, 3751, 826, 53721, 570, 16200, 1935, 1022,
                                           10169, 218, 2477, 99240, 3831, 2455]
                    self.cls_loss_fn = ASLwithClassWeight(
                        class_instance_nums=class_instance_nums, 
                        total_instance_num=total_instance_num
                    )
                    
                else:
                    print(f'clsloss_type: {self.clsloss_type}')
                    print('Using pos_weight for 40 classes')
                    # pos_weight = [74.9375, 2.9597, 1299.8593, 60.0689, 2.4637, 1539.8988,
                    # 15.8415, 5.9484, 69.7105, 7.7374, 220.4465, 91.3550,
                    # 21.3782, 86.3089, 63.9451, 399.7291, 355.0811, 24.6638,
                    # 331.7391, 2005.7520,  109.7233, 2.3410, 47.9544, 33.3741,
                    # 6.5490, 2.8986,  419.2451, 78.1171, 366.7145, 4.5480,
                    # 500.6880, 17.6803, 157.7192, 285.6789, 28.0247, 1504.0640,
                    # 125.5254, 2.0074, 76.5992, 123.5770] # no extra
                    pos_weight = [  75.7277,    2.9488, 1360.4795,   60.6934,    2.4541, 1593.4598,
                    15.7980,    5.9016,   66.7338,    7.8027,  222.8468,   93.5352,
                    21.6844,   88.0574,   62.9837,  384.2248,  361.2892,   24.7193,
                    334.0157, 1922.6387,  111.4299,    2.3422,   48.0643,   33.4698,
                    6.5715,    2.8808,  427.3965,   78.4892,  359.9734,    4.5502,
                    522.0947,   17.4052,  153.0899,  290.7456,   28.3209, 1366.7247,
                    119.3730,    2.0045,   76.8293,  120.4517] # extra data

                    if 'log2' in self.clsloss_type:
                        smoothed = np.log2(pos_weight)
                    elif 'log1p' in self.clsloss_type:
                        smoothed = np.log1p(pos_weight)
                    elif 'logmean' in self.clsloss_type:
                        log = np.log2(pos_weight)
                        smoothed = log / np.mean(log)
                    elif 'sqrt' in self.clsloss_type:
                        smoothed = np.sqrt(pos_weight)
                    elif 'clip' in self.clsloss_type:
                        clipnum = int(self.clsloss_type.replace('clip',''))
                        smoothed = np.clip(pos_weight, 0, clipnum)
                    self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(smoothed).cuda())

            elif self.config.num_classes == 7:
                # pos_weight = [7.9982,  0.4937, 18.4844, 29.6269,  8.1128, 86.0870, 69.5282]
                if self.clsloss_type is None or self.clsloss_type == 'focalonly':
                    self.cls_loss_fn = nn.BCEWithLogitsLoss().cuda()
                else:
                    print(f'clsloss_type: {self.clsloss_type}')
                    pos_weight = [8.0, 0.5, 18.5, 29.6, 8.1, 86.1, 69.5]
                    if 'log2' in self.clsloss_type:
                        smoothed = np.log2(pos_weight)
                    elif 'log1p' in self.clsloss_type:
                        smoothed = np.log1p(pos_weight)
                    elif 'logmean' in self.clsloss_type:
                        log = np.log2(pos_weight)
                        smoothed = log / np.mean(log)
                    elif 'sqrt' in self.clsloss_type:
                        smoothed = np.sqrt(pos_weight)
                    elif 'clip' in self.clsloss_type:
                        clipnum = int(self.clsloss_type.replace('clip',''))
                        smoothed = np.clip(pos_weight, 0, clipnum)
                    elif 'pos' in self.clsloss_type:
                        smoothed = pos_weight
                    self.cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(smoothed).cuda())
        else:
            self.cls_head_final = None
        ''''''''''''''''''''''''
            
            
        '''useless-but maybe for jbhi usuful'''
        # self.seg_fc  = nn.Linear(embed_dim, hidden_size)
        # self.add21 = nn.Conv2d(self.SEG.image_encoder.encoder_embed_dim, config.out_dim, kernel_size=1)
        # self.add22 = nn.MultiheadAttention(config.out_dim, 4, batch_first=True)
        # self.add3_mask_prompt_proj = nn.Linear(
        #     config.image_size * config.image_size,
        #     self.VL.language.config.hidden_size
        # )
        # self.add3_mask2text_attn = nn.MultiheadAttention(
        #     embed_dim=self.VL.language.config.hidden_size,
        #     num_heads=8,
        #     batch_first=True
        # )
        '''useless'''
        
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

    def get_points_from_xor_masks(self, gt_hard_masks, pred_bin_masks, get_point=1):
        """
        Sample points only from the union of gt_hard and pred_bin:
        - if a sampled point lies in gt_hard (under-segmentation), label=1
        - otherwise (over-segmentation), label=0
        gt_hard_masks, pred_bin_masks: ndarray or Tensor, shape (B,H,W) or (B,1,H,W)
        returns coords: Tensor[B,1,2], labels: Tensor[B,1]
        """
        # Convert inputs to numpy arrays with shape (H, W).
        if isinstance(gt_hard_masks, torch.Tensor):
            gt_hard_masks = gt_hard_masks.squeeze(1).float().cpu().numpy().astype(np.uint8)
        if isinstance(pred_bin_masks, torch.Tensor):
            pred_bin_masks = pred_bin_masks.squeeze(1).cpu().numpy().astype(np.uint8)


        B = gt_hard_masks.shape[0]
        coords_batch, labels_batch = [], []
        H, W = gt_hard_masks.shape[1], gt_hard_masks.shape[2]

        for i in range(B):
            gt = gt_hard_masks[i]
            pb = pred_bin_masks[i]
            union = (gt > 0) | (pb > 0)
            
            if np.array_equal(gt, pb):
                # If they are identical, sampling from the intersection is allowed.
                union = (gt > 0)
            else:
                # Union minus intersection (XOR).
                union = (gt > 0) ^ (pb > 0)
            
            pts = np.argwhere(union)  # [[y,x], …]
            if pts.shape[0] == 0:
                # If the union is empty, sample a random point from the whole image.
                y, x = np.random.randint(0, H), np.random.randint(0, W)
            else:
                idx = np.random.randint(pts.shape[0])
                y, x = pts[idx]
            label = 1 if gt[y, x] > 0 else 0
            coords_batch.append([x, y])
            labels_batch.append(label)

        coords = torch.tensor(coords_batch, dtype=torch.float).unsqueeze(1)  # [B,1,2]
        labels = torch.tensor(labels_batch, dtype=torch.int).unsqueeze(1)    # [B,1]
        return coords, labels

    def _get_hard_samples(self, samples: torch.Tensor, k: int, top_k_ratio: float = 0.3):
        """
        losses: Tensor, shape [B], per-sample loss values (Dice loss or another metric)
        top_k_ratio: ratio of the hardest samples to keep
        returns: bool mask, shape [B]
        """
        B = samples.size(0)
        k = max(1, int(k * top_k_ratio))
        # Select the indices of the top-k largest losses.
        _, idx = torch.topk(samples, k)
        mask = torch.zeros(B, dtype=torch.bool, device=samples.device)
        mask[idx] = True
        return mask, k
    
    def forward(self, **kwargs):
        points = (kwargs.get("point_coords"), kwargs.get("point_labels"))
        bboxs = kwargs.get("bboxs")
        images = kwargs.get("images")
        image_labels = kwargs.get("image_labels")
        class_labels = kwargs.get("class_tail_label_list", None)
        
        prepare_vl_inputs = kwargs.get("prepare_vl")
        task_list = kwargs.get("task_list")
        valid = kwargs.get("valid")
        self.multimask_output = False

        B = images.size(0)
        image_embeddings = self.SEG.image_encoder(images)

        lm_outputs = self.VL(**prepare_vl_inputs, output_hidden_states=True)
        attention_mask = prepare_vl_inputs["attention_mask"]
        last_hidden_state = lm_outputs.hidden_states[-1]
        lm_loss = lm_outputs.loss
        

        if self.cls_head_final is not None:
            if self.lineartype == 'vlmcross':
                image_mask = (prepare_vl_inputs["input_ids"] == self.VL.config.image_token_id)
                text_mask = ~image_mask
                vision_feats = []
                text_feats = []

                # Iterate over the batch and extract image/text token features for each sample.
                for i in range(last_hidden_state.shape[0]):  # [B, seq_len, hidden_dim]
                    vision_feat = last_hidden_state[i][image_mask[i]]   # [num_img_tokens, hidden_dim]
                    text_feat = last_hidden_state[i][text_mask[i]]      # [num_text_tokens, hidden_dim]
                    vision_feats.append(vision_feat)
                    text_feats.append(text_feat)

                # Pad variable-length sequences so they can be passed into MultiheadAttention.
                vision_feats = torch.nn.utils.rnn.pad_sequence(vision_feats, batch_first=True)
                text_feats = torch.nn.utils.rnn.pad_sequence(text_feats, batch_first=True)

                # cls_head_cross_attn expects [B, seq_len, hidden_dim]
                # and internally converts to [seq_len, B, hidden_dim].
                fused, _ = self.cls_head_cross_attn(query=text_feats, key=vision_feats, value=vision_feats)
                fused = self.cls_head_ln(fused + text_feats)
                cls_feat = fused.mean(dim=1)
            elif self.lineartype == 'vlmcrosscombine':
                image_mask = (prepare_vl_inputs["input_ids"] == self.VL.config.image_token_id)
                text_mask = ~image_mask
                vision_feats = []
                text_feats = []

                # Iterate over the batch and extract image/text token features for each sample.
                for i in range(last_hidden_state.shape[0]):  # [B, seq_len, hidden_dim]
                    vision_feat = last_hidden_state[i][image_mask[i]]   # [num_img_tokens, hidden_dim]
                    text_feat = last_hidden_state[i][text_mask[i]]      # [num_text_tokens, hidden_dim]
                    vision_feats.append(vision_feat)
                    text_feats.append(text_feat)

                # Pad variable-length sequences so they can be passed into MultiheadAttention.
                vision_feats = torch.nn.utils.rnn.pad_sequence(vision_feats, batch_first=True)
                text_feats = torch.nn.utils.rnn.pad_sequence(text_feats, batch_first=True)

                # cls_head_cross_attn expects [B, seq_len, hidden_dim]
                # and internally converts to [seq_len, B, hidden_dim].
                fused, _ = self.cls_head_cross_attn(query=text_feats, key=vision_feats, value=vision_feats)
                fused = self.cls_head_ln(fused + text_feats)
                cls_feat = fused.mean(dim=1)
                cls_feat = self.cls_head_pro(cls_feat)
                img_feat = image_embeddings.mean(dim=(2,3))
                cls_feat = torch.cat([cls_feat, img_feat], dim=1)
                
            elif self.lineartype == 'vlmcrossdualcrossplustext':
                img_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # [B, HW, D]
                image_mask = (prepare_vl_inputs["input_ids"] == self.VL.config.image_token_id)
                text_mask = ~image_mask
                vision_feats = []
                text_feats = []

                # Iterate over the batch and extract image/text token features for each sample.
                for i in range(last_hidden_state.shape[0]):  # [B, seq_len, hidden_dim]
                    vision_feat = last_hidden_state[i][image_mask[i]]   # [num_img_tokens, hidden_dim]
                    text_feat = last_hidden_state[i][text_mask[i]]      # [num_text_tokens, hidden_dim]
                    vision_feats.append(vision_feat)
                    text_feats.append(text_feat)

                # Pad variable-length sequences so they can be passed into MultiheadAttention.
                vision_feats = torch.nn.utils.rnn.pad_sequence(vision_feats, batch_first=True)
                text_feats = torch.nn.utils.rnn.pad_sequence(text_feats, batch_first=True)

                # cls_head_cross_attn expects [B, seq_len, hidden_dim]
                # and internally converts to [seq_len, B, hidden_dim].
                vlmfused, _ = self.cls_head_cross_attn1(query=text_feats, key=vision_feats, value=vision_feats)
                vlmfused = self.cls_head_ln1(vlmfused + text_feats)
                vlmfused = self.cls_head_pro(vlmfused)
                
                fused1, _ = self.cls_head_cross_attn2(query=img_flat, key=vlmfused, value=vlmfused)
                fused1 = self.cls_head_ln2(fused1 + img_flat)
                
                fused2, _ = self.cls_head_cross_attn3(query=vlmfused, key=img_flat, value=img_flat)
                fused2 = self.cls_head_ln3(fused2 + vlmfused)
                
                cls_feat1 = fused1.mean(dim=1)
                cls_feat2 = fused2.mean(dim=1)
                cls_feat = torch.cat([cls_feat1, cls_feat2], dim=1)
            
            
            elif self.lineartype == 'vlmcrossdualcross':
                img_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # [B, HW, D]
                image_mask = (prepare_vl_inputs["input_ids"] == self.VL.config.image_token_id)
                text_mask = ~image_mask
                vision_feats = []
                text_feats = []

                # Iterate over the batch and extract image/text token features for each sample.
                for i in range(last_hidden_state.shape[0]):  # [B, seq_len, hidden_dim]
                    vision_feat = last_hidden_state[i][image_mask[i]]   # [num_img_tokens, hidden_dim]
                    text_feat = last_hidden_state[i][text_mask[i]]      # [num_text_tokens, hidden_dim]
                    vision_feats.append(vision_feat)
                    text_feats.append(text_feat)

                # Pad variable-length sequences so they can be passed into MultiheadAttention.
                # vision_feats = torch.nn.utils.rnn.pad_sequence(vision_feats, batch_first=True)
                # text_feats = torch.nn.utils.rnn.pad_sequence(text_feats, batch_first=True)
                vision_feats = torch.stack(vision_feats)
                text_feats = torch.stack(text_feats)

                # cls_head_cross_attn expects [B, seq_len, hidden_dim]
                # and internally converts to [seq_len, B, hidden_dim].
                vlmfused, _ = self.cls_head_cross_attn1(query=text_feats, key=vision_feats, value=vision_feats)
                vlmfused = self.cls_head_ln1(vlmfused)
                vlmfused = self.cls_head_pro(vlmfused)
                
                fused1, _ = self.cls_head_cross_attn2(query=img_flat, key=vlmfused, value=vlmfused)
                fused1 = self.cls_head_ln2(fused1 + img_flat)
                
                fused2, _ = self.cls_head_cross_attn3(query=vlmfused, key=img_flat, value=img_flat)
                fused2 = self.cls_head_ln3(fused2 + vlmfused)
                
                cls_feat1 = fused1.mean(dim=1)
                cls_feat2 = fused2.mean(dim=1)
                cls_feat = torch.cat([cls_feat1, cls_feat2], dim=1)
                
            elif self.lineartype == 'vlmdualcrossdualcross':
                img_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # [B, HW, D]
                image_mask = (prepare_vl_inputs["input_ids"] == self.VL.config.image_token_id)
                text_mask = ~image_mask
                vision_feats = []
                text_feats = []
                
                for i in range(last_hidden_state.shape[0]):  # [B, seq_len, hidden_dim]
                    vision_feat = last_hidden_state[i][image_mask[i]]   # [num_img_tokens, hidden_dim]
                    text_feat = last_hidden_state[i][text_mask[i]]      # [num_text_tokens, hidden_dim]
                    vision_feats.append(vision_feat)
                    text_feats.append(text_feat)
                    
                vision_feats = torch.stack(vision_feats)
                text_feats = torch.stack(text_feats)

                vlmfused1, _ = self.cls_head_cross_attn1(query=text_feats, key=vision_feats, value=vision_feats)
                vlmfused1 = self.cls_head_ln1(vlmfused1 + text_feats)
                
                vlmfused2, _ = self.cls_head_cross_attn2(query=vision_feats, key=text_feats, value=text_feats)
                vlmfused2 = self.cls_head_ln2(vlmfused2 + vision_feats)

                vlmfused = torch.cat([vlmfused1, vlmfused2], dim=1)
                vlmfused = self.cls_head_pro(vlmfused)

                fused1, _ = self.cls_head_cross_attn3(query=img_flat, key=vlmfused, value=vlmfused)
                fused1 = self.cls_head_ln3(fused1 + img_flat)
                
                fused2, _ = self.cls_head_cross_attn4(query=vlmfused, key=img_flat, value=img_flat)
                fused2 = self.cls_head_ln4(fused2 + vlmfused)
                
                cls_feat1 = fused1.mean(dim=1)
                cls_feat2 = fused2.mean(dim=1)
                cls_feat = torch.cat([cls_feat1, cls_feat2], dim=1)
            
            elif self.lineartype == 'vlmtokendualcrossdualcross1':
                img_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # [B, HW, D]
                image_mask = (prepare_vl_inputs["input_ids"] == self.VL.config.image_token_id)
                text_mask = ~image_mask
                vision_feats = []

                cls_token_idx = self.config.cls_token_idx
                cls_mask = (prepare_vl_inputs["input_ids"] == cls_token_idx)
                text_feats = last_hidden_state[cls_mask].unsqueeze(1)  # [B,1,hidden_dim]
                
                for i in range(last_hidden_state.shape[0]):  # [B, seq_len, hidden_dim]
                    vision_feat = last_hidden_state[i][image_mask[i]]   # [num_img_tokens, hidden_dim]
                    vision_feats.append(vision_feat)
                    
                vision_feats = torch.stack(vision_feats)

                tokenfused, _ = self.cls_head_cross_attn1(query=text_feats, key=vision_feats, value=vision_feats)
                tokenfused = self.cls_head_ln1(tokenfused + text_feats)

                vlmfused = self.cls_head_pro(tokenfused)

                fused1, _ = self.cls_head_cross_attn2(query=img_flat, key=vlmfused, value=vlmfused)
                fused1 = self.cls_head_ln2(fused1 + img_flat)
                
                fused2, _ = self.cls_head_cross_attn3(query=vlmfused, key=img_flat, value=img_flat)
                fused2 = self.cls_head_ln3(fused2 + vlmfused)
                
                cls_feat1 = fused1.mean(dim=1)
                cls_feat2 = fused2.mean(dim=1)
                cls_feat = torch.cat([cls_feat1, cls_feat2], dim=1)
            
            elif self.lineartype == 'vlmtokendualcrossdualcross2':
                img_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # [B, HW, D]
                image_mask = (prepare_vl_inputs["input_ids"] == self.VL.config.image_token_id)
                text_mask = ~image_mask
                vision_feats = []

                cls_token_idx = self.config.cls_token_idx
                cls_mask = (prepare_vl_inputs["input_ids"] == cls_token_idx)
                text_feats = last_hidden_state[cls_mask].unsqueeze(1)  # [B,1,hidden_dim]
                
                for i in range(last_hidden_state.shape[0]):  # [B, seq_len, hidden_dim]
                    vision_feat = last_hidden_state[i][image_mask[i]]   # [num_img_tokens, hidden_dim]
                    vision_feats.append(vision_feat)
                    
                vision_feats = torch.stack(vision_feats)

                tokenfused, _ = self.cls_head_cross_attn1(query=vision_feats, key=text_feats, value=text_feats)
                tokenfused = self.cls_head_ln1(tokenfused + vision_feats)

                vlmfused = self.cls_head_pro(tokenfused)

                fused1, _ = self.cls_head_cross_attn2(query=img_flat, key=vlmfused, value=vlmfused)
                fused1 = self.cls_head_ln2(fused1 + img_flat)
                
                fused2, _ = self.cls_head_cross_attn3(query=vlmfused, key=img_flat, value=img_flat)
                fused2 = self.cls_head_ln3(fused2 + vlmfused)
                
                cls_feat1 = fused1.mean(dim=1)
                cls_feat2 = fused2.mean(dim=1)
                cls_feat = torch.cat([cls_feat1, cls_feat2], dim=1)

            elif self.lineartype == 'twovisiondualcross':
                img_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # [B, HW, D]
                vision_feats = []
                for i in range(last_hidden_state.shape[0]):  # batch loop
                    vision_feat = last_hidden_state[i][(prepare_vl_inputs["input_ids"][i] == self.VL.config.image_token_id)]
                    vision_feats.append(vision_feat)
                vision_feats = torch.stack(vision_feats)
                vision_feats = self.cls_head_pro(vision_feats)
                fused1, _ = self.cls_head_cross_attn1(query=img_flat, key=vision_feats, value=vision_feats)
                cls_feat1 = self.cls_head_ln1(fused1 + img_flat)
                cls_feat1 = cls_feat1.mean(dim=1)
                fused2, _ = self.cls_head_cross_attn2(query=vision_feats, key=img_flat, value=img_flat)
                cls_feat2 = self.cls_head_ln2(fused2 + vision_feats)
                cls_feat2 = cls_feat2.mean(dim=1)
                cls_feat = torch.cat([cls_feat1, cls_feat2], dim=1)
                
            elif self.lineartype == 'twovisiondualcrosstoken':
                img_flat = image_embeddings.flatten(2).permute(0, 2, 1)   # [B, HW, D]
                vision_feats = []
                for i in range(last_hidden_state.shape[0]):  # batch loop
                    vision_feat = last_hidden_state[i][(prepare_vl_inputs["input_ids"][i] == self.VL.config.image_token_id)]
                    vision_feats.append(vision_feat)
                vision_feats = torch.stack(vision_feats)
                vision_feats = self.cls_head_pro(vision_feats)
                fused1, _ = self.cls_head_cross_attn1(query=img_flat, key=vision_feats, value=vision_feats)
                cls_feat1 = self.cls_head_ln1(fused1 + img_flat)
                cls_feat1 = cls_feat1.mean(dim=1)
                fused2, _ = self.cls_head_cross_attn2(query=vision_feats, key=img_flat, value=img_flat)
                cls_feat2 = self.cls_head_ln2(fused2 + vision_feats)
                cls_feat2 = cls_feat2.mean(dim=1)
                fused = torch.cat([cls_feat1, cls_feat2], dim=1)

                cls_token_idx = self.config.cls_token_idx
                cls_mask = (prepare_vl_inputs["input_ids"] == cls_token_idx)
                token_feat = last_hidden_state[cls_mask]
                token_feat = self.cls_head_pro_token(token_feat)
                token_feat = self.cls_head_Lrelu(token_feat)
                cls_feat = torch.cat([token_feat, fused], dim=1)
                
            cls_logits = self.cls_head_final(cls_feat)

        else:
            cls_logits = None
        B, L, H = last_hidden_state.shape
        
        query = self.add11.expand(B, -1, -1)     # [B,1,H]
        attn_mask = (attention_mask == 0)     # [B, L]
        pooled, _ = self.add12(query, last_hidden_state, last_hidden_state, key_padding_mask=attn_mask)
        text_prompt_embeddings = self.text_fc(pooled.squeeze(1))

        sparse_embeddings, dense_embeddings, point_embeddings, box_embeddings, text_embeddings = self.SEG.prompt_encoder(points=points, boxes=bboxs, masks=None, text=text_prompt_embeddings)

        dense_pe = self.SEG.prompt_encoder.get_dense_pe()  # Get the global dense positional encoding.
        seg_loss_all = torch.zeros((), device=images.device)
        seg_cnt = 0
        if not valid:
            prompt_options = [[None, box_embeddings, text_embeddings], 
                            [point_embeddings, None, text_embeddings], 
                            [point_embeddings, box_embeddings, None], 
                            [None, None, text_embeddings], 
                            [None, box_embeddings, None], 
                            [point_embeddings, None, None], 
                            [point_embeddings, box_embeddings, text_embeddings]]

            prompt_options = random.sample(prompt_options, k=1) 
            for prompt in prompt_options:
                sparse_embeddings = torch.empty((B, 0, self.SEG.image_encoder.encoder_embed_dim), device=image_embeddings.device)
                bembed, pembed, tembed = prompt
                if bembed is not None:
                    sparse_embeddings = torch.cat([sparse_embeddings, bembed], dim=1)
                if pembed is not None:
                    sparse_embeddings = torch.cat([sparse_embeddings, pembed], dim=1)
                if tembed is not None:
                    sparse_embeddings = torch.cat([sparse_embeddings, tembed], dim=1)
                SEG_outputs = self.SEG.mask_decoder(
                                image_embeddings=image_embeddings,      # [B, C, H, W]
                                image_pe=dense_pe,                      # [B, ...] or broadcastable to [B, C, H, W]
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,           # [B, 1, hidden_dim], or None if dense embeddings are absent
                                text_prompt_embeddings=text_prompt_embeddings,  # [B, 1, hidden_dim]
                                multimask_output=self.multimask_output,
                                )
                low_res_masks = SEG_outputs['low_res_masks']   
                
                pred_masks = F.interpolate(low_res_masks, size=self.image_size, mode='bilinear', align_corners=False)
                out_masks = torch.sigmoid(pred_masks)
                
                
                valid_logits = torch.tensor([('seg' in t or 'det' in t) for t in task_list], device=images.device, dtype=torch.float)
                masks = pred_masks   # [B,1,H,W]
                gts   = image_labels.unsqueeze(1)  # [B,1,H,W]
                pred_sig = torch.sigmoid(masks)
                pred_flat = pred_sig.view(B, -1)
                gt_flat   = gts.view(B, -1)
                inter = (pred_flat * gt_flat).sum(dim=1)
                union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
                dice_loss_per_sample  = 1 - (2*inter + 1e-6) / (union + 1e-6)   # [B]
                if self.config.aware_loss:
                    self.config.beta_vl = 0.5
                    pred_mask = torch.sigmoid(low_res_masks)  # [B, 1, 56, 56]
                    pred_mask_down = F.interpolate(pred_mask, size=image_embeddings.shape[2:], mode="bilinear", align_corners=False)  # [B, 1, 14, 14]
                    
                    z_img = (image_embeddings * pred_mask_down).sum(dim=(2,3)) / (pred_mask_down.sum(dim=(2,3)) + 1e-6)  # [B, C]
                    z_txt = text_prompt_embeddings.squeeze(1)                      # [B, C]

                    vl_sim = F.cosine_similarity(z_img, z_txt, dim=-1)             # [B] #[0.0359, 0.0583, 0.0076, 0.0299, 0.0148]
                    vl_uncertainty = 1.0 - vl_sim.detach()                         # [B]

                    uncertainty = dice_loss_per_sample + self.config.beta_vl * vl_uncertainty
                    uncertainty_weight = torch.exp(self.config.lambda_uncertainty * uncertainty)
                    uncertainty_weight = uncertainty_weight * valid_logits
                    uncertainty_weight = uncertainty_weight / (uncertainty_weight.max().clamp(min=1.0) + 1e-6)
                    
                    # seg_cnt += valid_logits.sum()
                    seg_loss = (dice_loss_per_sample * uncertainty_weight).sum() / (valid_logits.sum().clamp(min=1.0))
                    
                else:
                    # seg_cnt += valid_logits.sum()
                    seg_loss = (dice_loss_per_sample * valid_logits).sum() / (valid_logits.sum().clamp(min=1.0))
                
                seg_loss_all += seg_loss
                
                
                if self.config.tail_loop:
                    # print(f"Using tail_loop")
                    if self.config.aware_loss:
                        hard_mask, hard_num = self._get_hard_samples(uncertainty_weight, k=valid_logits.sum(), top_k_ratio=0.3)
                    else:
                        hard_mask, hard_num = self._get_hard_samples(dice_loss_per_sample, k=valid_logits.sum(), top_k_ratio=0.3)

                    if hard_mask.sum() > 0:
                        gt_hard   = image_labels[hard_mask]
                        pred_bin  = (out_masks[hard_mask] > 0.5).float()

                        refine_points, refine_labels = self.get_points_from_xor_masks(
                            gt_hard_masks=(gt_hard),          # torch Tensor [K,H,W], where K is the number of hard samples
                            pred_bin_masks=(pred_bin.squeeze(1)>0).int(), 
                            get_point=1
                        )

                        refine_points = refine_points.to(images.device)  # [K, 2]
                        refine_labels = refine_labels.to(images.device)
                        refine_sparse, refine_dense, _, _, _ = self.SEG.prompt_encoder(
                            points=(refine_points, refine_labels),
                            boxes=bboxs[hard_mask], masks=low_res_masks[hard_mask], text=text_prompt_embeddings[hard_mask]
                        )
                        image_emb_hard = image_embeddings[hard_mask]          # [K, C, H', W']                 # [K, …]
                        refine_text = text_prompt_embeddings[hard_mask]  # [K, 1, hidden_dim]
                        refine_out = self.SEG.mask_decoder(
                            image_embeddings=image_emb_hard,
                            image_pe=dense_pe,
                            sparse_prompt_embeddings=refine_sparse,
                            dense_prompt_embeddings=refine_dense,
                            text_prompt_embeddings=refine_text,
                            multimask_output=self.multimask_output,
                        )
                        # Upsample and apply activation.
                        refine_low = refine_out['low_res_masks']              # [K,1,h,w]
                        refine_up  = F.interpolate(refine_low, size=self.image_size, mode='bilinear', align_corners=False)
                        refine_sig = torch.sigmoid(refine_up)                 # [K,1,H,W]

                        # Recompute Dice.
                        refine_flat = refine_sig.flatten(1)                   # [K, H*W]
                        gt_flat2    = image_labels[hard_mask].flatten(1)      # [K, H*W]

                        inter2 = (refine_flat * gt_flat2).sum(dim=1)
                        union2 = refine_flat.sum(dim=1) + gt_flat2.sum(dim=1)
                        dice2  = 1 - (2 * inter2 + 1e-6) / (union2 + 1e-6)

                        valid_hard   = valid_logits[hard_mask]                # [K]
                        refine_loss  = (dice2 * valid_hard).sum() / (valid_hard.sum().clamp(min=1.0))
                        seg_loss_all += refine_loss
                        # seg_cnt += hard_num
                        # seg_loss_all = seg_loss_all / (seg_cnt.clamp(min=1.0))
            
            
            cls_loss = torch.zeros((), device=images.device)
            if (self.cls_head_final is not None):
                labels = class_labels.float().to(images.device)
                cls_loss = self.cls_loss_fn(cls_logits, labels)
                if 'focal' in self.clsloss_type and 'focalonly' not in self.clsloss_type:
                    focal_loss = sigmoid_focal_loss(
                        cls_logits,           # logits, shape [N, C] or [N]
                        labels,          # labels, same shape as logits, values in {0,1}
                        alpha=0.25,       # positive/negative balancing coefficient
                        gamma=2.0,        # focusing parameter (larger values emphasize hard samples)
                        reduction="mean"  # "none" | "mean" | "sum"
                    )
                    cls_loss += focal_loss
                elif self.clsloss_type == 'focalonly':
                    focal_loss = sigmoid_focal_loss(
                        cls_logits,           # logits, shape [N, C] or [N]
                        labels,          # labels, same shape as logits, values in {0,1}
                        alpha=0.25,       # positive/negative balancing coefficient
                        gamma=2.0,        # focusing parameter (larger values emphasize hard samples)
                        reduction="mean"  # "none" | "mean" | "sum"
                    )
                    cls_loss = focal_loss
                elif self.clsloss_type == 'focalonlygamma4':
                    focal_loss = sigmoid_focal_loss(
                        cls_logits,           # logits, shape [N, C] or [N]
                        labels,          # labels, same shape as logits, values in {0,1}
                        alpha=0.25,       # positive/negative balancing coefficient
                        gamma=4.0,        # focusing parameter (larger values emphasize hard samples)
                        reduction="mean"  # "none" | "mean" | "sum"
                    )
                    cls_loss = focal_loss
                
        else:
            ''''''''''''
            box_embeddings = None
            # print(f"Validating with None box_embeddings prompts")
            ''''''''''''
            
            sparse_embeddings = torch.empty((B, 0, self.SEG.image_encoder.encoder_embed_dim), device=image_embeddings.device)
            if point_embeddings is not None:
                sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            if box_embeddings is not None:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
            if text_embeddings is not None:
                sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)
            
            
            SEG_outputs = self.SEG.mask_decoder(
                image_embeddings=image_embeddings,      # [B, C, H, W]
                image_pe=dense_pe,                      # [B, ...] or broadcastable to [B, C, H, W]
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,           # [B, 1, hidden_dim], or None if dense embeddings are absent
                text_prompt_embeddings=text_prompt_embeddings,  # [B, 1, hidden_dim]
                multimask_output=self.multimask_output,
            )
            low_res_masks = SEG_outputs['low_res_masks']   
            pred_masks = F.interpolate(low_res_masks, size=self.image_size, mode='bilinear', align_corners=False)
            out_masks = torch.sigmoid(pred_masks)
            
            '''------------------------------------------------human-in-the-loop point refinement-------------------------------------------------------------'''
            human_in_the_loop = False
            if human_in_the_loop :
                # print(f"Validating with human_in_the_loop")
                pred_flat = out_masks.view(B, -1)
                gts   = image_labels.unsqueeze(1)
                gt_flat   = gts.view(B, -1)
                inter = (pred_flat * gt_flat).sum(dim=1)
                union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
                dice_loss_per_sample  = 1 - (2*inter + 1e-6) / (union + 1e-6)   # [B]
                hard_mask, hard_num = self._get_hard_samples(dice_loss_per_sample, k=out_masks.shape[0], top_k_ratio=1.0)

                if hard_mask.sum() > 0:
                    gt_hard   = image_labels[hard_mask]
                    pred_bin  = (out_masks[hard_mask] > 0.5).float()

                    refine_points, refine_labels = self.get_points_from_xor_masks(
                        gt_hard_masks=(gt_hard),          # torch Tensor [K,H,W], where K is the number of hard samples
                        pred_bin_masks=(pred_bin.squeeze(1)>0).int(), 
                        get_point=1
                    )

                    refine_points = refine_points.to(images.device)  # [K, 2]
                    refine_labels = refine_labels.to(images.device)
                    
                    use_mask = random.choice([True, False, False])
                    selected_mask = low_res_masks[hard_mask] if use_mask else None
                    refine_sparse, refine_dense, _, _, _ = self.SEG.prompt_encoder(
                        points=(refine_points, refine_labels),
                        boxes=bboxs[hard_mask],
                        masks=selected_mask,
                        text=text_prompt_embeddings[hard_mask]
                    )
                    
                    image_emb_hard = image_embeddings[hard_mask]          # [K, C, H', W']                 # [K, …]
                    refine_text = text_prompt_embeddings[hard_mask]  # [K, 1, hidden_dim]
                    refine_out = self.SEG.mask_decoder(
                        image_embeddings=image_emb_hard,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=refine_sparse,
                        dense_prompt_embeddings=refine_dense,
                        text_prompt_embeddings=refine_text,
                        multimask_output=self.multimask_output,
                    )
                    # Upsample and apply activation.
                    refine_low = refine_out['low_res_masks']              # [K,1,h,w]
                    refine_up  = F.interpolate(refine_low, size=self.image_size, mode='bilinear', align_corners=False)
                    refine_sig = torch.sigmoid(refine_up)                 # [K,1,H,W]
                    out_masks = refine_sig

            return {
                "pred_masks": out_masks,
                "gt_masks": image_labels,
                "cls_logits": cls_logits,
                "cls_labels": class_labels,
            }
    
        
        total_loss = self.lm_weight * lm_loss + seg_loss_all + cls_loss

        return {
            "loss": total_loss,   
            "lm_loss": lm_loss,   
            "seg_loss": seg_loss_all,
            "cls_loss": cls_loss,
        }
    @torch.no_grad()
    def generate_with_prompt(
        self,
        point_coords=None,             # [B,N,2]
        point_labels=None,             # [B,N]
        bboxs=None,                    # [B,4]
        prepare_vl_inputs=None,        # dict from shape_input()
        max_new_tokens=128,
        use_cache=True,
        **generate_kwargs
    ):  

        gen_ids = self.VL.generate(
            input_ids=prepare_vl_inputs["input_ids"],
            attention_mask=prepare_vl_inputs["attention_mask"],
            pixel_values=prepare_vl_inputs["pixel_values"],
            image_grid_thw=prepare_vl_inputs["image_grid_thw"],
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            **generate_kwargs
        )
        n_prompt = 0
        return gen_ids, n_prompt 
    
