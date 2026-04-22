import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["NCCL_TIMEOUT"] = "5000"
import argparse
import sys
import time
import math
import cv2
import numpy as np
import json
import re
import string
import shutil
from collections import defaultdict

from functools import partial
import deepspeed
import torch
import tqdm
import torch.distributed as dist
from datetime import timedelta

from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import logging as transformers_logging
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM
# from model.deepseek_vl2.models import DeepseekVLV2Processor, conversation as conversation_lib
from transformers import AutoProcessor

from torch.utils.data import DataLoader
from utils.dataset import HybridDataset, ValDataset_seg, ValDataset_vqa, collate_fn, DistributedGroupSampler

from utils.utils import set_seed, lora_assemble, log_parameter_info, shape_input
from utils.utils import AverageMeter, ProgressMeter,parse_bbox, calculate_iou
from utils.utils import DEFAULT_SEG_Task_TOKEN, DEFAULT_DET_Task_TOKEN, DEFAULT_PROMPT_TOKEN, gather_all, GLOBAL_SEED, DEFAULT_POINT_TOKEN,DEFAULT_PLABEL_TOKEN,DEFAULT_BBOX_TOKEN, DEFAULT_CLS_Task_TOKEN
from utils.utils import seed_worker, compute_flops_with_profiler
import logging
import copy
import random
import evaluate
import gc
from typing import List
from collections import Counter
import math
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(False)
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args):
    parser = argparse.ArgumentParser(description="Chat2Tail")
    # general config
    parser.add_argument("--version", default="weight/HuatuoGPT-Vision-7B-Qwen2.5VL")
    parser.add_argument("--vision_pretrained", default="./runs/pretrainseg4_bp_224_1e-05_BG62_RA816_lmw1.0_sr1_mpT/hf_model/pytorch_model.bin", type=str)
    parser.add_argument("--conv_type", default="Qwen", type=str)
    # model config
    parser.add_argument("--image_size", default=256, type=int, help="image size")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.01, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", type=str) # Added o_proj and MLP layers

    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.98, type=float)
    parser.add_argument("--out_dim", default=768, type=int)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    # dataset config
    parser.add_argument("--dataset_dir", default="dataset", type=str)
    parser.add_argument("--sample_rates", default="1,1", type=str)
    parser.add_argument("--dataset", default="refer_seg||vqa", type=str)
    parser.add_argument("--refer_seg_data", default="IMed361M", type=str)
    parser.add_argument("--vqa_data", default="PubMedVision||MIMIC||SLAKE||PMC_VQA||vqa-rad||path-vqa", type=str)   
    parser.add_argument("--val_dataset", default="refer_seg||vqa", type=str)
    
    # training config
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--divided_save", default=0.025, type=float) # 0.025
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=16, type=int)
    parser.add_argument("--grad_accumulation_steps", default=2, type=int)
    parser.add_argument("--workers", default=12, type=int) 
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--lm_weight", default=1.0, type=float)
    parser.add_argument("--promptype", default="tpb", type=str)

    # other config
    parser.add_argument("--transfer_HF", action="store_true", default=False, help="transfer to HF weights")
    parser.add_argument("--finetune", action="store_true", default=False, help="Finetune the model using saved weights with additional linear paremeters.")
    parser.add_argument("--aware_loss", action="store_true", default=False, help="Use aware_loss.")
    parser.add_argument("--lambda_uncertainty", default=1.0, type=float)
    parser.add_argument("--tail_loop", action="store_true", default=False, help="Use mask prompt for SEG module.")
    
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--reset_optimizer", action="store_true", default=False, help="Reset optimizer when resuming training.")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="debug", type=str)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--openclosetype", default="all", type=str)
    parser.add_argument("--modality", default='', type=str)
    parser.add_argument("--calc_flops", action="store_true", help="If set, profile one (or a few) forward pass(es) to estimate FLOPs before training.")
    parser.add_argument("--cls_num", default=0, type=int, help="Number of classes for classification head. If 0, classification head will not be used.")
    parser.add_argument("--lineartype", default="seg", type=str, help="seg or combine")
    parser.add_argument("--linearprobing", action="store_true", default=False)
    parser.add_argument("--clsloss_type", default=None, type=str, help="")
    

    parser.add_argument("--early_stop", default=0, type=int, help="Early stopping patience in epochs. If 0, no early stopping.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    
    parser.add_argument("--lr_seg", default=None, type=float, help="Learning rate for SEG module. Defaults to --lr if None.")
    parser.add_argument("--lr_vl", default="5e-6", type=float, help="Learning rate for VL module. Defaults to --lr if None.")
    parser.add_argument("--lr_cls", default=None, type=float, help="Learning rate for VL module. Defaults to --lr if None.")
    

    return parser.parse_args(args)

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def deepspeed_init_speciallr(model, train_dataset, args): 
    # Define learning rates with defaults.
    lr_seg = args.lr_seg if args.lr_seg is not None else args.lr
    lr_vl = args.lr_vl if args.lr_vl is not None else args.lr
    lr_cls = args.lr_cls if args.lr_cls is not None else args.lr
    lr_main = args.lr

    # Parameter groups.
    param_groups = []
    seg_params, vl_params, cls_params, other_params = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "SEG." in name:
            seg_params.append(param)
        elif "VL." in name:
            vl_params.append(param)
        elif "cls_head" in name:
            cls_params.append(param)
        else:
            other_params.append(param)

    # Add parameter groups with dedicated learning rates.
    if seg_params:
        param_groups.append({"params": seg_params, "lr": lr_seg})
        if is_main_process():
            print(f"SEG module parameters → LR: {lr_seg}")

    if vl_params:
        param_groups.append({"params": vl_params, "lr": lr_vl})
        if is_main_process():
            print(f"VL module parameters → LR: {lr_vl}")

    if cls_params:
        param_groups.append({"params": cls_params, "lr": lr_cls})
        if is_main_process():
            print(f"Classification heads → LR: {lr_cls}")

    if other_params:
        param_groups.append({"params": other_params, "lr": lr_main})
        if is_main_process():
            print(f"Other parameters → LR: {lr_main}")

    # Fallback in case no parameter groups are found.
    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": lr_main}]


    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.01,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.cal_step * args.epochs,
                "warmup_min_lr": args.lr * 0.05,
                "warmup_max_lr": args.lr,
                "warmup_type": "cosine" if args.finetune else "linear",
                "warmup_num_steps": int(args.cal_step * args.epochs * 0.03) if args.finetune else 500,
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0,              # Enable dynamic loss scaling.
            "loss_scale_window": 1000,    # Optional: adjust the scaling window.
            "hysteresis": 2,              # Optional: hysteresis parameter.
            "min_loss_scale": 1           # Optional: minimum loss scale.
        },
        "bf16": {"enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,  # Reduced from 1e8 to improve stability
            "allgather_bucket_size": 5e7,  # Reduced from 1e8 to improve stability
            "round_robin_gradients": True,  # Add round-robin gradient distribution
        },
    }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        model_parameters=param_groups,
        config=ds_config,
        )
    return model_engine, optimizer, _, scheduler

def main(cli_args):
    args = parse_args(cli_args)
    
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60)) 
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    args.distributed = world_size > 1
    args.device = torch.device("cuda", args.local_rank)
    if args.test:
        args.valid = True
    
    if "puresam" in args.exp_name:
        args.lora_r = 0
        args.image_size = 1024
        args.val_dataset = "refer_seg"
        args.dataset = "refer_seg"
        args.test = True
        
    if 'release' in args.exp_name:
        exp_name_orig = args.exp_name
        args.lr_vl = args.lr_vl if args.lr_vl is not None else args.lr
        sep_idx = min([exp_name_orig.find(sep) if exp_name_orig.find(sep) != -1 else len(exp_name_orig) for sep in ["_", "-"]])
        exp_name_mod = exp_name_orig[:sep_idx] + f"{world_size}" + exp_name_orig[sep_idx:]
        args.exp_name = f"{exp_name_mod}_{args.lr:1.0e}_{args.lr_vl:1.0e}_B{args.batch_size}"
        args.exp_name += f"_sr{args.sample_rates}"
        args.exp_name += f"_tl{str(args.tail_loop)[0]}"
        if args.aware_loss:
            args.exp_name += f"_awl{str(args.lambda_uncertainty)}"
      
    if args.finetune and args.test:
        args.lora_r = 0
    
    
    if "pretrainllm" in args.exp_name or "purellm" in args.exp_name:
        print("\n\npretrainllm\n\n")
        from model.Chat2Tail_pretrainllm import Chat2TailConfig, Chat2TailForCausalLMStandard
    elif "pretrainseg" in args.exp_name or "puresam" in args.exp_name:
        print("\n\puresam\n\n")
        from model.Chat2Tail_pretrainsam import Chat2TailConfig, Chat2TailForCausalLMStandard
    else:
        print("\n\standardset\n\n")
        from model.Chat2Tail import Chat2TailConfig, Chat2TailForCausalLMStandard
    
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if is_main_process() and args.test == False:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Initialize the processor and tokenizer.
    if "Qwen" in args.version or "pretrainllm" in args.version:
        args.conv_type = "Qwen"
        path = args.version
        # vl_chat_processor = AutoProcessor.from_pretrained(path, model_max_length=2048, cache_dir='/cache_dir', padding_side="right", use_fast=False,)
        vl_chat_processor = AutoProcessor.from_pretrained(path, model_max_length=2048, max_pixels=args.image_size**2*3)
        vl_chat_processor.tokenizer.padding_side = 'left'
        conv = None
    else:
        raise ValueError(f"Unsupported version: {args.version}")
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SEG_Task_TOKEN, DEFAULT_DET_Task_TOKEN, DEFAULT_PROMPT_TOKEN, DEFAULT_POINT_TOKEN, DEFAULT_PLABEL_TOKEN, DEFAULT_BBOX_TOKEN]}
    # special_tokens_dict = {"additional_special_tokens": [DEFAULT_SEG_Task_TOKEN, DEFAULT_DET_Task_TOKEN, DEFAULT_PROMPT_TOKEN, DEFAULT_POINT_TOKEN, DEFAULT_PLABEL_TOKEN, DEFAULT_BBOX_TOKEN, DEFAULT_CLS_Task_TOKEN]}
    vl_chat_processor.tokenizer.add_special_tokens(special_tokens_dict)
    args.seg_token_idx = vl_chat_processor.tokenizer.convert_tokens_to_ids(DEFAULT_SEG_Task_TOKEN)
    args.det_token_idx = vl_chat_processor.tokenizer.convert_tokens_to_ids(DEFAULT_DET_Task_TOKEN)
    args.prompt_token_idx = vl_chat_processor.tokenizer.convert_tokens_to_ids(DEFAULT_PROMPT_TOKEN)
    if 'token' in args.lineartype:
        args.cls_token_idx = vl_chat_processor.tokenizer.convert_tokens_to_ids(DEFAULT_CLS_Task_TOKEN)
        cls_token = DEFAULT_CLS_Task_TOKEN
    else:
        args.cls_token_idx = -1
        cls_token = None

    args.point_token_idx = vl_chat_processor.tokenizer.convert_tokens_to_ids(DEFAULT_POINT_TOKEN)
    args.plabel_token_idx = vl_chat_processor.tokenizer.convert_tokens_to_ids(DEFAULT_PLABEL_TOKEN)
    args.bbox_token_idx = vl_chat_processor.tokenizer.convert_tokens_to_ids(DEFAULT_BBOX_TOKEN)


    if is_main_process():
        print(
            "Added SEG-related tokens:\n"
            f"{DEFAULT_SEG_Task_TOKEN}: {vl_chat_processor.tokenizer.encode(DEFAULT_SEG_Task_TOKEN, add_special_tokens=False)[0]}\n"
            f"{DEFAULT_DET_Task_TOKEN}: {vl_chat_processor.tokenizer.encode(DEFAULT_DET_Task_TOKEN, add_special_tokens=False)[0]}\n"
            f"{DEFAULT_PROMPT_TOKEN}: {vl_chat_processor.tokenizer.encode(DEFAULT_PROMPT_TOKEN, add_special_tokens=False)[0]}\n"
            f"{DEFAULT_POINT_TOKEN}: {vl_chat_processor.tokenizer.encode(DEFAULT_POINT_TOKEN, add_special_tokens=False)[0]}\n"
            f"{DEFAULT_PLABEL_TOKEN}: {vl_chat_processor.tokenizer.encode(DEFAULT_PLABEL_TOKEN, add_special_tokens=False)[0]}\n"
            f"{DEFAULT_BBOX_TOKEN}: {vl_chat_processor.tokenizer.encode(DEFAULT_BBOX_TOKEN, add_special_tokens=False)[0]}\n"
            f"{DEFAULT_CLS_Task_TOKEN}: {vl_chat_processor.tokenizer.encode(DEFAULT_CLS_Task_TOKEN, add_special_tokens=False)[0]}\n"
        )
        
    # Set the target dtype.
    # torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.half}.get(args.precision, torch.float32)
    torch_dtype= torch.float32
    # torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision, torch.float32)

    if args.finetune and args.cls_num == 0:
        if "isic2018" in args.vqa_data or "isic2019" in args.vqa_data:
            args.cls_num = 7
        elif "cxr-lt" in args.vqa_data:
            args.cls_num = 40
    config = Chat2TailConfig(
                            seg_token_idx = args.seg_token_idx,
                            det_token_idx = args.det_token_idx,
                            prompt_token_idx = args.prompt_token_idx,
                            cls_token_idx = args.cls_token_idx,
                            aware_loss= args.aware_loss,
                            tail_loop= args.tail_loop,
                            vision_pretrained = args.vision_pretrained,
                            version = args.version,
                            image_size = args.image_size,
                            out_dim = args.out_dim,
                            lm_weight = args.lm_weight,
                            promptype= args.promptype,
                            lambda_uncertainty=args.lambda_uncertainty,
                            cls_num = args.cls_num,
                            clsloss_type = args.clsloss_type,
                            )
    
    model = Chat2TailForCausalLMStandard(config=config, processor=vl_chat_processor, lineartype=args.lineartype)
    
    if args.finetune:
        ft_path = './runs/' + args.resume if args.test == False and args.resume != "" else './runs/' + args.exp_name
        hf_bin = os.path.join(ft_path, "hf_model/pytorch_model.bin")
        print(f"[finetune] loading weights from {hf_bin}")
        state_dict = torch.load(hf_bin, map_location="cpu")
        ck_vocab_size = state_dict["VL.model.embed_tokens.weight"].shape[0]
        model.VL.resize_token_embeddings(ck_vocab_size)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[finetune] missing keys: {missing}")
        print(f"[finetune] unexpected keys: {unexpected}")
        model.to(dtype=torch_dtype, device=args.device)

        hf_bin = os.path.join(ft_path, "hf_model/pytorch_model.bin")
        print(f"[finetune] loading weights from {hf_bin}")
        state_dict = torch.load(hf_bin, map_location="cpu")
        ck_vocab_size = state_dict["VL.model.embed_tokens.weight"].shape[0]
        model.VL.resize_token_embeddings(ck_vocab_size)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[finetune] missing keys: {missing}")
        print(f"[finetune] unexpected keys: {unexpected}")
        model.to(dtype=torch_dtype, device=args.device)

    if "puresam" not in args.exp_name:
        model.VL.resize_token_embeddings(len(vl_chat_processor.tokenizer))

    model.to(dtype=torch_dtype, device=args.device)

    # Freeze selected submodules.
    try:
        for p in model.SEG.parameters():
            p.requires_grad = False
    except:
        print('no model.SEG module!!')
    try:
        for p in model.VL.parameters():
            p.requires_grad = False
        if "Qwen" in args.version:
            args.tune_mm_vision = True
            args.tune_mm_mlp = True
            args.tune_mm_llm = False
            if args.tune_mm_vision:
                for n, p in model.VL.visual.named_parameters():
                    p.requires_grad = True
            else:
                for n, p in model.VL.visual.named_parameters():
                    p.requires_grad = False

            if args.tune_mm_mlp:
                for n, p in model.VL.visual.merger.named_parameters():
                    p.requires_grad = True
            else:
                for n, p in model.VL.visual.merger.named_parameters():
                    p.requires_grad = False

            if args.tune_mm_llm:
                for n, p in model.VL.model.named_parameters():
                    p.requires_grad = True
                model.VL.lm_head.requires_grad = True
            else:
                for n, p in model.VL.model.named_parameters():
                    p.requires_grad = False
                # model.VL.lm_head.requires_grad = False   # Original Qwen setting.
                model.VL.lm_head.requires_grad = True 
    except:
        print('no model.VL module!!')
    
    # full_weight_names = ["lm_head", "embed_tokens", "patch_embed","visual.merger", "SEG.prompt_encoder", "SEG.mask_decoder", "seg_fc", "text_fc", "fusion_fc", "add"]
    # lora_except_weight = ["SEG"]
    # full_weight_names = ["lm_head", "embed_tokens", "patch_embed","VL.visual", "SEG.prompt_encoder", "seg_fc", "text_fc", "fusion_fc", "add", "cls_head"]
    full_weight_names = ["lm_head", "embed_tokens", "patch_embed","VL.visual", "SEG", "seg_fc", "text_fc", "fusion_fc", "add", "cls_head"]
    if args.finetune:
        full_weight_names = ["lm_head", "embed_tokens", "patch_embed", "seg_fc", "text_fc", "fusion_fc", "add", "cls_head", "SEG.image_encoder"]
    if 'jbhiround1' in args.resume:
        full_weight_names = ['lm_head', 'embed_tokens', 'patch_embed', 'visual.merger', 'SEG.prompt_encoder', 'seg_fc', 'text_fc', 'fusion_fc', 'add']
    if 'hotweight' in args.exp_name:
        full_weight_names = ["lm_head", "embed_tokens", "patch_embed", "seg_fc", "text_fc", "fusion_fc", "add", "cls_head", "SEG.image_encoder", "VL.visual"]
    lora_except_weight = None

    if args.finetune and args.linearprobing:
        args.lora_r = 0
        full_weight_names = ["cls_head_final"]
        for p in model.parameters():
            p.requires_grad = False
        for p in model.cls_head_final.parameters():
            p.requires_grad = True
    elif args.lora_r > 0:
        if args.test and args.finetune:
            print("[LoRA] test finetune set lora_r to 0")
        else:
            except_weight = lora_except_weight + full_weight_names if lora_except_weight else full_weight_names
            model = lora_assemble(model, except_weight=except_weight, args=args)    
    log_parameter_info(model, full_weight_names, lora_except_weight, args, args.log_dir)

    sample_rates = [int(x) for x in args.sample_rates.split(",")]

    
    train_dataset = HybridDataset(
        base_image_dir=args.dataset_dir,
        chat_processor=vl_chat_processor,
        image_size=args.image_size,
        dataset=args.dataset,
        sample_rates=sample_rates,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        conv=conv,
        torch_dtype=torch_dtype,
        promptype=args.promptype, 
        cls_num = args.cls_num,
        cls_token = cls_token,
    )
    train_sampler = DistributedGroupSampler(
        group_indices=train_dataset.group_indices,
        group_weights=train_dataset.group_weights,
        sample_rates=sample_rates,
        num_replicas=world_size,
        rank=args.local_rank,
        shuffle=True,
        seed=GLOBAL_SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.workers,
        prefetch_factor=6,   
        persistent_workers=True,       # Keep workers alive across iterations.
        worker_init_fn = seed_worker,
        collate_fn=partial(collate_fn, processor=vl_chat_processor, valid=False, device=args.device),
        drop_last=True,
    )

    
    if "refer_seg" in args.val_dataset:
        val_dataset_seg = ValDataset_seg(
            base_image_dir=args.dataset_dir,
            chat_processor=vl_chat_processor,
            dataset=args.refer_seg_data,
            image_size=args.image_size,
            conv=conv,
            torch_dtype=torch_dtype,
            test = args.test,
            cls_num = args.cls_num,
            
        )
        
        val_sampler_seg = (
            torch.utils.data.distributed.DistributedSampler(
                val_dataset_seg,
                num_replicas=world_size, 
                shuffle=False,
                drop_last=False
            )
            if world_size > 1
            else None
        )

        val_loader_seg = torch.utils.data.DataLoader(
            val_dataset_seg, 
            batch_size=args.val_batch_size, 
            shuffle=False, 
            num_workers=args.workers,
            prefetch_factor=6,   
            worker_init_fn=seed_worker,
            pin_memory=True, sampler=val_sampler_seg, collate_fn=partial(collate_fn, processor=vl_chat_processor, valid=True, device=args.device),)
    
    
    if "vqa" in args.val_dataset:
        vqa_data = args.vqa_data
        val_dataset_vqa = ValDataset_vqa(
            base_image_dir=args.dataset_dir,
            chat_processor=vl_chat_processor,
            dataset=vqa_data,
            image_size=args.image_size,
            conv=conv,
            torch_dtype=torch_dtype,
            test = args.test,
            modality = args.modality,
            openclosetype = args.openclosetype,
            cls_num = args.cls_num,
            cls_token = cls_token
        )
        
        val_sampler_vqa = (
            torch.utils.data.distributed.DistributedSampler(
                val_dataset_vqa,
                num_replicas=world_size, 
                shuffle=False,
                drop_last=False
            )
            if world_size > 1
            else None
        )

        val_loader_vqa = torch.utils.data.DataLoader(
            val_dataset_vqa, 
            batch_size=args.val_batch_size, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True,
            prefetch_factor=6,   
            worker_init_fn=seed_worker,
            sampler=val_sampler_vqa, collate_fn=partial(collate_fn, processor=vl_chat_processor, valid=True, device=args.device,promptype=args.promptype),)

    args.cal_step = len(train_loader)
    args.divided_save = int(args.cal_step * args.divided_save)
    
    if "debug" in args.exp_name:
        args.divided_save = 16
        args.print_freq = 1

    '''Estimate FLOPs'''
    if args.calc_flops and is_main_process():
        try:        
            input_dict = next(iter(train_loader))
            sample = shape_input(input_dict, args.device, args.precision)
            want_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.precision, torch.float32)
            if any(p.is_floating_point() and p.dtype != want_dtype for p in model.parameters()):
                model.to(want_dtype)

            compute_flops_with_profiler(model, sample, args)
        except Exception as e:
            print(f"[FLOPs] Estimation failed: {e}")
    '''Estimate FLOPs'''
    # 128 tailT 16.745 TFLOPs/ tailF 16.744 TFLOPs
    # 176 tailT 17.854 TFLOPs/ tailF 17.853 TFLOPs
    # 224 tailT 20.624 TFLOPs/ tailF 20.622 TFLOPs
    # 256 tailT 22.443 TFLOPs/ tailF 22.441 TFLOPs

    model_engine, _, _, scheduler = deepspeed_init_speciallr(model, train_dataset, args)
    global_step = 0
    best_score = 0.0
    resume_epoch = 0
    resume_interval_idx = 0
    resume_batches_in_interval = 0
    save_dir = os.path.join(args.log_dir, "ckpt_model")

    if args.resume or os.path.isdir(save_dir):
        if args.finetune:
            print("Finetune not loaded new weight.")
        else:
            load_dir = "./runs/" + args.resume + "/ckpt_model" if args.resume else save_dir
            if args.reset_optimizer:
                load_path, client_states = model_engine.load_checkpoint(load_dir, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=False)
                if is_main_process():
                    print(f"Loaded weights from {load_path}, optimizer/scheduler reset, starting from scratch.")
            else:
                load_path, client_states = model_engine.load_checkpoint(load_dir, tag=None, load_module_only=False)
                global_step = client_states.get("global_step", global_step)
                resume_epoch = client_states.get("epoch", resume_epoch)
                best_score = client_states.get("best_score", best_score)
                resume_interval_idx = client_states.get("resume_interval_idx", 0)
                resume_batches_in_interval  = client_states.get("resume_batches_in_interval", 0)
                
                
            print(f"Loaded weights from {load_path}, optimizer/scheduler states loaded.")
            if is_main_process():
                print(f"Loaded ckpt from {load_path}, epoch={resume_epoch}, seg_idx={resume_interval_idx}, batch_in_seg={resume_batches_in_interval}, global_step={global_step}")
                if "vqa" in args.val_dataset:
                    print(f"World_size {world_size}, Training with {len(train_dataset)} examples and validating with {len(val_dataset_vqa)} VQA examples.")
                if "refer_seg" in args.val_dataset:
                    print(f"World_size {world_size}, Training with {len(train_dataset)} examples and validating with {len(val_dataset_seg)} Segmentation examples.")
                print(f"Epoch steps: {args.cal_step}, save model every {args.divided_save} steps with {args.divided_save // (args.print_freq)} print times.")
    if args.distributed:
        dist.barrier()

    '''Start loop'''        
    save_count = 0
    
    if args.transfer_HF:
        if is_main_process():
            model_to_save = copy.deepcopy(model_engine.module)
            if args.lora_r > 0:
                model_to_save = model_to_save.merge_and_unload().cpu().float()
            output_dir = os.path.join(args.log_dir, "hf_model")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            # Save in Hugging Face format.
            # model_to_save.save_pretrained(output_dir, safe_serialization=False)
            # vl_chat_processor.tokenizer.save_pretrained(output_dir)
            # print(f"Saved HF model to {output_dir}")
            
            bin_output_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), bin_output_path)
            print(f"Saved model state_dict to {bin_output_path}")
            del model_to_save
            torch.cuda.empty_cache()

            # if os.path.exists(save_dir):
            #     shutil.rmtree(save_dir)
        dist.barrier()
        print("Model converted to Hugging Face format and saved. Exiting...")
        return
        
        

    if args.test:
        if args.resume == '':
            resume_version = args.exp_name
        else:
            resume_version = args.resume if args.finetune == False else args.exp_name
        with torch.no_grad():
            if "pretrainllm" not in args.exp_name:
                if "refer_seg" in args.val_dataset:
                    dice, iou = 0.0, 0.0
                    det_iou = 0.0
                    print("Running segmentation validation in test mode...")
                    dice, iou = valid_seg(model_engine, val_loader_seg, global_step, 0, writer, args, vl_chat_processor)
                    # det_iou = valid_det(model_engine, val_loader_seg, global_step, 0, writer, args, vl_chat_processor)
                    if is_main_process():
                        print(f"Test Results - Dice: {dice:.4f}, IoU: {iou:.4f}, Detection IoU: {det_iou:.4f}")

                if "vqa" in args.val_dataset:
                    print("Running VQA validation in test mode...")
                    bleu_score, rouge_score, acc, f1, recall, vqa_extra = valid_vqa(model_engine, val_loader_vqa, global_step, 0, writer, args, vl_chat_processor)
                    if is_main_process() and args.test:
                        print(f"Test Results - BLEU: {bleu_score:.4f}, ROUGE1: {rouge_score:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
                        save_path = os.path.join(args.log_dir)
                        os.makedirs(save_path, exist_ok=True)
                        metrics = {
                            "resume":        resume_version,
                            "val_dataset":   args.val_dataset,
                            "vqa_data":      args.vqa_data,
                        }
                        metrics.update(vqa_extra)
                        if "SLAKE" in args.vqa_data or "vqa-rad" in args.vqa_data or "path-vqa" in args.vqa_data or "kvasir_vqa":
                            args.openclosetype = args.openclosetype
                        else:
                            args.openclosetype = 'all'
                            
                        if "PubMedVision" in args.vqa_data or 'PMC_VQA' in args.vqa_data or 'SLAKE' in args.vqa_data:
                            args.modality = args.modality
                        else:
                            args.modality = 'all'
                        
                        fn = os.path.join(save_path + '/' + args.vqa_data + "_" + args.openclosetype + "_" + args.modality + "_metrics.json")
                            
                        with open(fn, "w") as f:
                            json.dump(metrics, f, indent=4)
                        print(f"Saved test metrics to {fn}")
    else:
        print(args.exp_name)
        skipped_resume = False
        args.start_epoch = resume_epoch if resume_epoch > 0 else args.start_epoch
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_iter = iter(train_loader)

            # if not skipped_resume and epoch == resume_epoch and resume_batches_in_interval > 0:   # here need to be fixed
            if not skipped_resume and epoch == 50 and resume_batches_in_interval > 0:
                # 1) Record the segment index to resume from.
                current_save_idx = resume_interval_idx + 1

                # 2) Compute how many micro-batches and samples to skip.
                total_skip_batches = resume_interval_idx * args.divided_save + resume_batches_in_interval
                total_skip_samples = total_skip_batches * args.batch_size

                # 3) Fast-forward the sampler.
                all_idx = list(train_sampler)
                new_idx = all_idx[total_skip_samples:]
                train_sampler = SubsetRandomSampler(new_idx)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    sampler=train_sampler,
                    num_workers=args.workers,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=6,
                    worker_init_fn=seed_worker,
                    collate_fn=partial(collate_fn, processor=vl_chat_processor, valid=False, device=args.device,promptype=args.promptype),
                    drop_last=True,
                )
                train_iter = iter(train_loader)
                skipped_resume = True

                if is_main_process():
                    print(f"Resuming epoch {epoch}. "
                        f"Skipped {resume_interval_idx} full segments and "
                        f"{resume_batches_in_interval} batches, "
                        f"starting from segment index {current_save_idx}.")
            else:
                current_save_idx = 0
            
            total_saves_per_epoch = math.ceil(args.cal_step / args.divided_save)
            print(f"Epoch {epoch} starts, total_saves_per_epoch: {total_saves_per_epoch}, starting from current_save_idx: {current_save_idx}")
            try:
                for seg_idx in range(current_save_idx, total_saves_per_epoch-1):

                    micro_steps = args.divided_save
                    
                    # Save the current seed.
                    fn = os.path.join(args.log_dir + '/' + "seed.txt")
                    with open(fn, "w") as f:
                        f.write(str(args.seed))
                    # print(f"Saved seed to {fn}")
                        
                    if not args.test:
                        time_train = time.time()
                        model_engine, train_iter, global_step, batch_in_epoch = train(model_engine, scheduler, train_iter, global_step, epoch, writer, args, vl_chat_processor.tokenizer,micro_steps)
                        torch.cuda.empty_cache()
                        if is_main_process():
                            writer.add_scalar("time/train", (time.time() - time_train)//60, global_step)

                    with torch.no_grad():
                        dice, iou = 0.0, 0.0
                        bleu_score, rouge_score = 0.0, 0.0
                        if "refer_seg" in args.val_dataset:
                            time_seg = time.time()
                            dice, iou = valid_seg(model_engine, val_loader_seg, global_step, epoch, writer, args, vl_chat_processor)
                            if is_main_process():
                                writer.add_scalar("time/valid_seg", (time.time() - time_seg)//60, global_step)
                            torch.cuda.empty_cache()
                        if "vqa" in args.val_dataset:
                            time_vqa = time.time()
                            bleu_score, rouge_score, acc, f1, recall, vqa_extra = valid_vqa(model_engine, val_loader_vqa, global_step, epoch, writer, args, vl_chat_processor)
                            if is_main_process():
                                writer.add_scalar("time/valid_vqa", (time.time() - time_vqa)//60, global_step)
                            torch.cuda.empty_cache()

                        combined_score = dice + iou + bleu_score + rouge_score
                        if args.finetune:
                            auc_cls = vqa_extra.get("auc_cls", None)
                            f1_cls = vqa_extra.get("f1_cls", None)
                            precision_cls = vqa_extra.get("precision_cls", None)
                            recall_cls = vqa_extra.get("recall_cls", None)
                            combined_score = auc_cls + precision_cls + recall_cls + f1_cls
                        
                        if is_main_process():
                            writer.add_scalar("time/combined_score", combined_score, global_step)
                        save_best_model = combined_score >= best_score
                        best_score = max(combined_score, best_score)

                        client_state = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_score": best_score,
                            "resume_interval_idx": seg_idx,
                            "resume_batches_in_interval": batch_in_epoch,
                        }
                        
                        if args.finetune == False:
                            ckpt_dir = os.path.join(args.log_dir, "ckpt_model")
                            if is_main_process():
                                if os.path.isdir(ckpt_dir):
                                    shutil.rmtree(ckpt_dir)
                                os.makedirs(ckpt_dir, exist_ok=True)
                            dist.barrier()
                            model_engine.save_checkpoint(ckpt_dir, client_state=client_state)
                            print(f"Saved last ckpt at seg_idx={seg_idx}, batch_in_seg={batch_in_epoch}")

                        if save_best_model:
                            if args.finetune:
                                if is_main_process():
                                    model_to_save = copy.deepcopy(model_engine.module)
                                    if args.lora_r > 0:
                                        model_to_save = model_to_save.merge_and_unload().cpu().float()
                                    output_dir = os.path.join(args.log_dir, "hf_model")
                                    if os.path.exists(output_dir):
                                        shutil.rmtree(output_dir)
                                    os.makedirs(output_dir, exist_ok=True)
                                    # Save in Hugging Face format.
                                    # model_to_save.save_pretrained(output_dir, safe_serialization=False)
                                    # vl_chat_processor.tokenizer.save_pretrained(output_dir)
                                    # print(f"Saved HF model to {output_dir}")
                                    
                                    bin_output_path = os.path.join(output_dir, "pytorch_model.bin")
                                    torch.save(model_to_save.state_dict(), bin_output_path)
                                    print(f"Saved model state_dict to {bin_output_path}")
                                    del model_to_save
                                    torch.cuda.empty_cache()
                            else:
                                save_dir = os.path.join(args.log_dir, "ckpt_best")
                                if is_main_process():
                                    if os.path.isdir(save_dir):
                                        shutil.rmtree(save_dir)
                                    os.makedirs(save_dir, exist_ok=True)
                                dist.barrier()
                                model_engine.save_checkpoint(save_dir, client_state=client_state)
                                if is_main_process():
                                    print(f"Saved ckpt at seg_idx={seg_idx}, batch_in_seg={batch_in_epoch}")
                                    
                        #  save last
                        if is_main_process():
                            model_to_save = copy.deepcopy(model_engine.module)
                            if args.lora_r > 0:
                                model_to_save = model_to_save.merge_and_unload().cpu().float()
                            output_dir = os.path.join(args.log_dir, "hf_model_last")
                            if os.path.exists(output_dir):
                                shutil.rmtree(output_dir)
                            os.makedirs(output_dir, exist_ok=True)                            
                            bin_output_path = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), bin_output_path)
                            print(f"Saved model state_dict to {bin_output_path}")
                            del model_to_save
                            torch.cuda.empty_cache()
                        dist.barrier()

                                
                        if args.early_stop > 0:
                            save_count += 1
                            if is_main_process():
                                print(f"[EarlyStop] Trained for {save_count} save intervals.")
                            if save_count >= args.early_stop:
                                if is_main_process():
                                    print(f"Reached early_stop={args.early_stop}; stopping training early.")
                                # Destroy the process group before exiting distributed training.
                                if args.distributed:
                                    dist.destroy_process_group()
                                sys.exit(0)
            
            except StopIteration:
                print(f"Epoch {epoch} dataloader exhausted on {args.local_rank}.")
                break
            resume_interval_idx      = 0
            resume_batches_in_interval = 0
    if args.distributed:
        dist.destroy_process_group()

def train(model_engine, scheduler, train_iter, global_step, epoch, writer, args, tokenizer, micro_steps):
    cost_time = AverageMeter("cost_time", ":.1f")
    loss_meter = AverageMeter("Loss", ":.3f")
    lm_loss_meter = AverageMeter("lm_loss", ":.3f")
    seg_loss_meter = AverageMeter("seg_loss", ":.3f")
    cls_loss_meter = AverageMeter("cls_loss", ":.3f")
    progress = ProgressMeter(args.cal_step, [cost_time, loss_meter, lm_loss_meter, seg_loss_meter, cls_loss_meter],
                               prefix=f"Epoch: [{epoch}/{args.epochs}]")

    model_engine.train()
    freq_start_time = time.time()
    last_print_step = global_step

    for step in range(micro_steps):
        # print(step, args.divided_save)
        # print("step:", step, args.divided_save)
        # input_dict = next(train_iter)
        input_dict = next(train_iter)


        input_dict = shape_input(input_dict, args.device, args.precision)

        output_dict = model_engine(**input_dict)
        loss = output_dict["loss"]

        batch = input_dict["images"].size(0)
        loss_meter.update(loss.item(), batch)
        lm_loss_meter.update(output_dict["lm_loss"].item(), batch)
        seg_loss_meter.update(output_dict["seg_loss"].item(), batch)
        cls_loss_meter.update(output_dict["cls_loss"].item(), batch)
        global_step += 1

        model_engine.backward(loss)
        model_engine.step()
        scheduler.step()

        if (global_step - last_print_step >= args.print_freq) or (step + 1 == args.divided_save):
            last_print_step = global_step
            cost_time.update(time.time() - freq_start_time)
            freq_start_time = time.time()

            if dist.is_initialized():
                loss_meter.all_reduce()
                lm_loss_meter.all_reduce()
                seg_loss_meter.all_reduce()
                cls_loss_meter.all_reduce()
                cost_time.all_reduce()

            if is_main_process():
                local_step = global_step % args.cal_step
                progress.display(local_step)
                writer.add_scalar("train/loss", loss_meter.avg, global_step)
                writer.add_scalar("train/lm_loss", lm_loss_meter.avg, global_step)
                writer.add_scalar("train/seg_loss", seg_loss_meter.avg, global_step)
                writer.add_scalar("train/cls_loss", cls_loss_meter.avg, global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            loss_meter.reset()
            lm_loss_meter.reset()
            seg_loss_meter.reset()
            cls_loss_meter.reset()
            cost_time.reset()
    return model_engine, train_iter, global_step, step + 1

def _default_cls_metrics():
    return {
        "dice_sum": 0.0,
        "dice_sq_sum": 0.0,
        "iou_sum": 0.0,
        "iou_sq_sum": 0.0,
        "cnt": 0,
    }

def _default_det_metrics():
    return {"det_iou": 0.0, "cnt": 0}

@torch.no_grad()
def valid_seg(model_engine, val_loader, global_step, epoch, writer, args, processor):
    model_engine.eval()
    eps = 1e-5
    # Segmentation metrics.
    cls_metrics = defaultdict(_default_cls_metrics)
    dir_metrics = defaultdict(_default_cls_metrics)
    total_dice, total_iou, total_samples = 0.0, 0.0, 0


    valid_det = 1000
    total_det_prompt_iou = 0.0
    valid_det_prompt_samples = 0

    saved_samples = 0
    save_count_per_folder = defaultdict(int)
    max_per_folder = 50

    show_gap = 66

    for batch_idx, input_dict in enumerate(tqdm.tqdm(val_loader, desc="Validation Seg")):
        # 1) Move the full batch to the GPU.
        input_dict = shape_input(input_dict, args.device, args.precision)
        input_dict["inference"] = True

        det_prep = input_dict["det_valid_prepare_vl"]
        if det_prep is not None:
            det_prep = det_prep.to(args.device)
        input_dict["prepare_vl"] = input_dict["prepare_vl"].to(args.device)
        

        # 2) Run segmentation inference.
        seg_out = model_engine(**input_dict)
        pred_tensor = seg_out["pred_masks"]  # list of [1,H,W]
        gt_tensor   = seg_out["gt_masks"]

        # 3) Compute Dice / IoU in batch mode.
        if pred_tensor is not None:
            B, _, H, W = pred_tensor.shape

            pred_flat = (pred_tensor > 0.5).view(B, -1)
            gt_flat   = gt_tensor.view(B, -1)

            inter = (pred_flat * gt_flat).sum(dim=1)
            sum_p = pred_flat.sum(dim=1)
            sum_g = gt_flat.sum(dim=1)
            dice_b = (2 * inter + eps) / (sum_p + sum_g + eps)
            union = sum_p + sum_g - inter + eps
            iou_b  = inter / union

            total_dice   += dice_b.sum().item()
            total_iou    += iou_b.sum().item()
            total_samples += B
            
            for i in range(B):
                
                cls = input_dict["class_list"][i]
                # cls_metrics[cls]["dice"] += dice_b[i].item()
                # cls_metrics[cls]["iou"]  += iou_b[i].item()
                # cls_metrics[cls]["cnt"]  += 1
                
                cls_metrics[cls]["dice_sum"]    += dice_b[i].item()
                cls_metrics[cls]["dice_sq_sum"] += dice_b[i].item() ** 2
                cls_metrics[cls]["iou_sum"]     += iou_b[i].item()
                cls_metrics[cls]["iou_sq_sum"]  += iou_b[i].item() ** 2
                cls_metrics[cls]["cnt"]         += 1
                
                dir = input_dict["dir_list"][i]
                # dir_metrics[dir]["dice"] += dice_b[i].item()
                # dir_metrics[dir]["iou"]  += iou_b[i].item()
                # dir_metrics[dir]["cnt"]  += 1

                dir_metrics[dir]["dice_sum"]    += dice_b[i].item()
                dir_metrics[dir]["dice_sq_sum"] += dice_b[i].item() ** 2
                dir_metrics[dir]["iou_sum"]     += iou_b[i].item()
                dir_metrics[dir]["iou_sq_sum"]  += iou_b[i].item() ** 2
                dir_metrics[dir]["cnt"]         += 1
                
        if not args.test:
            # 4) Gather detection-task indices and ground-truth boxes.
            task_list   = input_dict["task_list"]
            det_indices = [i for i, t in enumerate(task_list) if DEFAULT_DET_Task_TOKEN in t]
            det_gt = []
            for i in det_indices:
                raw = input_dict["sft_format"][i][1]["content"]
                coords = raw.split('"bbox_2d": [')[-1].split(']\n}')[0]
                det_gt.append(parse_bbox(coords))

            # 5b) Detection via mask prompt.
            if det_indices and valid_det_prompt_samples < valid_det:
                point_coords = input_dict["point_coords"][det_indices].to(args.device)
                point_labels = input_dict["point_labels"][det_indices].to(args.device)
                bboxs = input_dict["bboxs"][det_indices].to(args.device)
                det_inputs = input_dict["det_valid_prepare_vl"].to(args.device)
                gen_ids, n_prompt = model_engine.module.generate_with_prompt(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    bboxs=bboxs,
                    prepare_vl_inputs=det_inputs,
                    max_new_tokens=64,
                    use_cache=True,
                )

                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(det_inputs["input_ids"], gen_ids)]
                preds_prompt = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                for gt_box, p in zip(det_gt, preds_prompt):
                    if "[" in p and "]" in p:
                        pb = parse_bbox(p.split("[")[-1].split("]")[0])
                        total_det_prompt_iou += calculate_iou(gt_box, pb)
                        valid_det_prompt_samples += 1
                        
            if is_main_process() and batch_idx % show_gap == 0 and saved_samples < 5:
                for i in range(min(pred_tensor.__len__(), 2)):
                    img = input_dict["images"][i].float().cpu()
                    mean = torch.tensor([123.675,116.28,103.53]).view(3,1,1)
                    std  = torch.tensor([58.395,57.12,57.375]).view(3,1,1)
                    img = (img * std + mean).permute(1,2,0).contiguous().numpy().astype(np.uint8)
                    if i in det_indices:
                        idx = det_indices.index(i)
                        gt = det_gt[idx]
                        cv2.rectangle(img, (int(gt[0]),int(gt[1])), (int(gt[2]),int(gt[3])), (0,255,0),2)
                        try:
                            pp = preds_prompt[idx]
                            print(f"Ref: {gt}, Pred: {pp}")
                            if "[" in pp:
                                pb = parse_bbox(pp.split("[")[-1].split("]")[0])
                                cv2.rectangle(img, (int(pb[0]),int(pb[1])), (int(pb[2]),int(pb[3])), (255,0,0),2)
                        except: pass
                    writer.add_image(f"val_img/{saved_samples}/img",torch.from_numpy(img).permute(2,0,1), global_step)
                saved_samples += 1
            
        elif args.test:

            image_path_list = input_dict["image_path_list"]
            fname = ''

            vis_save_path = os.path.join(args.log_dir, 'visualization')
            os.makedirs(vis_save_path, exist_ok=True)
            for i in range(pred_tensor.shape[0]):
                img = input_dict["images"][i].float().cpu()
                image_path = image_path_list[i]
                folder = os.path.normpath(image_path).split('/')[6]
                if save_count_per_folder[folder] >= max_per_folder:
                    continue

                mean = torch.tensor([123.675,116.28,103.53]).view(3,1,1)
                std  = torch.tensor([58.395,57.12,57.375]).view(3,1,1)
                orig = (img * std + mean).permute(1,2,0).contiguous().numpy().astype(np.uint8)

                gt_mask = (input_dict["image_labels"][i].float().cpu().numpy()>0.5).astype(np.uint8)*255
                overlay_gt   = orig.copy(); overlay_gt[...,1] = np.where(gt_mask>0,255,orig[...,1])
                pred_mask = (pred_tensor[i,0].float().cpu().numpy()>0.5).astype(np.uint8)*255
                overlay_pred = orig.copy(); overlay_pred[...,2] = np.where(pred_mask>0,255,orig[...,2])

                orig_resized      = cv2.resize(orig,      (224,224), interpolation=cv2.INTER_LINEAR)
                gt_mask_resized   = cv2.resize(gt_mask,   (224,224), interpolation=cv2.INTER_NEAREST)
                pred_mask_resized = cv2.resize(pred_mask, (224,224), interpolation=cv2.INTER_NEAREST)

                # overlay
                overlay_gt   = orig_resized.copy()
                overlay_gt[gt_mask_resized > 0, 1] = 255   # Mark GT on the green channel.
                overlay_pred = orig_resized.copy()
                overlay_pred[pred_mask_resized > 0, 2] = 255  # Mark predictions on the red channel.

                combined = np.concatenate([orig_resized, overlay_gt, overlay_pred], axis=1)
                                    
                caption_q = input_dict["sft_format"][i][0]["content"].strip()
                caption_a = input_dict["sft_format"][i][1]["content"].strip()
                caption = caption_q+caption_a

                img_label_flat = input_dict["image_labels"][i].view(-1).float()
                mask_flat      = (pred_tensor[i,0] > 0.5).view(-1).float()
                intersection   = (img_label_flat * mask_flat).sum().item()
                sum_g          = img_label_flat.sum().item()
                sum_p          = mask_flat.sum().item()
                dice = (2 * intersection + eps) / (sum_p + sum_g + eps) 
                dice = round(dice, 4)
                image_path = image_path.split('/')[6:]
                image_path = '@'.join(image_path)
                caption = caption.split('>')[-1].strip().replace('\n', '')
                # fname = str(saved_samples) + '_'
                fname = image_path + '_'  # waiting to debug
                fname += str(dice) + '_' +caption + '.png'
                fpath = os.path.join(vis_save_path, fname)
                combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fpath, combined)

                save_count_per_folder[folder] += 1
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            

    # Aggregate results across distributed processes.
    m = torch.tensor([total_dice, total_iou, total_samples], device=args.device)
    d = torch.tensor([total_det_prompt_iou, valid_det_prompt_samples], device=args.device)
    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(m)
        dist.all_reduce(d)
    
    if dist.is_initialized():
        gathered_cls_metrics = [None for _ in range(dist.get_world_size())]
        gathered_dir_metrics = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_cls_metrics, cls_metrics)
        dist.all_gather_object(gathered_dir_metrics, dir_metrics)
    else:
        gathered_cls_metrics = [cls_metrics]
        gathered_dir_metrics = [dir_metrics]

    if is_main_process():
        merged_cls_metrics = defaultdict(_default_cls_metrics)
        for cm in gathered_cls_metrics:
            for cls, metrics in cm.items():
                # merged_cls_metrics[cls]["dice"]     += metrics["dice"]
                # merged_cls_metrics[cls]["iou"]      += metrics["iou"]
                # merged_cls_metrics[cls]["cnt"]      += metrics["cnt"]
        
                merged_cls_metrics[cls]["dice_sum"]    += metrics["dice_sum"]
                merged_cls_metrics[cls]["dice_sq_sum"] += metrics["dice_sq_sum"]
                merged_cls_metrics[cls]["iou_sum"]     += metrics["iou_sum"]
                merged_cls_metrics[cls]["iou_sq_sum"]  += metrics["iou_sq_sum"]
                merged_cls_metrics[cls]["cnt"]         += metrics["cnt"]

        merged_dir_metrics = defaultdict(_default_cls_metrics)
        for dm in gathered_dir_metrics:
            for dir_name, metrics in dm.items():
                # merged_dir_metrics[dir_name]["dice"]    += metrics["dice"]
                # merged_dir_metrics[dir_name]["iou"]     += metrics["iou"]
                # merged_dir_metrics[dir_name]["cnt"]     += metrics["cnt"]
                
                merged_dir_metrics[dir_name]["dice_sum"]    += metrics["dice_sum"]
                merged_dir_metrics[dir_name]["dice_sq_sum"] += metrics["dice_sq_sum"]
                merged_dir_metrics[dir_name]["iou_sum"]     += metrics["iou_sum"]
                merged_dir_metrics[dir_name]["iou_sq_sum"]  += metrics["iou_sq_sum"]
                merged_dir_metrics[dir_name]["cnt"]         += metrics["cnt"]
        
        if args.test:
            cls_metrics_path = os.path.join(args.log_dir, 'cls_metrics.json')
            dir_metrics_path = os.path.join(args.log_dir, 'dir_metrics.json')
            
            def summarize(merged):
                out = {}
                for name, v in merged.items():
                    cnt = v["cnt"]
                    if cnt > 0:
                        dice_mean = v["dice_sum"] / cnt
                        dice_var  = max(v["dice_sq_sum"] / cnt - dice_mean * dice_mean, 0.0)
                        dice_std  = dice_var ** 0.5
                        iou_mean  = v["iou_sum"] / cnt
                        iou_var   = max(v["iou_sq_sum"] / cnt - iou_mean * iou_mean, 0.0)
                        iou_std   = iou_var ** 0.5
                    else:
                        dice_mean = dice_std = iou_mean = iou_std = 0.0
                    out[name] = {
                        "dice_mean": dice_mean,
                        "dice_std":  dice_std,
                        "iou_mean":  iou_mean,
                        "iou_std":   iou_std,
                        "cnt":       cnt
                    }
                return out

            # with open(cls_metrics_path, 'w') as f:
            #     json.dump({cls: dict(v) for cls, v in merged_cls_metrics.items()}, f, indent=4)
            # with open(dir_metrics_path, 'w') as f:
            #     json.dump({d: dict(v) for d, v in merged_dir_metrics.items()}, f, indent=4)
            with open(cls_metrics_path, 'w') as f:
                json.dump(summarize(merged_cls_metrics), f, indent=4)
            with open(dir_metrics_path, 'w') as f:
                json.dump(summarize(merged_dir_metrics), f, indent=4)
            print(f"Saved cls_metrics to {cls_metrics_path}")
    dist.barrier()
    avg_dice = m[0].item() / m[2].item() if m[2].item() > 0 else 0.0
    avg_iou  = m[1].item() / m[2].item() if m[2].item() > 0 else 0.0
    avg_det_iou  = d[0].item() / d[1].item() if d[1].item() > 0 else 0.0
    if is_main_process() and args.test is False:
        writer.add_scalar("val_seg/dice", avg_dice, global_step)
        writer.add_scalar("val_seg/iou",  avg_iou,  global_step)
        writer.add_scalar("val_seg/det_iou", avg_det_iou, global_step)

    return avg_dice, avg_iou

@torch.no_grad()
def valid_vqa(model_engine, val_loader, global_step, epoch, writer, args, processor):
    model_engine.eval()
    gap=5
    print_current = 0
    print_max = 5
    all_preds, all_refs, all_cls_logits, all_cls_labels = [], [], [], []
    # accuracy = 0.0
    # F1_score = 0.0
    acc_list = []
    f1_list = []
    recall_list = []
    bleu1_list = []
    rouge1_sample_list = []
    precision_list = []


    def normalize_answer(s):
        """Simple text normalization: lowercase, strip punctuation, and collapse spaces."""
        s = s.lower()
        s = re.sub(f"[{string.punctuation}]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    
    def compute_vqa_accuracy(preds, refs):
        """
        preds: List[str] - model predictions
        refs: List[str] - reference answers
        """
        scores = []
        for pred, ref in zip(preds, refs):
            pred_norm = normalize_answer(pred)
            refs_norm = normalize_answer(ref)
            matched = (pred_norm == refs_norm)
            scores.append(1.0 if matched else 0.0)
        return sum(scores) / len(scores) if scores else 0.0, scores

        
    def compute_vqa_f1(preds, refs):
        f1_scores = []
        for pred, ref in zip(preds, refs):
            p_tokens = normalize_answer(pred).split()
            r_tokens = normalize_answer(ref).split()
            p_cnt = Counter(p_tokens)
            r_cnt = Counter(r_tokens)
            common = sum((p_cnt & r_cnt).values())
            prec = common / len(p_tokens) if p_tokens else 0.0
            rec = common / len(r_tokens) if r_tokens else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            f1_scores.append(f1)
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0, f1_scores

    def compute_vqa_recall(preds, refs):
        recall_scores = []
        for pred, ref in zip(preds, refs):
            p_tokens = normalize_answer(pred).split()
            r_tokens = normalize_answer(ref).split()
            p_cnt = Counter(p_tokens)
            r_cnt = Counter(r_tokens)
            common = sum((p_cnt & r_cnt).values())
            rec = common / len(r_tokens) if r_tokens else 0.0
            recall_scores.append(rec)
        return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0, recall_scores
    
    def compute_vqa_precision(preds, refs):
        precision_scores = []
        for pred, ref in zip(preds, refs):
            p_tokens = normalize_answer(pred).split()
            r_tokens = normalize_answer(ref).split()
            p_cnt = Counter(p_tokens)
            r_cnt = Counter(r_tokens)
            common = sum((p_cnt & r_cnt).values())
            prec = common / len(p_tokens) if p_tokens else 0.0
            precision_scores.append(prec)
        return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0, precision_scores

    def multilabel_metrics(preds: List[List[str]], refs: List[List[str]]):
        precisions, recalls, f1s, accs = [], [], [], []
        for p, r in zip(preds, refs):
            p_norm = [normalize_answer(tok) for tok in p]
            r_norm = [normalize_answer(tok) for tok in r]
            p_set, r_set = set(p_norm), set(r_norm)
            tp = len(p_set & r_set)
            prec = tp / len(p_set) if p_set else 0.0
            rec  = tp / len(r_set) if r_set else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            acc  = 1.0 if p_set == r_set else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            accs.append(acc)

        return accs, f1s, recalls, precisions

    def compute_bleu1(all_preds_split: List[str], all_refs_split: List[str]):
        """
        Compute BLEU-1 (unigram precision).
        Returns: (average score, per-sample score list)
        """
        scores = []
        for pred, ref in zip(all_preds_split, all_refs_split):
            pred_tokens = normalize_answer(pred).split()
            ref_tokens = normalize_answer(ref).split()
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)

            # Number of overlapping tokens.
            overlap = sum((pred_counter & ref_counter).values())
            precision = overlap / len(pred_tokens) if pred_tokens else 0.0
            scores.append(precision)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, scores

    def compute_rouge1(all_preds_split: List[str], all_refs_split: List[str]):
        """
        Compute ROUGE-1 F1.
        Returns: (average score, per-sample score list)
        """
        scores = []
        for pred, ref in zip(all_preds_split, all_refs_split):
            pred_tokens = normalize_answer(pred).split()
            ref_tokens = normalize_answer(ref).split()
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)

            overlap = sum((pred_counter & ref_counter).values())
            precision = overlap / len(pred_tokens) if pred_tokens else 0.0
            recall = overlap / len(ref_tokens) if ref_tokens else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            scores.append(f1)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, scores
    
    def compute_rougeL(all_preds_split: List[str], all_refs_split: List[str]):
        """
        Compute ROUGE-L F1 based on the longest common subsequence (LCS).
        Returns: (average score, per-sample score list)
        """
        def lcs_len(a, b):
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if a[i] == b[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
            return dp[m][n]

        scores = []
        for pred, ref in zip(all_preds_split, all_refs_split):
            p_tokens = normalize_answer(pred).split()
            r_tokens = normalize_answer(ref).split()
            l = lcs_len(p_tokens, r_tokens)
            prec = l / len(p_tokens) if p_tokens else 0.0
            rec  = l / len(r_tokens) if r_tokens else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            scores.append(f1)
        avg = sum(scores) / len(scores) if scores else 0.0
        return avg, scores
    
    def roc_without_bad(y_true, y_scores):
        # if in validation, some classes may have all positive or all negative samples, which makes roc_auc_score undefined
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        n, c = y_true.shape
        pos_sum = y_true.sum(axis=0)
        good_indices = np.where((pos_sum != 0) & (pos_sum != n))[0]
        if len(good_indices) == 0:
            return 0.0
        return roc_auc_score(y_true[:, good_indices], y_scores[:, good_indices], average="macro")
    
    
    for batch_idx, batch in enumerate(tqdm.tqdm(val_loader, desc="Validation VQA")):
        input_dict = shape_input(batch, args.device, args.precision)
        input_dict["inference"] = True
        prepare_vl_inputs = input_dict["prepare_vl"].to(args.device)

        # Assume the batch contains images, point_coords, point_labels, bboxs, and prepare_vl.
        if "SLAKE" in args.vqa_data or "PMC_VQA" in args.vqa_data or "vqa-rad" in args.vqa_data or "path-vqa" in args.vqa_data:
            max_new_tokens = 64
        else:
            max_new_tokens = 128

        if args.finetune and args.cls_num>0:
            input_dict["prepare_vl"] = input_dict["prepare_vl"].to(args.device)
            model_out = model_engine(**input_dict)
            logit = model_out["cls_logits"] # [B, num_classes]
            all_cls_logits.extend(logit.cpu().tolist())
            all_cls_labels.extend(input_dict["class_tail_label_list"].cpu().tolist())

        gen_ids, n_prompt  = model_engine.module.generate_with_prompt(
            point_coords=batch.get("point_coords"),
            point_labels=batch.get("point_labels"),
            bboxs=batch.get("bboxs"),
            prepare_vl_inputs=prepare_vl_inputs,
            max_new_tokens=max_new_tokens,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(prepare_vl_inputs["input_ids"], gen_ids)]
        preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        all_preds.extend([p.strip() for p in preds])

        if is_main_process() and batch_idx % gap == 0 and print_current < print_max:
            question = batch['sft_format'][0][0]['content'].replace("\n", "")
            ref = batch["sft_format"][0][-1]["content"].strip()
            print(f"Ques: {question}\nRef: {ref}\nPred: {preds[0].strip()}\n\n")
            print_current += 1
        
        # 5) Collect reference answers.
        for conv in batch["sft_format"]:
            all_refs.append(conv[-1]["content"].strip())

    # 7) Distributed aggregation.
    all_refs  = gather_all(all_refs, args.device)
    all_preds = gather_all(all_preds, args.device)


    all_refs_split = all_refs
    all_preds_split = all_preds

    if args.finetune and args.cls_num>0:
        all_cls_logits = gather_all(all_cls_logits, args.device)
        all_cls_labels = gather_all(all_cls_labels, args.device)
        all_cls_logits = np.array(all_cls_logits)
        all_cls_labels = np.array(all_cls_labels)
        probs_cls = torch.sigmoid(torch.tensor(all_cls_logits)).numpy()
        preds_cls = (probs_cls > 0.5).astype(int)
        
        auc_cls = roc_without_bad(all_cls_labels, preds_cls)
        acc_cls = accuracy_score(all_cls_labels, preds_cls)
        precision_cls = precision_score(all_cls_labels, preds_cls, average="macro", zero_division=0)
        recall_cls = recall_score(all_cls_labels, preds_cls, average="macro", zero_division=0)
        f1_cls = f1_score(all_cls_labels, preds_cls, average="macro", zero_division=0)

    # save refs & preds
    if args.test:
        save_path = os.path.join(args.log_dir + '/results/')
        os.makedirs(save_path, exist_ok=True)
        results = []
        if args.finetune and args.cls_num>0:
            for r, p, probs, cls_label in zip(all_refs, all_preds, probs_cls, all_cls_labels):
                results.append({"ref": r, "pred": p, "probs": probs.tolist(), "cls_label": cls_label.tolist()})
        else:
            for r, p in zip(all_refs, all_preds):
                results.append({"ref": r, "pred": p})

        with open(os.path.join(save_path + args.vqa_data + "_" + args.openclosetype + "_" + args.modality + "_pred.json"), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved VQA results to {save_path}")
        
    if args.test and args.vqa_data in ["cxr-lt", "isic2018", "isic2019"]:
        all_refs_multilabel = [[tok.strip() for tok in r.split(',')] for r in all_refs]
        all_preds_multilabel = [[tok.strip() for tok in p.split(',')] for p in all_preds]
        acc_list, f1_list, recall_list, precision_list = multilabel_metrics(all_preds_multilabel, all_refs_multilabel)
    else:
        if args.openclosetype == "close":
            all_preds_split = [p.split(":")[0] for p in all_preds]
            all_preds_split = [p.split(",")[0] for p in all_preds_split]
            if 'PMC_VQA' in args.vqa_data:
                all_refs_split  = [r.split(":")[0].split(",")[0] for r in all_refs]
        
    # 8) Compute metrics.
    accuracy, acc_list = compute_vqa_accuracy(all_preds_split, all_refs_split)
    F1_score, f1_list = compute_vqa_f1(all_preds_split, all_refs_split)
    recall, recall_list = compute_vqa_recall(all_preds_split, all_refs_split)
    precision, precision_list = compute_vqa_precision(all_preds_split, all_refs_split)
    bleu1, bleu1_list = compute_bleu1(all_preds_split, all_refs_split)
    rouge1, rouge1_sample_list = compute_rouge1(all_preds_split, all_refs_split)
    rougeL_mean, rougeL_list = compute_rougeL(all_preds_split, all_refs_split)  

        
    def five_number_summary(a):
        """Return min, Q1, median, Q3, max for box plots."""
        arr = np.array(a, dtype=np.float32)
        if arr.size == 0:
            return {"min": 0.0, "q1": 0.0, "median": 0.0, "q3": 0.0, "max": 0.0}
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        return {
            "min": float(arr.min()),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "max": float(arr.max()),
        }

    def m_s(a):
        arr = np.array(a, dtype=np.float32)
        if arr.size == 0:
            return 0.0, 0.0
        return float(arr.mean()), float(arr.std(ddof=0))

    accuracy_mean, accuracy_std = m_s(acc_list)
    f1_mean, f1_std             = m_s(f1_list)
    recall_mean, recall_std     = m_s(recall_list)
    precision_mean, precision_std = m_s(precision_list)
    bleu1_mean, bleu1_std       = m_s(bleu1_list)
    rouge1_mean, rouge1_std     = m_s(rouge1_sample_list)
    rougeL_mean, rougeL_std     = m_s(rougeL_list)   # Std is already returned here.

    boxplot_stats = {
        "accuracy": five_number_summary(acc_list),
        "f1": five_number_summary(f1_list),
        "recall": five_number_summary(recall_list),
        "precision": five_number_summary(precision_list),
        "bleu1": five_number_summary(bleu1_list),
        "rouge1": five_number_summary(rouge1_sample_list),
        "rougeL": five_number_summary(rougeL_list),
    }

    extra_stats = {
        "bleu1_mean": bleu1_mean,
        "bleu1_std": bleu1_std,
        "rouge1_mean": rouge1_mean,
        "rouge1_std": rouge1_std,
        "rougeL_mean": rougeL_mean,
        "rougeL_std": rougeL_std,
        "acc_mean": accuracy_mean,
        "acc_std": accuracy_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "num_samples": len(acc_list),
        "boxplot": boxplot_stats,
    }

    if args.finetune and args.cls_num>0:
        extra_stats.update({
            "auc_cls": auc_cls,
            "acc_cls": acc_cls,
            "f1_cls": f1_cls,
            "recall_cls": recall_cls,
            "precision_cls": precision_cls,
        })
    

    # 9) Log and print.
    if is_main_process() and args.test is False:
        writer.add_scalar("val_vqa/bleu", bleu1_mean, global_step)
        writer.add_scalar("val_vqa/rouge1", rouge1_mean, global_step)
        if args.finetune and args.cls_num > 0:
            writer.add_scalar("val_vqa/auc_cls", auc_cls, global_step)
            writer.add_scalar("val_vqa/precision_cls", precision_cls, global_step)
            writer.add_scalar("val_vqa/recall_cls", recall_cls, global_step)
            writer.add_scalar("val_vqa/f1_cls", f1_cls, global_step)
            print(f"Step {global_step}: VQA BLEU {bleu1_mean:.4f}, ROUGE1 {rouge1_mean:.4f}, AUC {auc_cls:.4f}, F1 {f1_cls:.4f}, Recall {recall_cls:.4f}")
        else:
            print(f"Step {global_step}: VQA BLEU {bleu1_mean:.4f}, ROUGE1 {rouge1_mean:.4f}, Acc {accuracy_mean:.4f}, F1 {f1_mean:.4f}, Recall {recall_mean:.4f}, Precision {precision_mean:.4f}")

    return bleu1_mean, rouge1_mean, accuracy, F1_score, recall, extra_stats

@torch.no_grad()
def valid_det(model_engine, val_loader, global_step, epoch, writer, args, processor):
    print("Running Detection validation in test mode...")
    pure_qwen = False
    if pure_qwen:
        print("Using pure Qwen2VL model, refer_seg data is set to qwen mode!!!.")
    
    model_engine.eval()
    cls_det_metrics = defaultdict(_default_det_metrics)
    dir_det_metrics = defaultdict(_default_det_metrics)

    saved_samples = 0

    for batch_idx, input_dict in enumerate(tqdm.tqdm(val_loader, desc="Validation Det")):
        # 1) Move the full batch to the GPU.
        input_dict = shape_input(input_dict, args.device, args.precision)
        input_dict["inference"] = True

        det_prep = input_dict["det_valid_prepare_vl"]
        if det_prep is not None:
            det_prep = det_prep.to(args.device)
        input_dict["prepare_vl"] = input_dict["prepare_vl"].to(args.device)

            
        if args.test:
            task_list = input_dict["task_list"]
            det_indices = [i for i, t in enumerate(task_list) if DEFAULT_DET_Task_TOKEN in t]
            det_gt = []
            for i in det_indices:
                raw = input_dict["sft_format"][i][1]["content"]
                coords = raw.split('"bbox_2d": [')[-1].split(']\n}')[0]
                det_gt.append(parse_bbox(coords))

            # 2) Generate predicted boxes.
            if det_indices:
                point_coords = input_dict["point_coords"][det_indices].to(args.device)
                point_labels = input_dict["point_labels"][det_indices].to(args.device)
                bboxs        = input_dict["bboxs"][det_indices].to(args.device)
                det_inputs   = input_dict["det_valid_prepare_vl"].to(args.device)

                gen_ids, _ = model_engine.module.generate_with_prompt(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    bboxs=bboxs,
                    prepare_vl_inputs=det_inputs,
                    max_new_tokens=64,
                    use_cache=True,
                )
                preds_trim = [out_ids[len(in_ids):] 
                              for in_ids, out_ids in zip(det_inputs["input_ids"], gen_ids)]
                preds_box  = processor.batch_decode(preds_trim, skip_special_tokens=True)

                # 3) Accumulate IoU and visualize the first 200 samples.
                for idx, gt_box in enumerate(det_gt):
                    # get corresponding prediction (if exists)
                    p = preds_box[idx] if idx < len(preds_box) else ""
                    if "[" in p and "]" in p:
                        try:
                            pb = parse_bbox(p.split("[")[-1].split("]")[0])
                            iou = calculate_iou(gt_box, pb)
                        except:
                            iou = 0.0
                    else:
                        iou = 0.0

                    # Accumulate metrics by class and directory.
                    cls = input_dict["class_list"][det_indices[idx]]
                    dir_name = input_dict["dir_list"][det_indices[idx]]
                    cls_det_metrics[cls]["det_iou"] += iou
                    cls_det_metrics[cls]["cnt"]  += 1
                    dir_det_metrics[dir_name]["det_iou"] += iou
                    dir_det_metrics[dir_name]["cnt"]  += 1

                    # Visualization.
                    if pure_qwen:
                        if is_main_process() and saved_samples < 200 and batch_idx % 200 == 0:
                            img = input_dict["images"][idx].float().cpu()
                            mean = torch.tensor([123.675,116.28,103.53]).view(3,1,1)
                            std  = torch.tensor([58.395,57.12,57.375]).view(3,1,1)
                            orig = (img * std + mean).permute(1,2,0).contiguous().numpy().astype(np.uint8)
                            
                            x1 = int(gt_box[0] / 1000 * 224)
                            y1 = int(gt_box[1] / 1000 * 224)
                            x2 = int(gt_box[2] / 1000 * 224)
                            y2 = int(gt_box[3] / 1000 * 224)
                            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            if iou > 0:
                                x1 = int(pb[0] / 1000 * 224)
                                y1 = int(pb[1] / 1000 * 224)
                                x2 = int(pb[2] / 1000 * 224)
                                y2 = int(pb[3] / 1000 * 224)
                                cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            writer.add_image(f"det_vis/{saved_samples}", torch.from_numpy(orig).permute(2,0,1), global_step)
                            saved_samples += 1
                    else:
                        if is_main_process() and saved_samples < 200 and batch_idx % 200 == 0:
                            img = input_dict["images"][idx].float().cpu()
                            mean = torch.tensor([123.675,116.28,103.53]).view(3,1,1)
                            std  = torch.tensor([58.395,57.12,57.375]).view(3,1,1)
                            orig = (img * std + mean).permute(1,2,0).contiguous().numpy().astype(np.uint8)
                            cv2.rectangle(orig, (int(gt_box[0]),int(gt_box[1])),
                                          (int(gt_box[2]),int(gt_box[3])), (0,255,0),2)
                            if iou > 0:
                                cv2.rectangle(orig, 
                                              (int(pb[0]),int(pb[1])),
                                              (int(pb[2]),int(pb[3])), 
                                              (255,0,0),2)
                            writer.add_image(f"det_vis/{saved_samples}", torch.from_numpy(orig).permute(2,0,1), global_step)
                            saved_samples += 1
                        

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Distributed aggregation.
    if dist.is_initialized():
        dist.barrier()
        gathered_cls = [None] * dist.get_world_size()
        gathered_dir = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_cls, cls_det_metrics)
        dist.all_gather_object(gathered_dir, dir_det_metrics)
    else:
        gathered_cls = [cls_det_metrics]
        gathered_dir = [dir_det_metrics]
    
    
    # Merge results on all processes.
    merged_cls = defaultdict(lambda: {"det_iou": 0.0, "cnt": 0})
    merged_dir = defaultdict(lambda: {"det_iou": 0.0, "cnt": 0})
    for cm in gathered_cls:
        for k, v in cm.items():
            merged_cls[k]["det_iou"] += v["det_iou"]
            merged_cls[k]["cnt"]  += v["cnt"]
    for dm in gathered_dir:
        for k, v in dm.items():
            merged_dir[k]["det_iou"] += v["det_iou"]
            merged_dir[k]["cnt"]  += v["cnt"]

    # Only the main process writes files.
    if is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        with open(os.path.join(args.log_dir, "cls_det_metrics.json"), "w") as f:
            json.dump(merged_cls, f, indent=4)
        with open(os.path.join(args.log_dir, "dir_det_metrics.json"), "w") as f:
            json.dump(merged_dir, f, indent=4)
        print("Saved metrics to", args.log_dir)

    dist.barrier()

    # Global average IoU.
    total_iou = sum(v["det_iou"] for v in merged_cls.values())
    total_cnt = sum(v["cnt"] for v in merged_cls.values())
    avg_det_iou = total_iou / total_cnt if total_cnt > 0 else 0.0

    return avg_det_iou


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    GLOBAL_SEED = args.seed
    if args.test:
        GLOBAL_SEED = 42  
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    set_seed(GLOBAL_SEED)

    print(args.local_rank)
    main(sys.argv[1:])
