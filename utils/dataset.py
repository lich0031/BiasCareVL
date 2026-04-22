import math
import json
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

from typing import Dict
from .refer_seg_dataset import ReferSegDataset
from .vqa_dataset_trainsplit import VQADataset


import hashlib
from itertools import combinations

import random
from PIL import Image
# from model.deepseek_vl2.models import DeepseekVLV2Processor
from model.Qwen2_vl25.qwenfinetune.qwenvl.data.data_qwen import DataCollatorForSupervisedDataset
from model.Qwen2_vl25.qwenfinetune.qwenvl.data.data_qwen import preprocess_qwen_2_visual
from model.Qwen2_vl25.qwenfinetune.qwenvl.data.rope2d import get_rope_index_25
from qwen_vl_utils import process_vision_info
from .utils import DEFAULT_IMAGE_TOKEN, DEFAULT_DET_Task_TOKEN, DEFAULT_SEG_Task_TOKEN, GLOBAL_SEED

torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)


def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def collate_fn(batch=None, processor=None, valid=False, device=None, promptype='tbp', model_type='Qwen'):
    # if valid==False:
    #     MAX_TOTAL_IMAGES = len(batch)
    #     accum = 0
    #     new_batch = []
    #     for item in batch:
    #         num_images = item[0].shape[0]
    #         if accum + num_images > MAX_TOTAL_IMAGES:
    #             continue
    #         new_batch.append(item)
    #         accum += num_images
    #         if accum == MAX_TOTAL_IMAGES:
    #             break
    #     # if len(new_batch) < len(batch):
    #     #     print(f"[Truncating] Reduced batch from {len(batch)} to {len(new_batch)} based on image count.")
    #     batch = new_batch
        
    images_list = torch.cat([item[0] for item in batch], dim=0)
    images_label_list = torch.cat([item[1] for item in batch], dim=0)
    bboxs_list = torch.cat([item[2] for item in batch], dim=0)
    point_coords_list = torch.cat([item[3] for item in batch], dim=0)
    point_labels_list = torch.cat([item[4] for item in batch], dim=0)
    task_list = [path for item in batch for path in item[7]]
    class_list = [cls for item in batch for cls in item[9]]
    dir_list = [d for item in batch for d in item[10]] # Assuming item[10] is directory list# Assuming item[11] is class_cxr_lt_label list
    class_tail_label_list = torch.stack([cls for item in batch for cls in item[11]], dim=0)

    sft_format_list = []
    image_path_list = [path for item in batch for path in item[6]] # Assuming item[6] is image_path list
    all_prepare_inputs_dicts = [proc_dict for item in batch for proc_dict in item[5]] # List of dictionaries from dataset __getitem__
    try:
        sft_format_list = [d.get("sft_format", None) for d in all_prepare_inputs_dicts] # Use .get for safety
    except:
        sft_format_list = [d["sft_format"] for d in all_prepare_inputs_dicts]
    
    promptype = [p for item in batch for p in (item[8] if isinstance(item[8], (list, tuple)) else [item[8]])]

    prepare_vl = None
    prepare_vl_det = None
    
    # type = "deepseek"
    if "Qwen" in model_type:
        if valid:
            # sampled_prompt_types = promptype
            vqa_indices = [i for i, task in enumerate(task_list) if "vqa" in task]
            seg_indices = [i for i, task in enumerate(task_list) if DEFAULT_SEG_Task_TOKEN in task]
            det_indices = [i for i, task in enumerate(task_list) if DEFAULT_DET_Task_TOKEN in task]
            # Handle VQA tasks.
            if vqa_indices:
                vqa_prepare_inputs = [all_prepare_inputs_dicts[i] for i in vqa_indices]
                if any("messages" in d for d in vqa_prepare_inputs):
                    all_messages = [d["messages"] for d in vqa_prepare_inputs if "messages" in d]
                    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in all_messages]
                    image_inputs, video_inputs = process_vision_info(all_messages)
                    prepare_vl = processor(
                        text=texts,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
            
            # Handle segmentation tasks, including pure segmentation and detection+segmentation.
            elif seg_indices:
                seg_prepare_inputs = [all_prepare_inputs_dicts[i] for i in seg_indices]
                if seg_prepare_inputs and any("messages" in d for d in seg_prepare_inputs):
                    # Handle all segmentation tasks, including pure segmentation and detection+segmentation.
                    seg_messages_list = [d["messages"] for d in seg_prepare_inputs if "messages" in d]
                    seg_texts = [
                        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                        for msg in seg_messages_list
                    ]
                    seg_image_inputs, seg_video_inputs = process_vision_info(seg_messages_list)
                    prepare_vl = processor(
                        text=seg_texts,
                        images=seg_image_inputs,
                        videos=seg_video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    if det_indices:
                        det_inputs = [all_prepare_inputs_dicts[i] for i in det_indices]
                        # Only keep messages.
                        det_msgs = [d["messages"] for d in det_inputs if "messages" in d]
                        # Text prompts.
                        det_texts = [
                            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                            for m in det_msgs
                        ]
                        # vision inputs
                        det_image_inputs, _ = process_vision_info(det_msgs)
                        # Process the batch in a single call.
                        prepare_vl_det = processor(
                            text=det_texts,           # List length == len(det_indices)
                            images=det_image_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
        else: 
            qwen_collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
            prepare_vl = qwen_collator(all_prepare_inputs_dicts)
            
        result_dict =  {
            "images": images_list,
            "image_labels": images_label_list,
            "bboxs": bboxs_list,
            "point_coords": point_coords_list,
            "point_labels": point_labels_list,
            "prepare_vl": prepare_vl,
            "sft_format": sft_format_list,
            "task_list": task_list,
            "det_valid_prepare_vl": prepare_vl_det,
            "promptype": promptype,
            "valid": valid,
            "class_list": class_list,
            "dir_list": dir_list,
            "image_path_list": image_path_list,
            "class_tail_label_list": class_tail_label_list
            }
    else:
        prepare_vl = processor.batchify(all_prepare_inputs_dicts)
        result_dict =  {
            "images": images_list,
            "image_labels": images_label_list,
            "bboxs": bboxs_list,
            "point_coords": point_coords_list,
            "point_labels": point_labels_list,
            "prepare_vl": prepare_vl,
            "sft_format": sft_format_list,
            "task_list": task_list,
            "det_valid_prepare_vl": prepare_vl,
            "promptype": promptype,
            "valid": valid,
            "class_list": class_list,
            "dir_list": dir_list,
            "image_path_list": image_path_list,
            "class_tail_label_list": class_tail_label_list
            }
    return result_dict

class HybridDataset(Dataset):
    def __init__(self,
                 base_image_dir,
                 chat_processor,
                 image_size=224,
                 dataset="refer_seg||vqa",
                 sample_rates=[1, 1],
                 refer_seg_data="IMed361M",
                 vqa_data="PubMedVision",
                 conv=None,
                 promptype='tbp',
                 cls_num=0,
                 cls_token=None,
                 torch_dtype=torch.float32):
        # Split sub-datasets.
        self.dataset_names = dataset.split("||")
        self.sample_rates = sample_rates
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.chat_processor = chat_processor
        self.conv = conv
        # Instantiate sub-datasets and record their lengths.
        self.all_datasets = []
        self.dataset_lengths = []
        for name in self.dataset_names:
            if name == "refer_seg":
                ds = ReferSegDataset(
                    base_image_dir=base_image_dir,
                    processor=chat_processor,
                    conv=conv,
                    image_size=image_size,
                    refer_seg_data=refer_seg_data,
                    torch_dtype=torch_dtype,
                    promptype=promptype,
                    cls_num=cls_num,
                )
            elif name == "vqa":
                ds = VQADataset(
                    base_image_dir=base_image_dir,
                    processor=chat_processor,
                    conv=conv,
                    image_size=image_size,
                    vqa_data=vqa_data,
                    torch_dtype=torch_dtype,
                    cls_num=cls_num,
                    cls_token=cls_token,
                )
            else:
                raise ValueError(f"Unsupported sub-dataset {name}")
            self.all_datasets.append(ds)
            self.dataset_lengths.append(len(ds))
        # Build global index ranges for each sub-dataset.
        self.group_indices = {}
        offset = 0
        for i, L in enumerate(self.dataset_lengths):
            self.group_indices[i] = list(range(offset, offset + L))
            offset += L
        # Weight tensor for each group.
        self.group_weights = {
            i: torch.full((len(idxs),), fill_value=float(self.sample_rates[i]), dtype=torch.double)
            for i, idxs in self.group_indices.items()
        }

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        # Locate the corresponding sub-dataset.
        cum = 0
        if len(self.dataset_lengths) == 1:
            return self.all_datasets[0][idx]
        else:
            for i, L in enumerate(self.dataset_lengths):
                if idx < cum + L:
                    return self.all_datasets[i][idx - cum]
                cum += L
        raise IndexError


class DistributedGroupSampler(Sampler):
    def __init__(self,
                 group_indices: Dict[int, list],
                 group_weights: Dict[int, torch.Tensor],
                 sample_rates: list,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        # Distributed setup.
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.group_indices = group_indices
        self.group_weights = group_weights
        self.sample_rates = sample_rates
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Samples per group = floor(total * rate / sum_rate).
        total = sum(len(v) for v in group_indices.values())
        sum_rate = sum(sample_rates)
        self.group_sizes = {
            g: math.floor(total * sample_rates[g] / sum_rate)
            for g in group_indices
        }
        # pad to divisible by num_replicas
        self.padded_group = {
            g: math.ceil(sz / num_replicas) * num_replicas
            for g, sz in self.group_sizes.items()
        }
        # Number of samples on each rank.
        self.num_samples = sum(sz // num_replicas for sz in self.padded_group.values())

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_idxs = []
        for g_id, idxs in self.group_indices.items():
            w = self.group_weights[g_id]
            sz = self.group_sizes[g_id]
            # Sample without replacement.
            sampled = torch.multinomial(w, sz, replacement=True, generator=g).tolist()
            # pad up
            pad = self.padded_group[g_id] - sz
            if pad > 0:
                extra = torch.multinomial(w, pad, replacement=True, generator=g).tolist()
                sampled += extra
            per_rank = self.padded_group[g_id] // self.num_replicas
            st = self.rank * per_rank
            ed = st + per_rank
            all_idxs += [idxs[i] for i in sampled[st:ed]]
        if self.shuffle:
            perm = torch.randperm(len(all_idxs), generator=g).tolist()
            all_idxs = [all_idxs[i] for i in perm]
        return iter(all_idxs)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        
class ValDataset_seg(torch.utils.data.Dataset):
    def __init__(self, 
                 base_image_dir, 
                 chat_processor,
                 image_size=224,
                 dataset="IMed361M",
                 conv=None, 
                 test = False,
                 model_type = "Qwen",
                 torch_dtype=torch.float32,
                 cls_num = 0,
                 ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.chat_processor = chat_processor
        self.conv=conv
        self.cls_num = cls_num
        refer_seg_data = dataset

        self.all_datasets = ReferSegDataset(base_image_dir=base_image_dir, processor=self.chat_processor, conv=self.conv,
                                    image_size=image_size, refer_seg_data=refer_seg_data, torch_dtype=torch_dtype, valid=True, test=test, model_type=model_type, cls_num = self.cls_num,)
        
    def __len__(self):
        return len(self.all_datasets)
    def __getitem__(self, idx):
        data = self.all_datasets
        return data[idx]
    
class ValDataset_vqa(torch.utils.data.Dataset):
    def __init__(self, 
                 base_image_dir, 
                 chat_processor,
                 image_size=224,
                 dataset="PubMedVision",
                 conv=None, 
                 test = False,
                 modality = '',
                 openclosetype = "close",
                 torch_dtype=torch.float32,
                 cls_num = 0,
                 cls_token = None,
                 ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.chat_processor = chat_processor
        self.conv=conv
        vqa_data = dataset
        self.conv = conv
        self.cls_num = cls_num
        self.all_datasets = VQADataset(
            base_image_dir=base_image_dir,
            processor=chat_processor,
            conv=conv,
            image_size=image_size,
            vqa_data=vqa_data,
            torch_dtype=torch_dtype,
            valid=True,
            test=test,
            modality=modality,
            openclosetype = openclosetype,
            cls_num = self.cls_num,
            cls_token = cls_token
        )
    def __len__(self):
        return len(self.all_datasets)

    def __getitem__(self, idx):
        data = self.all_datasets
        return data[idx]
