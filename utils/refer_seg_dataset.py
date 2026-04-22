import os
import ast
import json
import torch
import random
import copy
import bisect
import functools
from PIL import Image
import numpy as np
from scipy import sparse
from PIL import UnidentifiedImageError
from torchvision.transforms.functional import resize
# from typing import Sequence
from collections.abc import Sequence
import math, random
import torch.nn.functional as F

from model.deepseek_vl2.utils.io import load_pil_images
from model.IMIS.dataloaders.data_utils import get_bboxes_from_mask, get_points_from_mask_batch
from model.Qwen2_vl25.qwenfinetune.qwenvl.data.data_qwen import preprocess_qwen_2_visual
from model.Qwen2_vl25.qwenfinetune.qwenvl.data.rope2d import get_rope_index_25
from .utils import SHORT_QUESTION_LIST, DEFAULT_SEG_Task_TOKEN, DEFAULT_DET_Task_TOKEN, ANSWER_LIST_DET, ANSWER_LIST_SEG, Disease_name, Disease_QUESTION_LIST, DEFAULT_IMAGE_TOKEN, GLOBAL_SEED, DEFAULT_PROMPT_TOKEN
from .utils import GLOBAL_SEED
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

@functools.lru_cache(maxsize=None)
def load_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

class ReferSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __init__(
        self,
        base_image_dir=None,
        processor=None,
        conv=None,
        image_size: int = 224,
        refer_seg_data="IMed361M",
        torch_dtype=torch.float32,
        valid=False,
        test=False,
        chat=False,
        model_type='Qwen',
        promptype="tbp",
        cls_num = 0,
    ):  
        self.image_size = image_size
        print("image_size:" + str(self.image_size))
        self.all_num = 0
        self.torch_dtype = torch_dtype
        self.conv = conv
        self.model_type = model_type
        self.promptype = promptype
        self.test = test
        self.pure_qwen = False
        self.cls_num = cls_num
        # self.pure_qwen = True
        if self.pure_qwen:
            print("Using pure Qwen2VL model, refer_seg data is set to qwen mode!!!.")

        if not chat:
            self.short_question_list = SHORT_QUESTION_LIST
            self.vl_chat_processor = processor
            self.answer_list_seg = ANSWER_LIST_SEG
            self.answer_list_det = ANSWER_LIST_DET
            
            self.refer_seg_ds_list = refer_seg_data.split("||")
            self.valid = valid
            self.refer_seg_data = {}
            for ds in self.refer_seg_ds_list:
                if ds == "SAMed2Dv1":
                    self.refer_seg_data[ds] = None
                    self.base_path = '/mnt/disk3/jiarunliu/Documents/datasets/SA-Med2D-20M/raw/SAMed2Dv1/'
                    self.reshape_mapping = load_json('./reshape_mapping.json')
                    self.reshape_imgpath = load_json('./reshape_imgpath.json')
                    self.image_path = list(self.reshape_imgpath)
                    self.all_num += len(self.image_path)
                elif ds == 'IMed361M':
                    self.refer_seg_data[ds] = None
                    self.base_path = '/mnt/disk3/hwj/IMed361M/data/'
                    self.Imed_list = os.listdir(self.base_path)
                    self.Imed_list.remove('.cache')
                    
            self.folder_total_count = {}
            # LUNA16 for detection
            for dir_name in self.Imed_list:
                data_path = os.path.join(self.base_path, dir_name)
                json_path = os.path.join(data_path, 'dataset.json')
                if not os.path.exists(data_path) or not os.path.exists(json_path):
                    continue
                datajson = load_json(json_path)
                # print(dir_name, datajson['modality'])
                count = datajson["numTest"] if self.valid else datajson["numTraining"]
                
                if self.valid:
                    if count == 0:
                        continue
                    if not self.test:
                        count = max(1, count // 30)
                    #     # count = max(1, count // 5000)
                    # else:
                    #     count = max(1, count // 5000)
                    #     print("Only for Testing Refer_seg counts!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:", count)
                self.folder_total_count[dir_name] = count
                
            '''# For modality counting'''
            # self.folder_total_count = {}
            # for dir_name in self.Imed_list:
            #     data_path = os.path.join(self.base_path, dir_name)
            #     json_path = os.path.join(data_path, 'dataset.json')
            #     if not os.path.exists(data_path) or not os.path.exists(json_path):
            #         continue
            #     datajson = load_json(json_path)
            #     count = datajson["numTraining"]
            #     self.folder_total_count[dir_name] = count

            # for dir_name in self.Imed_list:
            #     data_path = os.path.join(self.base_path, dir_name)
            #     json_path = os.path.join(data_path, 'dataset.json')
            #     if not os.path.exists(data_path) or not os.path.exists(json_path):
            #         continue
            #     datajson = load_json(json_path)
            #     count = datajson["numTest"]
            #     self.folder_total_count[dir_name] += count
            ''''''''''''''
                
                
                
                
            self.all_num = sum(self.folder_total_count.values())
            if self.valid:
                print("Testing Refer_seg counts:", self.all_num)
            else:
                print("Training Refer_seg counts:", self.all_num)

            self.folder_keys = sorted(self.folder_total_count.keys())
            self.cum_counts = []
            total = 0
            for key in self.folder_keys:
                total += self.folder_total_count[key]
                self.cum_counts.append(total)

    def __len__(self):
        # return 400
        return self.all_num

    def convert_to_qwen2vl_format(self, bboxs, label):
        h, w = label.shape[1], label.shape[2]
        std_bboxs = []
        for bbox in bboxs[:, 0, :]:
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.tolist()
            x1, y1, x2, y2 = bbox
            x1_new = round(x1 / w * 1000)
            y1_new = round(y1 / h * 1000)
            x2_new = round(x2 / w * 1000)
            y2_new = round(y2 / h * 1000)
            
            x1_new = max(0, min(x1_new, 1000))
            y1_new = max(0, min(y1_new, 1000))
            x2_new = max(0, min(x2_new, 1000))
            y2_new = max(0, min(y2_new, 1000))
            std_bboxs.append([x1_new, y1_new, x2_new, y2_new])
        std_bboxs = torch.tensor(std_bboxs, dtype=torch.int, device=label.device)
        std_bboxs = std_bboxs.unsqueeze(1)
        
        return std_bboxs

    def box_for_deepseek(self, bboxs, label):
        mask_h, mask_w = label.shape[1], label.shape[2]
        std_bboxs = []
        for bbox in bboxs[:, 0, :]:
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.tolist()
            x_min, y_min, x_max, y_max = bbox
            y_min_std = int(y_min / mask_h * 999)
            x_min_std = int(x_min / mask_w * 999)
            y_max_std = int(y_max / mask_h * 999)
            x_max_std = int(x_max / mask_w * 999)
            std_bboxs.append([y_min_std, x_min_std, y_max_std, x_max_std])
        std_bboxs = torch.tensor(std_bboxs, dtype=torch.int, device=label.device)
        std_bboxs = std_bboxs.unsqueeze(1)
        return std_bboxs

    def rearrange_class(self, s):
        for direction in ["left", "right"]:
            if direction in s:
                # Remove any existing occurrence and prepend the direction.
                s = f"{direction} " + s.replace(direction, "").strip()
                break
        return s
                
    def full_transform(self, pil_img, image_labels=None):
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)   # Be careful here!
        img_np = np.array(pil_img)
        img_np = img_np.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_np).float()
        seg_in = (img_tensor - self.pixel_mean) / self.pixel_std
        
        # img_np = np.array(pil_img).astype(np.float32)        # [H, W, C]
        # img_np = img_np.transpose(2, 0, 1)                    # → [C, H, W]
        # img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # Add batch dim, move to GPU
        # img_tensor_resized = F.interpolate(img_tensor, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        # seg_in = (img_tensor_resized - self.pixel_mean) / self.pixel_std
        # seg_in = seg_in.squeeze(0)  

        if image_labels is not None:
            image_labels = resize(image_labels, (self.image_size, self.image_size)) 
            bboxs = get_bboxes_from_mask(image_labels, offset=0)
            point_coords, point_labels = get_points_from_mask_batch(image_labels, get_point=1, top_num=0.5)
            if "deepseek" in self.model_type:
                bboxs = self.box_for_deepseek(bboxs, image_labels)
            elif "Qwen" in self.model_type and self.pure_qwen:
                bboxs = self.convert_to_qwen2vl_format(bboxs, image_labels)
            return seg_in, pil_img, image_labels, point_coords, point_labels, bboxs
        else:
            return seg_in, pil_img
    
    def _sample_promptstype(self, prompt_type: str) -> str:
        if not isinstance(prompt_type, str) or len(prompt_type) <= 1:
            return prompt_type or ""
        chars = list(prompt_type)
        num_to_keep = random.randint(1, len(chars))
        kept = random.sample(chars, num_to_keep)
        kept.sort(key=lambda c: prompt_type.index(c))
        return "".join(kept)
    
    def _data_augmentation(self, pil_img: Image.Image) -> Image.Image:
        """
        Lightweight data augmentation used only during training:
        - random horizontal flip
        - random vertical flip with low probability
        - random small-angle rotation while keeping the original size
        - random brightness / contrast / color jitter
        - random additive Gaussian noise
        - random center-jitter crop (approximate resized crop)
        """

        w, h = pil_img.size

        # # 1. flips
        # if random.random() < 0.5:
        #     pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        # if random.random() < 0.1:
        #     pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

        # # 2. small rotation
        # if random.random() < 0.5:
        #     angle = random.uniform(-12, 12)
        #     # rotate with expand=False to keep size, fill with black
        #     pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0))

        # 3. brightness / contrast / color jitter (small ranges for medical images)
        if random.random() < 0.6:
            def _clip01(arr): return np.clip(arr, 0, 255)
            img_np = np.array(pil_img).astype(np.float32)
            # brightness
            if random.random() < 0.7:
                b = random.uniform(0.9, 1.1)
                img_np = img_np * b
            # contrast
            if random.random() < 0.7:
                c = random.uniform(0.9, 1.1)
                mean = img_np.mean(axis=(0,1), keepdims=True)
                img_np = (img_np - mean) * c + mean
            # color jitter (limited effect on near-grayscale data, so only small shifts are used)
            if random.random() < 0.3:
                shift = np.random.uniform(-3, 3, size=(1,1,3))
                img_np = img_np + shift
            img_np = _clip01(img_np)
            pil_img = Image.fromarray(img_np.astype(np.uint8))

        # 4. gaussian noise
        if random.random() < 0.3:
            img_np = np.array(pil_img).astype(np.float32)
            sigma = random.uniform(2, 6)
            noise = np.random.normal(0, sigma, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 255)
            pil_img = Image.fromarray(img_np.astype(np.uint8))

        # # 5. random slight crop + resize (preserve medical semantics; crop only 0~8%)
        # if random.random() < 0.5:
        #     max_crop = 0.08
        #     dw = int(w * random.uniform(0, max_crop))
        #     dh = int(h * random.uniform(0, max_crop))
        #     left = random.randint(0, dw)
        #     top = random.randint(0, dh)
        #     right = w - (dw - left)
        #     bottom = h - (dh - top)
        #     if right - left > 10 and bottom - top > 10:
        #         pil_img = pil_img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)

        return pil_img
    
    def __getitem__(self, idx):
        
        max_attempts = 5
        sample_max = 2 # Assuming this means max samples to process from one multi-label entry
        
        if self.valid:
            sample_max = 4
        if self.test:
            sample_max = 16 # deepseek-tiny=24, deepseek-small=16, ours=16
        attempt = 0
        original_idx = idx

        while attempt < max_attempts:
            idx_to_process = original_idx
            try:
                # Locate the corresponding folder with binary search.
                folder_idx = bisect.bisect_right(self.cum_counts, idx_to_process)
                offset = 0 if folder_idx == 0 else self.cum_counts[folder_idx - 1]
                dir_name = self.folder_keys[folder_idx]
                sample_idx = idx_to_process - offset # This is the index within the specific folder/datajson

                data_path = os.path.join(self.base_path, dir_name)
                json_path = os.path.join(data_path, 'dataset.json')
                if not (os.path.exists(data_path) and os.path.exists(json_path)):
                    raise FileNotFoundError(f"Data path or json file not found: {data_path} / {json_path}")
                datajson = load_json(json_path)

                sample_key = 'test' if self.valid else 'training'
                entry = datajson[sample_key][sample_idx]
                image_path = os.path.join(data_path, entry['image'])
                label_path = os.path.join(data_path, entry['label'])
                image_path = image_path.replace('/data/', '/data_resize224/')
                # label_path = label_path.replace('/data/', '/data_resize224/')
                if self.image_size == 256:
                    image_path = image_path.replace('/data_resize224/', '/data_resize256/')
                    # label_path = label_path.replace('/data_resize224/', '/data_resize256/')
                if not (os.path.exists(image_path) and os.path.exists(label_path)):
                    raise FileNotFoundError(f"Image or label file not found: {image_path} / {label_path}")

                if dir_name=="Totalsegmentator_dataset":
                    class_list = list(sorted(datajson['labels'].values()))
                else:
                    class_list = list(datajson['labels'].values())
                
                if 'background' in class_list:
                    class_list.remove('background')

                gt_shape = ast.literal_eval(label_path.split('.')[-2])
                sparse_label = sparse.load_npz(label_path)
                image_labels = torch.from_numpy(sparse_label.toarray().reshape(gt_shape))
                # Find non-zero channel indices.
                label_sums = torch.sum(image_labels, dim=(1, 2))
                label_ids = torch.nonzero(label_sums != 0, as_tuple=True)[0].tolist()

                if len(label_ids) == 0:
                    raise FileNotFoundError(f"No valid labels found in file: {label_path}")

                image_labels = image_labels[label_ids][..., 0]
                class_list = [self.rearrange_class(class_list[i].replace('_', ' ')) for i in label_ids]
                    
                # Randomly subsample early if the number of labels exceeds the threshold.
                if image_labels.size(0) > sample_max:
                    indices = torch.randperm(image_labels.size(0))[:sample_max]
                    image_labels = image_labels[indices]
                    class_list = [class_list[i] for i in indices.tolist()]

                    
                pil_images = load_pil_images([{"images": [image_path]}])
                

                rotate180_list = ["AbdomenCT1K", "AMOS2022_CT", "AMOS2022_MR", "BTCV", "FLARE21", "FLARE22", "Learn2Reg2022_L2R_task1_AbdomenCTCT", "Learn2Reg2022_L2R_task1_AbdomenMRCT_seg_CT", "Learn2Reg2022_L2R_task1_AbdomenMRCT_seg_MR", "MMWHS_CT", "MMWHS_MR", "MSD_Heart", "MSD_Liver", "MSD_Prostate", "MSD_Spleen", "Myops2020_t2", "SPPIN2023", "Totalsegmentator_dataset", "μ-RegPro_us"]
                rotate90_list = ["Continuous_Registration_task1", "finding-lungs-in-ct-data_3d", "KiTS", "KiTS2021", "KiTS2023"]
                if dir_name in rotate180_list:
                    pil_images[0] = pil_images[0].rotate(180)
                    image_labels = image_labels.flip(1).flip(2)
                elif dir_name in rotate90_list:
                    pil_images[0] = pil_images[0].rotate(270)
                    image_labels = image_labels.transpose(1, 2).flip(2)

                    
                seg_in, pil_img, image_labels, point_coords, point_labels,bboxs = self.full_transform(pil_images[0], image_labels)
                if not self.test and not self.valid:
                    pil_img = self._data_augmentation(pil_img)
                
                questions = []
                answers = []
                task_list = []
                promptype_list = []

                for idx_label, cls in enumerate(class_list):
                    if not self.valid:
                        promptype = self._sample_promptstype(self.promptype)
                        # promptype = self.promptype
                    else:
                        promptype = 't'

                    # prompt_num = 0
                    # # promptype = 'tp'
                    # if "p" in promptype:
                    #     prompt_num += 2
                    # if "b" in promptype:
                    #     prompt_num += 2
                    # if prompt_num >= 3:
                    #     prompt_num = 3
                    # # prompt_num=3
                    promptype_list.append(promptype)
                    # visionprompt = DEFAULT_PROMPT_TOKEN * prompt_num
                    visionprompt = ""
                    # visionprompt += ("<|point|>"+str(point_coords[idx_label, 0, :].tolist())+"<|plabel|>"+str(point_labels[idx_label].tolist())) if "p" in promptype else ""
                    # visionprompt += ("<|bbox|>"+str(bboxs[idx_label, 0, :].tolist())) if "b" in promptype else ""
                    
                    
                    text = cls.strip().lower().replace('_', ' ')
                    
                    selected_task = DEFAULT_SEG_Task_TOKEN
                    if random.random() < 0.5:
                        selected_task += DEFAULT_DET_Task_TOKEN
                        
            
                    detection_info = ""

                    # If any keyword in Disease_name overlaps with text, sample from both
                    # Disease_QUESTION_LIST and SHORT_QUESTION_LIST.
                    if any(disease in text for disease in Disease_name):
                        question_list = Disease_QUESTION_LIST + self.short_question_list
                    else:
                        question_list = self.short_question_list
                    
                    ''''''''''''
                    if self.valid:
                        # if any(disease in text for disease in Disease_name):
                        #     question_list = Disease_QUESTION_LIST # general
                        #     # question_list = Disease_QUESTION_LIST + self.short_question_list # mixture
                        # else:
                        #     question_list = self.short_question_list
                        question_list = self.short_question_list  # specific
                        selected_task = DEFAULT_SEG_Task_TOKEN + DEFAULT_DET_Task_TOKEN
                        
                        # print(f"Valid mode: Using selected_task {selected_task}")
                    ''''''''''''
                    
                    question_template = random.choice(question_list)
                    q = question_template.format(task_name=selected_task + visionprompt, class_name=text)
                    # q = question_template.format(task_name=selected_task, class_name="<|ref|>" + text + "<|/ref|>")
                    # q = question_template.format(task_name=selected_task, class_name=text)
                    # q = question_template.format(task_name='', class_name=text)
                    if DEFAULT_DET_Task_TOKEN in selected_task:
                        detection_info = "<|det|>" + str(bboxs[idx_label][0].tolist()) + "<|/det|>"
                        a = random.choice(self.answer_list_det).format(class_name=text + detection_info)
                        bbox_str = a.split("<|det|>")[1].split("<|/det|>")[0]
                        a = a.replace("<|det|>", "").replace("<|/det|>", "").replace(bbox_str, "")
                        a =  a + "{\n\"bbox_2d\": %s\n}" % bbox_str
                    else:
                        a = random.choice(self.answer_list_seg).format(class_name=text)
                    if "deepseek" in self.model_type:
                        q = "<image>\nLocate <|ref|>"+text+"<|/ref|> in the given image."
                        # box_for_deepseek
                        # a = self.box_for_deepseek(bboxs[idx_label], image_labels)
                        a = str(bboxs[idx_label][0].tolist())
                    elif "Qwen" in self.model_type and self.pure_qwen:
                        q = "<image>\nLocate "+text+" in this image and output the bbox coordinates in JSON format."
                        # a = str(bboxs[idx_label][0].tolist())
                    questions.append(q)
                    answers.append(a)
                    task_list.append(selected_task)

                prepare_inputs = []
                
                if "deepseek" in self.model_type:  # For deepseek
                    for q, a in zip(questions, answers):
                        conversation = [
                            {"role": "<|User|>",
                                "content": q,
                                "images": [image_path]},
                            {"role": "<|Assistant|>",
                                "content": ''},
                        ]
                        proc = self.vl_chat_processor(
                            conversations=conversation,
                            images=[pil_img],
                            force_batchify=False,
                            inference_mode=self.valid,
                            # system_prompt=system_prompt,
                        )
                        proc["sft_format"] = a
                        prepare_inputs.append(proc)
                elif 'Qwen' in self.model_type: 
                    visual_processed=self.vl_chat_processor.image_processor.preprocess(pil_img, return_tensors="pt")
                    pixel_values = visual_processed["pixel_values"]
                    grid_thw = visual_processed["image_grid_thw"][0]
                    
                    grid_thw_merged = copy.deepcopy(grid_thw)
                    if not isinstance(grid_thw, Sequence):
                        grid_thw_merged = [grid_thw_merged]
                        grid_thw = [grid_thw]
                        
                    grid_thw_merged = [
                        merged_thw.prod() // self.vl_chat_processor.image_processor.merge_size**2
                        for merged_thw in grid_thw_merged
                    ]
                    
                    for q, a in zip(questions, answers):                           
                        if self.valid:
                            # Validation mode: return raw data and batch it later in collate_fn.
                            messages_val = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": pil_img},
                                        {"type": "text",  "text": q.replace(DEFAULT_IMAGE_TOKEN, "")}
                                    ],
                                }
                            ]
                            prepare_inputs.append({
                                "messages": messages_val,
                                "pixel_values": torch.cat([pixel_values], dim=0),
                                "image_grid_thw": torch.cat([thw.unsqueeze(0) for thw in grid_thw], dim=0),
                                "sft_format": [
                                    {"role": "human", "content": q},
                                    {"role": "gpt",   "content": a}
                                ],
                            })
                            a=1
                        else:
                            conversations = [{"from": "human", "value": q},
                                            {"from": "gpt", "value": a},]
                            
                            data_dict = preprocess_qwen_2_visual([conversations], self.vl_chat_processor.tokenizer, grid_thw_image=grid_thw_merged if grid_thw_merged else None,grid_thw_video=None,)
                            position_ids, _ = get_rope_index_25(
                                self.vl_chat_processor.image_processor.merge_size,
                                data_dict["input_ids"], image_grid_thw=torch.stack(grid_thw, dim=0),video_grid_thw=None,second_per_grid_ts=None)
                            
                            conversations_gt = [{"from": "human", "value": q}, {"from": "gpt", "value": a},]
                            
                            # attention_mask = torch.ones_like(data_dict["input_ids"][0])
                            attention_mask = [data_dict["input_ids"][0].size(0)]
                            prepare_inputs.append({
                                "input_ids": data_dict["input_ids"][0],
                                "labels": data_dict["labels"][0],
                                "position_ids": position_ids,
                                "pixel_values": pixel_values,
                                "image_grid_thw": torch.cat([thw.unsqueeze(0) for thw in grid_thw], dim=0), 
                                "sft_format":conversations_gt,
                                "attention_mask": attention_mask,
                            })
                else:
                    raise ValueError("Invalid model type specified. Supported types are 'deepseek' and 'Qwen'.")

                final_image_labels = image_labels
                final_bboxs = bboxs
                seg_in = seg_in.unsqueeze(0).expand(final_image_labels.size(0), -1, -1, -1)
                image_path = [image_path] * final_image_labels.size(0)
                dir_name = [dir_name] * final_image_labels.size(0)

                class_tail_label = torch.zeros((1, self.cls_num), dtype=torch.float)
                
                return (seg_in, final_image_labels, final_bboxs,
                    point_coords, point_labels,
                    prepare_inputs, image_path, task_list, promptype_list, class_list, dir_name, class_tail_label)

            except (FileNotFoundError, UnidentifiedImageError) as e:
                current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
                print(f"Rank {current_rank}: Error processing original_idx {original_idx} (sample_idx {sample_idx} in {dir_name}), attempt {attempt+1}/{max_attempts}. Error: {e}")
                attempt += 1
                if attempt >= max_attempts:
                    error_message = f"Rank {current_rank}: Failed to load data for original_idx {original_idx} after {max_attempts} attempts. Last error: {e}"
                    raise RuntimeError(error_message) from e
                original_idx = random.randint(0, self.__len__() - 1)
                idx_to_process = original_idx
        raise RuntimeError(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}: Reached unexpected location in __getitem__ for original_idx {original_idx}. This indicates a logic flaw.")
    

if __name__ == "__main__":
    from transformers import AutoProcessor
    path = "/mnt/disk2/hwj/Tails/weight/HuatuoGPT-Vision-7B-Qwen2.5VL"
    vl_chat_processor = AutoProcessor.from_pretrained(path, model_max_length=256, max_pixels=224**2*3)
    vl_chat_processor.tokenizer.padding_side = 'left'
    dataset = ReferSegDataset(processor=vl_chat_processor, model_type='Qwen', valid=True)
    for i in range(5):
        data = dataset[i]
        seg_in, image_labels, bboxs, point_coords, point_labels, prepare_inputs, image_path, task_list, promptype_list, class_list, dir_name = data
        print(f"Data {i}:")
        print(f"  seg_in shape: {seg_in.shape}")
        print(f"  image_labels shape: {image_labels.shape}")
        print(f"  bboxs shape: {bboxs.shape}")
        print(f"  point_coords shape: {point_coords.shape}")
        print(f"  point_labels shape: {point_labels.shape}")
        print(f"  prepare_inputs length: {len(prepare_inputs)}")
        print(f"  image_path: {image_path}")
        print(f"  task_list: {task_list}")
        print(f"  promptype_list: {promptype_list}")
        print(f"  class_list: {class_list}")
        print(f"  dir_name: {dir_name}")
