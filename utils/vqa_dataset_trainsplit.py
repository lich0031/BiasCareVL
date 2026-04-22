import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import copy
import matplotlib.pyplot as plt
import warnings
import math
import collections

from model.deepseek_vl2.utils.io import load_pil_images
from model.IMIS.segment_anything.utils.transforms import ResizeLongestSide
from model.Qwen2_vl25.qwenfinetune.qwenvl.data.data_qwen import preprocess_qwen_2_visual
from model.Qwen2_vl25.qwenfinetune.qwenvl.data.rope2d import get_rope_index_25
from .utils import DEFAULT_IMAGE_TOKEN, GLOBAL_SEED, DEFAULT_PROMPT_TOKEN, GENERATE_QUESTION_LIST

from qwen_vl_utils import process_vision_info
import pandas as pd

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
    
class VQADataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __init__(
        
        self,
        base_image_dir: str,
        processor: Any,
        conv: Any,
        vqa_data: str = "PubMedVision",
        # vqa_data: str = "PubMedVision||MIMIC||SLAKE||PMC_VQA||vqa-rad||cxr-lt",
        image_size: int = 224,
        samples_per_epoch: int = 500 * 8 * 2 * 10,
        torch_dtype: torch.dtype = torch.float32,
        valid: bool = False,
        promptype: str = "t",
        test = False,
        modality = '',  # "CT", "MRI", "X-ray"
        openclosetype = "all",  # "open", "close", "all"
        cls_num = 0,
        cls_token = None
    ):
        self.processor = processor
        self.conv = conv
        self.image_size = image_size
        self.resize = ResizeLongestSide(image_size)
        self.dtype = torch_dtype
        self.promptype = promptype
        self.samples_per_epoch = samples_per_epoch
        self.valid = valid
        self.test = test
        self.modality = modality
        self.openclosetype = openclosetype
        self.cls_num = cls_num
        self.cls_token = cls_token

        self.datasets = vqa_data.split("||")
        self._load_all_data()
        self._build_sampling_weights() 
    
        if valid:
            # print(f"Testing VQA samples - PubMedVision: {len(self.pubmed_data)}, MIMIC: {len(self.mimic_data)}")
            print(f"Testing VQA samples - PubMedVision: {len(self.pubmed_data)}, MIMIC: {len(self.mimic_data)}, SLAKE: {len(self.slake_data)}, PMC_VQA: {len(self.pmc_data)}, vqa-rad: {len(self.vqa_rad)}, path-vqa: {len(self.path_vqa)}, cxr-lt: {len(self.cxr_lt)}, isic2018: {len(self.isic2018)}, isic2019: {len(self.isic2019)}, vqamed2019: {len(self.vqamed2019)}, kvasir_vqa: {len(self.kvasir_vqa)}")
        else:
            # print(f"Training VQA samples - PubMedVision: {len(self.pubmed_data)}, MIMIC: {len(self.mimic_data)}")
            print(f"Training VQA samples - PubMedVision: {len(self.pubmed_data)}, MIMIC: {len(self.mimic_data)}, SLAKE: {len(self.slake_data)}, PMC_VQA: {len(self.pmc_data)}, vqa-rad: {len(self.vqa_rad)}, path-vqa: {len(self.path_vqa)},cxr-lt: {len(self.cxr_lt)}, isic2018: {len(self.isic2018)}, isic2019: {len(self.isic2019)}, vqamed2019: {len(self.vqamed2019)}, kvasir_vqa: {len(self.kvasir_vqa)}")

    def _build_sampling_weights(self):
        """Build weighted sampling probabilities based on dataset lengths."""
        self._ds_lengths = {}
        for ds in self.datasets:
            if ds == "PubMedVision":
                length = getattr(self, "pubmed_len", len(self.pubmed_data))
            elif ds == "MIMIC":
                length = getattr(self, "mimic_len", len(self.mimic_data))
            elif ds == "SLAKE":
                length = len(self.slake_data)
            elif ds == "PMC_VQA":
                length = len(self.pmc_data)
            elif ds == "vqa-rad":
                length = len(self.vqa_rad)
            elif ds == "path-vqa":
                length = len(self.path_vqa)
            elif ds == "cxr-lt":
                length = len(self.cxr_lt)
            elif ds == "isic2018":
                length = len(self.isic2018)
            elif ds == "isic2019":
                length = len(self.isic2019)
            elif ds == "vqamed2019":
                length = len(self.vqamed2019)
            elif ds == "kvasir_vqa":
                length = len(self.kvasir_vqa)
            else:
                assert False, f"Unsupported dataset: {ds}"
            self._ds_lengths[ds] = max(int(length), 0)

        total = sum(self._ds_lengths.values()) or 1
        self._sampling_weights = [self._ds_lengths[d] / total for d in self.datasets]

    def _weighted_sample_dataset(self):
        """Return a dataset name sampled according to length-based weights."""
        if self.valid:
            return self.datasets[0]
        return random.choices(self.datasets, weights=self._sampling_weights, k=1)[0]
    
    def _sample_promptstype(self, prompt_type: str) -> str:
        if not isinstance(prompt_type, str) or len(prompt_type) <= 1:
            return prompt_type or ""
        chars = list(prompt_type)
        num_to_keep = random.randint(1, len(chars))
        kept = random.sample(chars, num_to_keep)
        # Preserve the original order.
        kept.sort(key=lambda c: prompt_type.index(c))
        return "".join(kept)
    
    def labels_to_text(self, row, label_names):
        labels = [label_names[i] for i, v in enumerate(row) if v == 1]
        random.shuffle(labels)
        if len(labels) == 0:
            return "Normal"
        return ", ".join(labels)
    
    def _load_all_data(self):
        """Preload JSON files and image roots for all datasets."""
        self.pubmed_data = []
        self.slake_data = []
        self.mimic_data = []
        self.pmc_data = []
        self.vqa_rad = []
        self.path_vqa = []
        self.cxr_lt = []
        self.isic2018 = []
        self.isic2019 = []
        self.vqamed2019 = []
        self.kvasir_vqa = []
        self.length = 0
        # each_valid_num = 1000
        each_valid_num = 300
        # each_valid_num = -1
        for ds in self.datasets:
            if ds == "PubMedVision":
                self.pubmed_root = Path('/mnt/disk2/hwj/Tails/dataset/PubMedVision/')
                full_path  = self.pubmed_root / "PubMedVision_Alignment_VQA.json"
                train_path = self.pubmed_root / "PubMedVision_Alignment_VQA_train.json"
                test_path  = self.pubmed_root / "PubMedVision_Alignment_VQA_test.json"

                if train_path.exists() and test_path.exists():
                    with open(train_path, "r") as f:
                        train_data = json.load(f)
                    with open(test_path, "r") as f:
                        test_data = json.load(f)
                else:
                    with open(full_path, "r") as f:
                        all_data = json.load(f)
                    SPLIT_SEED = 2025
                    TEST_SIZE = len(all_data) // 20  # Reserve 10% for the test split.
                    rng = random.Random(SPLIT_SEED)
                    indices = list(range(len(all_data)))
                    rng.shuffle(indices)

                    test_data  = [all_data[i] for i in indices[:TEST_SIZE]]
                    train_data = [all_data[i] for i in indices[TEST_SIZE:]]

                    with open(train_path, "w") as f:
                        json.dump(train_data, f, ensure_ascii=False, indent=2)
                    with open(test_path, "w") as f:
                        json.dump(test_data, f, ensure_ascii=False, indent=2)
                    print(f"[PubMedVision] Saved split: train={len(train_data)} test={len(test_data)}")
                    
                if self.valid:
                    if self.test:
                        self.pubmed_data = test_data                        
                        pubmed_data_modality_filtered = []
                        if self.modality != '':
                            # modality_list = ['computed tomography', 'dermoscopy', 'digital photography', 'endoscopy', 'fundus photography', 'infrared reflectance imaging', 
                            # 'magnetic resonance imaging', 'microfluidic device', 'microscopy images', 'optical coherence tomography', 'others', 'ultrasound']
                            for m in range(len(test_data)):
                                row = test_data[m]
                                if self.modality.lower() == row["modality"].lower():
                                    pubmed_data_modality_filtered.append(row)
                            self.pubmed_data = pubmed_data_modality_filtered      
                    else:
                        self.pubmed_data = test_data[:each_valid_num]
                else:
                    self.pubmed_data = train_data
                    
                self.pubmed_len = len(self.pubmed_data)
                self.length += self.pubmed_len
                
            elif ds == "MIMIC":
                self.mimic_dir = Path('/mnt/disk3/MIMIC-DATA-Final/MIMIC-CXR')
                csv_path = self.mimic_dir / 'BASE-MIMIC.csv'
                self.mimic_data = pd.read_csv(csv_path, sep=',')
                self.mimic_data.fillna('', inplace=True)

                if self.valid:
                    if self.test:
                        self.mimic_data = self.mimic_data[self.mimic_data["split"]=="test"]
                    else:
                        self.mimic_data = self.mimic_data[self.mimic_data["split"]=="test"][:each_valid_num]
                else:
                    assert True
                self.mimic_len  = len(self.mimic_data)
                self.length += self.mimic_len
            elif ds == "cxr-lt":
                self.mimic_dir = Path('/mnt/disk3/MIMIC-DATA-Final/MIMIC-CXR')
                self.cxr_lt_dir = Path('/mnt/disk2/hwj/Tails/dataset/CXR-LT')
                
                jpath2 = None
                if not self.valid:
                    jpath = self.cxr_lt_dir / "train_labeled.csv"
                    jpath2 = self.cxr_lt_dir / "development_labeled_task1.csv"
                else:
                    jpath = self.cxr_lt_dir / "test_labeled_task1.csv"
                        
                cxr_lt = pd.read_csv(jpath, sep=",")
                img_paths = cxr_lt["fpath"].values
                label_names = cxr_lt.columns[6:]  # Multi-label columns start from the 7th column.
                multi_labels = cxr_lt[label_names].values
                label_names_str = [str(x) for x in label_names]
                    
                if jpath2 is not None and not self.valid:
                    cxr_lt2 = pd.read_csv(jpath2, sep=",")
                    img_paths2 = cxr_lt2["fpath"].values
                    multi_labels2 = cxr_lt2[label_names].values
                    img_paths = np.concatenate((img_paths, img_paths2), axis=0)
                    multi_labels = np.concatenate((multi_labels, multi_labels2), axis=0)

                self.cxr_lt = []
                for path, row in zip(img_paths, multi_labels):
                    ans = self.labels_to_text(row, label_names)
                    if self.cls_token is not None:
                        qa_item = {
                            "path": str(path),
                            # "question": self.cls_token + "What abnormalities are present in this chest X-ray? " + label_names_str.__str__(),
                            "question": "What abnormalities are present in this chest X-ray? " + label_names_str.__str__() + self.cls_token,
                            "answer": ans,
                            "cls_label": row.tolist()
                        }
                    else:
                        qa_item = {
                            "path": str(path),
                            "question": "What abnormalities are present in this chest X-ray? " + label_names_str.__str__(),
                            "answer": ans,
                            "cls_label": row.tolist()
                        }
                    self.cxr_lt.append(qa_item)
                    
                if self.valid and not self.test:
                    class_to_indices = collections.defaultdict(list)
                    for idx, item in enumerate(self.cxr_lt):
                        labels = np.array(item["cls_label"])
                        for cls_idx, val in enumerate(labels):
                            if val == 1:
                                class_to_indices[cls_idx].append(idx)

                    # Count samples for each class.
                    class_counts = {cls: len(idxs) for cls, idxs in class_to_indices.items()}
                    print("Class distribution before balancing:", class_counts)

                    # min_count = min(class_counts.values())
                    min_count = 30

                    # Random sampling.
                    balanced_indices = set()
                    for cls, idxs in class_to_indices.items():
                        if len(idxs) > min_count:
                            selected = random.sample(idxs, min_count)
                        else:
                            selected = idxs
                        balanced_indices.update(selected)

                    balanced_indices = sorted(list(balanced_indices))
                    self.cxr_lt = [self.cxr_lt[i] for i in balanced_indices]
                    print(f"Balanced validation set size: {len(self.cxr_lt)}")

                self.length += len(self.cxr_lt)
        
            elif ds == "isic2018":
                self.isic2018_dir = Path('/mnt/disk2/hwj/Tails/dataset/isic/isic2018')
                if not self.valid:
                    jpath = self.isic2018_dir / 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
                else:
                    if self.test:
                        jpath = self.isic2018_dir / 'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'
                    else:
                        jpath = self.isic2018_dir / 'ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv'
                        
                isic2018_csv = pd.read_csv(jpath, sep=",")
                img_paths = isic2018_csv.image
                label_names = isic2018_csv.columns[1:]  # Multi-label columns start from the 2nd column.
                multi_labels = isic2018_csv.values[:, 1:]
                label_names_str = [str(x) for x in label_names]

                # Build VQA samples.
                self.isic2018 = []
                for path, row in zip(img_paths, multi_labels):
                    ans = self.labels_to_text(row, label_names)
                    if self.cls_token is not None:
                        qa_item = {
                            "path": str(path),
                            # "question": self.cls_token + "What is the most likely diagnosis of this dermoscopic lesion? " + label_names_str.__str__()
                            "question": "What is the most likely diagnosis of this dermoscopic lesion? " + label_names_str.__str__()  + self.cls_token,
                            "answer": ans,
                            "cls_label": row.tolist()
                        }
                    else:
                        qa_item = {
                            "path": str(path),
                            "question": "What is the most likely diagnosis of this dermoscopic lesion? " + label_names_str.__str__(),
                            "answer": ans,
                            "cls_label": row.tolist()
                    }
                    self.isic2018.append(qa_item)
                if self.valid and not self.test:
                    self.isic2018 = self.isic2018[:each_valid_num]
                self.length += len(self.isic2018)

            elif ds == "isic2019":
                self.isic2019_dir = Path('/mnt/disk2/hwj/Tails/dataset/isic/isic2019')
                if self.valid or self.test:
                    jpath = '/mnt/disk2/hwj/Tails/dataset/isic/isic2019/ISIC_2019_Test_GroundTruth.csv'
                    isic2019_csv = pd.read_csv(jpath, sep=",")
                    img_paths = isic2019_csv.image
                    label_names = isic2019_csv.columns[1:-2]
                    multi_labels = isic2019_csv.values[:, 1:-2]
                else:
                    jpath = '/mnt/disk2/hwj/Tails/dataset/isic/isic2019/ISIC_2019_Training_GroundTruth.csv'
                    isic2019_csv = pd.read_csv(jpath, sep=",")
                    img_paths = isic2019_csv.image
                    label_names = isic2019_csv.columns[1:]  
                    multi_labels = isic2019_csv.values[:, 1:]
                label_names_str = [str(x) for x in label_names if x != 'UNK']
                self.isic2019 = []
                for path, row in zip(img_paths, multi_labels):
                    ans = self.labels_to_text(row, label_names)
                    if ans == 'AK':
                        ans = 'AKIEC'
                    qa_item = {
                        "path": str(path),
                        "question": "What is the most likely diagnosis of this dermoscopic lesion? " + label_names_str.__str__(),
                        "answer": ans,
                        "cls_label": row.tolist()
                    }
                    if ans == 'UNK':
                        continue
                    self.isic2019.append(qa_item)
                if self.valid and not self.test:
                    self.isic2019 = self.isic2019[:each_valid_num]
                self.length += len(self.isic2019)


            elif ds == "PMC_VQA":
                self.pmc_dir = Path('/mnt/disk2/hwj/Tails/dataset/PMC-VQA')
                if self.valid:
                    jpath = self.pmc_dir / "test_2.csv"
                    self.pmc_data = pd.read_csv(jpath)
                    self.pmc_data.fillna('', inplace=True)
                    
                    if not self.test:
                        self.pmc_data = self.pmc_data[:each_valid_num]
                else:
                    jpath = self.pmc_dir / "train_2.csv"
                    self.pmc_data = pd.read_csv(jpath)
                    self.pmc_data.fillna('', inplace=True)

                ''''''''''''''''''''''For modality filtering'''''''''''''''''''''
                if self.test and self.modality != '' and self.valid:

                    # ----------------- Modality filtering -----------------
                    import re
                    from collections import defaultdict

                    # Allow self.modality to be set externally; default to ALL otherwise.
                    self.modality = getattr(self, "modality", "ALL") or "ALL"
                    print(f"[PMC_VQA] modality filter target = {self.modality}")

                    MODALITY_SPECS = [
                        # High priority (composite / special cases).
                        dict(name="PET_CT", priority=1, keywords=["pet/ct", "pet-ct", "pet ct"], regex=[r"\bpet[-/ ]ct\b"]),
                        dict(name="PET_MRI", priority=1, keywords=["pet/mri", "pet-mri", "pet mri"], regex=[r"\bpet[-/ ]mri\b"]),
                        dict(name="MRI_MRA_TOF", priority=2, keywords=["tof mra", "tof angiography", "time-of-flight angiography"], regex=[r"\btof\b.*\b(mra|angiograph)"]),
                        dict(name="MRI_PERFUSION", priority=2, keywords=["perfusion mri", "mr perfusion"], regex=[r"\b(perfusion)\b.*\bmri\b"]),
                        dict(name="MRI_SWI", priority=2, keywords=["susceptibility weighted", "swi"], regex=[r"\bswi\b"]),
                        dict(name="MRI_FMIR", priority=2, keywords=["fmri", "functional mri"], regex=[r"\bfmri\b", r"\bfunctional\s+mri\b"]),
                        dict(name="MRI_DWI", priority=2, keywords=["diffusion weighted imaging", "dwi", "diffusion-weighted"], regex=[r"\bdwi\b", r"\bdiffusion(-|\s)weighted"]),
                        dict(name="MRI_ADC", priority=2, keywords=["adc map", "apparent diffusion coefficient"], regex=[r"\badc\b"]),
                        dict(name="MRI_FLAIR", priority=2, keywords=["flair", "t2 flair"], regex=[r"\bflair\b"]),
                        dict(name="MRI_T1C", priority=2, keywords=["t1ce", "t1 contrast", "post-contrast t1", "gadolinium-enhanced t1"], regex=[r"\bt1(\s*|-)?(ce|post|postcontrast)\b", r"\bcontrast(\s|-)?enhanced\s+t1\b"]),
                        dict(name="MRI_T1", priority=3, keywords=["t1-weighted", "t1 weighted", "t1w", "t1 wi"], regex=[r"\bt1(\s*|-)?(w|weighted|wi)\b"]),
                        dict(name="MRI_T2", priority=3, keywords=["t2-weighted", "t2 weighted", "t2w", "t2 wi","t2wi","t2w image","t2 sequence"], regex=[r"\bt2(\s*|-)?(w|weighted|wi)\b"]),
                        dict(name="MRI_SPECTROSCOPY", priority=3, keywords=["mr spectroscopy", "mrs"], regex=[r"\bmrs\b"]),
                        dict(name="MRI_CINE", priority=3, keywords=["cine mri", "cine sequence"], regex=[r"\bcine\s+mri\b"]),
                        # CT / other.
                        dict(name="CT_ANGIO", priority=3, keywords=["cta", "ct angiography", "computed tomography angiography"], regex=[r"\bcta\b"]),
                        dict(name="CT_PERFUSION", priority=3, keywords=["ct perfusion"], regex=[r"\bct\s+perfusion\b"]),
                        dict(name="CT_CONTRAST", priority=3, keywords=["contrast-enhanced ct", "contrast enhanced ct"], regex=[r"\bcontrast(-|\s)?enhanced\s+ct\b"]),
                        dict(name="CT_LOWDOSE", priority=3, keywords=["low dose ct", "low-dose ct"], regex=[r"\blow(-|\s)dose\s+ct\b"]),
                        dict(name="ULTRASOUND_DOPPLER", priority=3, keywords=["doppler ultrasound", "color doppler"], regex=[r"\bdoppler\b.*\bultra"]),
                        dict(name="ULTRASOUND", priority=4, keywords=["ultrasound", "us scan", "sonography", "ultrasonography"], regex=[r"\bultra(sound)?\b", r"\bsonograph(y)?\b"]),
                        dict(name="FUNDUS", priority=3, keywords=["fundus photography", "fundus photo", "retinal photograph", "ophthalmic fundus", "fundus", "retinal image"], regex=[r"\bfundus\b", r"\bretina(l)?\b"]),
                        dict(name="OCT", priority=3, keywords=["optical coherence tomography", "oct"], regex=[r"\boct\b"]),
                        dict(name="DERMOSCOPY", priority=3, keywords=["dermoscopy", "dermoscopic"], regex=[r"\bdermoscop(y|ic)\b"]),
                        dict(name="PHOTOACOUSTIC", priority=3, keywords=["photoacoustic imaging", "photoacoustic"], regex=[r"\bphotoacoustic\b"]),
                        dict(name="COLONOSCOPY", priority=3, keywords=["colonoscopy"], regex=[r"\bcolonoscopy\b"]),
                        dict(name="BRONCHOSCOPY", priority=3, keywords=["bronchoscopy"], regex=[r"\bbronchoscopy\b"]),
                        dict(name="ENDOSCOPY", priority=4, keywords=["endoscopy", "endoscopic image", "endoscopic"], regex=[r"\bendoscop(y|ic)\b"]),
                        dict(name="ELECTRON_MICROSCOPY", priority=3, keywords=["electron microscopy", "tem", "sem"], regex=[r"\b(t|s)em\b"]),
                        dict(name="CONFOCAL_MICROSCOPY", priority=3, keywords=["confocal microscopy", "confocal micrograph"], regex=[r"\bconfocal\b"]),
                        dict(name="IHC", priority=3, keywords=["immunohistochemistry", "ihc stain", "ihc"], regex=[r"\bihc\b"]),
                        dict(name="IMMUNOFLUORESCENCE", priority=3, keywords=["immunofluorescence", "if stain"], regex=[r"\bimmunofluorescence\b"]),
                        dict(name="FLUORESCENCE", priority=4, keywords=["fluorescence imaging", "fluorescent", "fluorescence"], regex=[r"\bfluorescen(ce|t)\b"]),
                        dict(name="H_E", priority=3, keywords=["h&e", "hematoxylin and eosin", "haematoxylin"], regex=[r"\bh&?e\b"]),
                        dict(name="HISTOPATHOLOGY", priority=3, keywords=["histopathology", "histopathological", "histological", "pathology slide", "whole slide image", "wsi"], regex=[r"\bhistopatholog(y|ic|ical)\b", r"\bwhole\s+slide\b"]),
                        dict(name="MICROSCOPY", priority=4, keywords=["microscopy", "microscopic", "micrograph"], regex=[r"\bmicroscop(y|ic)\b"]),
                        dict(name="INFRARED", priority=3, keywords=["infrared reflectance imaging", "infrared reflectance", "infrared", "iri"], regex=[r"\binfrared\b"]),
                        dict(name="POLARIZATION", priority=3, keywords=["polarization imaging", "polarized light"], regex=[r"\bpolariz(ed|ation)\b"]),
                        dict(name="DARKFIELD", priority=3, keywords=["darkfield", "dark-field"], regex=[r"\bdark[- ]field\b"]),
                        dict(name="PHASE_CONTRAST", priority=3, keywords=["phase contrast microscopy", "phase-contrast"], regex=[r"\bphase(-|\s)contrast\b"]),
                        dict(name="ANGIO_ULTRASOUND", priority=3, keywords=["intravascular ultrasound", "ivus"], regex=[r"\bivus\b"]),
                        dict(name="ANGIOGRAPHY", priority=3, keywords=["angiography", "angiogram"], regex=[r"\bangiograph(y|ic)?\b"]),
                        dict(name="DSA", priority=3, keywords=["digital subtraction angiography", "dsa"], regex=[r"\bdsa\b"]),
                        dict(name="ECHOCARDIOGRAPHY", priority=3, keywords=["echocardiography", "echo cardiography"], regex=[r"\bechocardiograph(y)?\b"]),
                        dict(name="MAMMOGRAPHY", priority=3, keywords=["mammography", "mammogram"], regex=[r"\bmammograph(y)?\b"]),
                        dict(name="SCINTIGRAPHY", priority=3, keywords=["scintigraphy", "bone scan"], regex=[r"\bscintigraphy\b"]),
                        dict(name="ELASTOGRAPHY", priority=3, keywords=["elastography", "shear wave elastography", "strain elastography"], regex=[r"\belastograph(y)?\b"]),
                        dict(name="FLUOROSCOPY", priority=3, keywords=["fluoroscopy", "fluoroscopic"], regex=[r"\bfluoroscopy\b"]),
                        dict(name="PET", priority=4, keywords=["positron emission tomography", "pet scan", "pet"], regex=[r"\bpet\b"]),
                        # Generic (lowest priority).
                        dict(name="CT_GENERIC", priority=9, keywords=["computed tomography", "ct scan", "ct image", "ct"], regex=[r"\bct\b"]),
                        dict(name="MRI_GENERIC", priority=9, keywords=["magnetic resonance imaging", "mri", "mr image"], regex=[r"\bmri\b"]),
                        dict(name="GENERIC_MEDICAL_PHOTO", priority=9, keywords=["clinical photograph", "digital photography", "medical photo"], regex=[r"\bphotograph\b"]),
                        dict(name="X_RAY", priority=9, keywords=["x-ray", "xray", "x ray", "radiograph", "radiography"], regex=[r"\bx[- ]?ray\b", r"\bradiograph(y)?\b"]),
                        ]

                    # Precompile regular expressions.
                    for spec in MODALITY_SPECS:
                        spec["keywords"] = list({kw.lower() for kw in spec["keywords"]})
                        compiled = []
                        for pat in spec.get("regex", []):
                            try:
                                compiled.append(re.compile(pat, re.IGNORECASE))
                            except re.error:
                                pass
                        spec["_cregex"] = compiled

                    modality_counts = defaultdict(int)
                    assigned_rows, detected_modalities = [], []

                    for ridx in range(len(self.pmc_data)):
                        row = self.pmc_data.iloc[ridx]
                        
                        answer_choice = "Choice " + row.get("Answer", "")
                        text = " ".join([
                            str(row.get("Question","")),
                            str(row.get("Caption","")),
                            str(row.get(answer_choice,"")),
                        ]).lower()

                        candidates = []
                        for spec in MODALITY_SPECS:
                            hit = False
                            for kw in spec["keywords"]:
                                if re.search(rf"\b{re.escape(kw)}\b", text) or kw in text:
                                    hit = True
                                    break
                            if not hit:
                                for rg in spec["_cregex"]:
                                    if rg.search(text):
                                        hit = True
                                        break
                            if hit:
                                candidates.append(spec)

                        if not candidates:
                            assigned = "others"
                        else:
                            candidates.sort(key=lambda s: (s["priority"], -len(s["name"])))
                            assigned = candidates[0]["name"]

                        modality_counts[assigned] += 1
                        assigned_rows.append(row)
                        detected_modalities.append(assigned)

                    # Write results back to the DataFrame.
                    self.pmc_data = pd.DataFrame(assigned_rows).reset_index(drop=True)
                    self.pmc_data["DetectedModality"] = detected_modalities

                    # Fine-grained statistics.
                    print("[PMC_VQA] Fine modality counts:")
                    for m, c in sorted(modality_counts.items(), key=lambda x: (-x[1], x[0])):
                        print(f"  {m}: {c}")
                    print(f"[PMC_VQA] Total samples after fine classification: {len(self.pmc_data)}")

                    # Map fine-grained labels to PubMedVision-style coarse modalities.
                    # Unmapped labels remain as independent coarse classes.
                    coarse_map = {
                        # CT
                        "CT_GENERIC": "computed tomography",
                        "CT_ANGIO": "computed tomography",
                        "CT_CONTRAST": "computed tomography",
                        "CT_PERFUSION": "computed tomography",
                        "CT_LOWDOSE": "computed tomography",
                        # MRI
                        "MRI_GENERIC": "magnetic resonance imaging",
                        "MRI_T1": "magnetic resonance imaging",
                        "MRI_T1C": "magnetic resonance imaging",
                        "MRI_T2": "magnetic resonance imaging",
                        "MRI_FLAIR": "magnetic resonance imaging",
                        "MRI_DWI": "magnetic resonance imaging",
                        "MRI_ADC": "magnetic resonance imaging",
                        "MRI_SWI": "magnetic resonance imaging",
                        "MRI_PERFUSION": "magnetic resonance imaging",
                        "MRI_FMIR": "magnetic resonance imaging",
                        "MRI_SPECTROSCOPY": "magnetic resonance imaging",
                        "MRI_CINE": "magnetic resonance imaging",
                        "MRI_MRA_TOF": "magnetic resonance imaging",
                        "MRI_MRA": "magnetic resonance imaging",
                        # Ultrasound
                        "ULTRASOUND": "ultrasound",
                        "ULTRASOUND_DOPPLER": "ultrasound",
                        "ANGIO_ULTRASOUND": "ultrasound",
                        "ECHOCARDIOGRAPHY": "ultrasound",
                        # Microscopy / Pathology
                        "ELECTRON_MICROSCOPY": "microscopy images",
                        "CONFOCAL_MICROSCOPY": "microscopy images",
                        "MICROSCOPY": "microscopy images",
                        "H_E": "microscopy images",
                        "HISTOPATHOLOGY": "microscopy images",
                        "IHC": "microscopy images",
                        "IMMUNOFLUORESCENCE": "microscopy images",
                        "FLUORESCENCE": "microscopy images",
                        "PHASE_CONTRAST": "microscopy images",
                        "DARKFIELD": "microscopy images",
                        "POLARIZATION": "microscopy images",
                        # Fundus / OCT / Dermoscopy
                        "FUNDUS": "fundus photography",
                        "OCT": "optical coherence tomography",
                        "DERMOSCOPY": "dermoscopy",
                        # Endoscopy group
                        "ENDOSCOPY": "endoscopy",
                        "BRONCHOSCOPY": "endoscopy",
                        "COLONOSCOPY": "endoscopy",
                        # Infrared
                        "INFRARED": "infrared reflectance imaging",
                        # Generic medical photo
                        "GENERIC_MEDICAL_PHOTO": "digital photography",
                        "X_RAY": "x-ray",
                    }

                    coarse_list = []
                    for fine in self.pmc_data["DetectedModality"]:
                        coarse_list.append(coarse_map.get(fine, fine))  # Keep the original name if no merge target exists.
                    self.pmc_data["CoarseModality"] = coarse_list

                    # Coarse statistics, including newly introduced unmapped categories.
                    coarse_counts = self.pmc_data["CoarseModality"].value_counts().to_dict()
                    print("[PMC_VQA] Coarse modality counts (merged + unmapped kept):")
                    for m, c in sorted(coarse_counts.items(), key=lambda x: (-x[1], x[0])):
                        print(f"  {m}: {c}")

                    # Filter by coarse modality first, then fall back to fine modality.
                    target = (self.modality or "ALL").strip().lower()
                    if target not in ("all", "*", ""):
                        before = len(self.pmc_data)
                        mask = self.pmc_data["CoarseModality"].str.lower() == target
                        if not mask.any():
                            mask = self.pmc_data["DetectedModality"].str.lower() == target
                        self.pmc_data = self.pmc_data[mask].reset_index(drop=True)
                        after = len(self.pmc_data)
                        print(f"[PMC_VQA] Filter modality={self.modality} -> {after}/{before}")
                        if after == 0:
                            print(f"[PMC_VQA][Warning] No samples matched '{self.modality}'.")
                    else:
                        print("[PMC_VQA] No modality filtering (ALL).")

                    print(f"[PMC_VQA] Final usable samples: {len(self.pmc_data)}")
                    # ----------------- End modality filtering ----------------
                ''''''''''''''''''''''For modality filtering'''''''''''''''''''''
                self.length += len(self.pmc_data)
                
            elif ds == "SLAKE":
                from datasets import load_dataset
                ds = load_dataset("BoKelvin/SLAKE")
                if self.valid:
                    if self.test:
                        self.slake_data = ds["test"]              
                        if self.openclosetype == "close":
                            self.slake_data = self.slake_data.filter(lambda x: x["answer_type"] != "OPEN")
                        elif self.openclosetype == "open":
                            self.slake_data = self.slake_data.filter(lambda x: x["answer_type"] == "OPEN")
                        elif self.openclosetype == "all":
                            pass

                        if self.modality != '':
                            slake_modality = []
                            for mod in range(len(self.slake_data)):
                                if self.slake_data[mod]["modality"].lower() == self.modality.lower():
                                    slake_modality.append(self.slake_data[mod])
                            self.slake_data = slake_modality
                        
                    else:
                        self.slake_data = ds["validation"].select(range(each_valid_num))
                else:
                    self.slake_data = ds["train"]
                self.length += len(self.slake_data)

            elif ds == "path-vqa":
                from datasets import load_dataset
                ds = load_dataset("flaviagiammarino/path-vqa")
                if self.valid:
                    if self.test:
                        self.path_vqa = ds["test"]
                        if self.openclosetype == "close":
                            self.path_vqa = self.path_vqa.filter(lambda x: x["answer"] == "yes" or x["answer"] == "no")
                            # self.path_vqa = self.path_vqa.map(lambda x: {"question": x["question"] + " Answer yes or no."})
                        elif self.openclosetype == "open":
                            self.path_vqa = self.path_vqa.filter(lambda x: x["answer"] != "yes" and x["answer"] != "no")
                        elif self.openclosetype == "all":
                            pass
                        else:
                            raise ValueError(f"Unsupported path-vqa type: {self.openclosetype}")
                    else:
                        self.path_vqa = ds["validation"].select(range(each_valid_num))
                else:
                    self.path_vqa = ds["train"]
                self.length += len(self.path_vqa)
                
            elif ds == "vqa-rad":
                from datasets import load_dataset
                ds = load_dataset("/mnt/disk2/hwj/Tails/dataset/vqa-rad/")
                if self.valid:
                    if self.test:
                        self.vqa_rad = ds["test"]
                        
                        modality_terms = {
                            'ct': ['ct', 'computed tomography', 'computerized tomography', 'ct-scan'],
                            'x-ray': ['x-ray', 'radiograph', 'radiography', 'xray', 'xr'],
                            'mri': ['mri', 'magnetic resonance imaging', 'magnetic resonance', 'mri scan', 'magnetic resonance imaging scan'],
                            'others': ['others', 'none', 'undefined']
                        }
                        def matches_modality(text, modality):
                            terms = modality_terms.get(modality.lower(), [])
                            return any(term in text.lower() for term in terms)                        
                        def filter_entries_by_modality(entries, modality):
                            filtered_entries = []
                            for entry in entries:
                                q = entry["question"]
                                a = entry["answer"]
                                text = q + ' ' + a
                                if matches_modality(text, modality):
                                    filtered_entries.append(entry)
                            return filtered_entries                        
                        if self.modality == '':
                            modality_counts = {'ct': 0, 'x-ray': 0, 'mri': 0, 'others': 0}
                            for entry in self.vqa_rad:
                                text = entry["question"] + ' ' + entry["answer"]
                                if matches_modality(text, 'ct'):
                                    modality_counts['ct'] += 1
                                elif matches_modality(text, 'x-ray'):
                                    modality_counts['x-ray'] += 1
                                elif matches_modality(text, 'mri'):
                                    modality_counts['mri'] += 1
                                else:
                                    modality_counts['others'] += 1
                            print("Modality counts:")
                            for mod, count in modality_counts.items():
                                print(f"{mod}: {count}")
                        elif self.modality.lower() != "others" and self.modality != '':
                            self.vqa_rad = filter_entries_by_modality(self.vqa_rad, self.modality)
                        elif self.modality.lower() == "others":
                            self.vqa_rad = [entry for entry in self.vqa_rad 
                                            if not any(matches_modality(entry["question"] + ' ' + entry["answer"], mod) 
                                                    for mod in ['ct', 'x-ray', 'mri'])]

                        if self.openclosetype == "close":
                            self.vqa_rad = self.vqa_rad.filter(lambda x: x["answer"].lower() == "yes" or x["answer"].lower() == "no")
                        elif self.openclosetype == "open":
                            self.vqa_rad = self.vqa_rad.filter(lambda x: x["answer"].lower() != "yes" and x["answer"].lower() != "no")
                        elif self.openclosetype == "all":
                            pass
                    else:
                        self.vqa_rad = ds["test"].select(range(each_valid_num))
                else:
                    self.vqa_rad = ds["train"]
                self.length += len(self.vqa_rad)

            elif ds == "vqamed2019":
                self.vqamed2019_dir = Path('/mnt/disk2/hwj/Tails/dataset/VQAMed2019Test')
                if self.test:
                    jpath = self.vqamed2019_dir / 'VQAMed2019_Test_Questions_w_Ref_Answers.txt' 
                elif self.valid:
                    jpath = self.vqamed2019_dir / 'VQAMed2019_Test_Questions_w_Ref_Answers.txt'
                else:
                    jpath = self.vqamed2019_dir / 'VQAMed2019_Test_Questions_w_Ref_Answers.txt' # train not downloaded yet

                self.vqamed2019 = open(jpath, 'r').readlines()
                # Keep only samples whose answers are yes/no.
                # if self.openclosetype == "close":
                #     vqamed2019_yesno = []
                #     for item in self.vqamed2019:
                #         item = item.strip().split('|')
                #         ans = item[3].strip().lower()
                #         if ans in ['yes', 'no']:
                #             vqamed2019_yesno.append(item[0] + '|' + item[1] + '|' + item[2] + '|' + ans)
                #     self.vqamed2019 = vqamed2019_yesno

                self.length += len(self.vqamed2019)

            elif ds == "kvasir_vqa":
                from datasets import load_dataset
                ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")
                self.kvasir_vqa = ds['raw']

                # entry = self.kvasir_vqa[idx]
                # img = entry["image"]
                # img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                # img_path = img
                # q = entry["question"]
                # a = entry["answer"]
                kvasir_vqa = []
                if self.openclosetype == 'close':
                    for i in self.kvasir_vqa:
                        if i["answer"] in ['yes', 'no']:
                            kvasir_vqa.append(i)
                    self.kvasir_vqa = kvasir_vqa
                
                self.length += len(self.kvasir_vqa)
            
            else:
                raise ValueError(f"Unsupported dataset: {ds}")

    def __len__(self) -> int:
        return self.length
    
    def _pick_item(self, ds: str, idx: int) -> Tuple[Dict, Path]:
        """Return the sample dict and corresponding image path for a given data source."""
        cls_label = torch.zeros((1,self.cls_num), dtype=torch.float)  # Multi-label classification target.
        if ds == "PubMedVision":
            data = self.pubmed_data
            entry = data[idx]
            # img_rel = entry["image"][0].replace("images/", "images_resize224/images/")
            if self.image_size == 256:
                img_rel = entry["image"][0].replace("images/", "images256/")
            else:
                img_rel = entry["image"][0].replace("images/", "images_resize224/images/")

            root = self.pubmed_root
            img_path = root / img_rel
            conv_list = entry["conversations"]
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} not found")
            q = conv_list[0]['value']
            a = conv_list[1]['value']

        
        elif ds == "MIMIC":
            if self.image_size == 256:
                root = self.mimic_dir / "MIMIC-256/files"
            elif self.image_size == 1024:
                root = self.mimic_dir / "MIMIC-1024/files"
            else:
                root = self.mimic_dir / "MIMIC-224-inter-area/files"

            row = self.mimic_data.iloc[idx]
            entry = row["report"]
            img_rel = row["path"]
            img_path = root / img_rel

            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} not found")
            question_template = random.choice(GENERATE_QUESTION_LIST)
            q = question_template.format(task_name='')
            a = entry
            # pil_img = Image.open(img_path).convert("RGB")
        elif ds == "cxr-lt":
            if self.image_size == 256:
                root = self.mimic_dir / "MIMIC-256/"
            elif self.image_size == 1024:
                root = self.mimic_dir / "MIMIC-1024/"
            else:
                root = self.mimic_dir / "MIMIC-224-inter-area/"
            row = self.cxr_lt[idx]
            img_rel = row["path"]
            img_path = root / img_rel
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} not found")
            q = row["question"]
            a = row["answer"]
            cls_label = row["cls_label"]
            cls_label = torch.tensor(cls_label, dtype=torch.float).unsqueeze(0)

        elif ds == "isic2018":
            if self.valid and not self.test:
                root = Path('/mnt/disk2/hwj/Tails/dataset/isic/isic2018/ISIC2018_Task3_Validation_Input/')
            elif self.test:
                root = Path('/mnt/disk2/hwj/Tails/dataset/isic/isic2018/ISIC2018_Task3_Test_Input/')
            else:
                root = Path('/mnt/disk2/hwj/Tails/dataset/isic/isic2018/ISIC2018_Task3_Training_Input/')
            row = self.isic2018[idx]
            img_rel = row["path"]
            img_path = root / f"{img_rel}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} not found")
            q = row["question"]
            a = row["answer"]
            cls_label = row["cls_label"]
            cls_label = torch.tensor(cls_label, dtype=torch.float).unsqueeze(0)

        elif ds == "isic2019":
            if self.valid or self.test:
                root = Path('/mnt/disk2/hwj/Tails/dataset/isic/isic2019/ISIC_2019_Test_Input/')
            else:
                root = Path('/mnt/disk2/hwj/Tails/dataset/isic/isic2019/ISIC_2019_Training_Input/')
            row = self.isic2019[idx]
            img_rel = row["path"]
            img_path = root / f"{img_rel}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} not found")
            q = row["question"]
            a = row["answer"]
            # cls_label = row["cls_label"]

            
        elif ds == "PMC_VQA":
            
            row = self.pmc_data.iloc[idx]
            img_rel = row["Figure_path"]

            if self.image_size == 256:
                img_path = self.pmc_dir / "figures256" / img_rel
            else:
                img_path = self.pmc_dir / "figures" / img_rel
            
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} not found")
            q = row["Question"].strip() + row["Choice A"] + row["Choice B"] + row["Choice C"] + row["Choice D"]
            a = row["Answer"]
            if a == "A":
                a = row["Choice A"].strip()
            elif a == "B":
                a = row["Choice B"].strip()
            elif a == "C":
                a = row["Choice C"].strip()
            elif a == "D":
                a = row["Choice D"].strip()
            else:
                raise ValueError(f"Unsupported answer choice: {a} in PMC_VQA dataset")     

        elif ds == "SLAKE":
            root = "/mnt/disk2/hwj/Tails/dataset/SLAKE/imgs/"
            entry = self.slake_data[idx]
            img_path = str(entry["img_name"])
            img_path = Path(root + img_path)
            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} not found")
            q = entry["question"]
            a = entry["answer"]
        
        elif ds == "path-vqa":
            entry = self.path_vqa[idx]
            img = entry["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_path = img
            q = entry["question"]
            a = entry["answer"]
            
        elif ds == "vqa-rad":
            # vqa-rad dataset directly load PIL images
            entry = self.vqa_rad[idx]
            img = entry["image"]
            
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_path = img
            q = entry["question"]
            a = entry["answer"]
            
        elif ds == "vqamed2019":
            dir = '/mnt/disk2/hwj/Tails/dataset/VQAMed2019Test/VQAMed2019_Test_Images/'
            item = self.vqamed2019[idx]
            item = item.strip().split('|')
            img_path = dir + item[0] + '.jpg'
            q = item[2]
            a = item[3]

        elif ds == "kvasir_vqa":
            entry = self.kvasir_vqa[idx]
            img = entry["image"]
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_path = img
            q = entry["question"]
            a = entry["answer"]

        else:  
            print(f"[Warning] Unsupported dataset: {ds}.")

        return q,a, img_path, cls_label

    def _prepare_pil_and_conv(self, q, a, img_path: Path) -> Tuple[Image.Image, List[Dict]]:
        """Load a PIL image and construct the initial conversation list including the image token."""
        if not q.startswith(DEFAULT_IMAGE_TOKEN):
            q = DEFAULT_IMAGE_TOKEN + q
        conv_src = [
            {"from": "human", "value": q},
            {"from": "gpt",   "value": a},
        ]
        # if isinstance(img_path, Image.Image):
        #     # vqa-rad dataset directly load PIL images
        #     pil_img = img_path
        # else:
        #     pil_img = Image.open(img_path).convert("RGB")

        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            if isinstance(img_path, Image.Image):
                pil_img = img_path
            else:
                pil_img = Image.open(img_path).convert("RGB")

            # Check whether a "Truncated File Read" warning was raised.
            for w in wlist:
                if "Truncated File Read" in str(w.message):
                    print(f"[WARNING] Truncated File Read from dataset: {self._current_ds}")

        return pil_img, conv_src
    
    def _full_transform(self, pil_img):
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(pil_img)
        img_np = img_np.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_np).float()
        try:
            seg_in = (img_tensor - self.pixel_mean) / self.pixel_std
        except:
            return 0, 0
        return seg_in, pil_img

    def _process_qwen(self, pil_img: Image.Image, conv_src: List[Dict]) -> Dict:
        """Run the Qwen preprocessing pipeline when conv is None."""
        proc = self.processor.image_processor.preprocess(pil_img, return_tensors="pt")
        pixel_values = [proc["pixel_values"]]
        grid_thw = proc["image_grid_thw"][0]
        
        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [
            merged_thw.prod() // self.processor.image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]

        if self.valid:
            # Validation mode: return raw data for batched processing in collate_fn.
            messages_val = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text",  "text": conv_src[0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}
                    ],
                }
            ]
            
            return [{
                "messages": messages_val,
                "pixel_values": torch.cat(pixel_values, dim=0),
                "image_grid_thw": torch.cat([thw.unsqueeze(0) for thw in grid_thw], dim=0),
                "sft_format": [
                    {"role": "human", "content": conv_src[0]["value"]},
                    {"role": "gpt",   "content": conv_src[1]["value"]}
                ],
            }]
        
        else:
            qwen_conv = [{"from": "human", "value": conv_src[0]["value"]},
                        {"from": "gpt",   "value": conv_src[1]["value"]}]
            data = preprocess_qwen_2_visual(
                [qwen_conv], self.processor.tokenizer,
                grid_thw_image=grid_thw_merged, grid_thw_video=None
            )
            pos_ids, _ = get_rope_index_25(
                self.processor.image_processor.merge_size,
                data["input_ids"],
                image_grid_thw=torch.stack(grid_thw, dim=0),
                video_grid_thw=None,
                second_per_grid_ts=None
            )
            # attention_mask = torch.ones_like(data["input_ids"][0])
            attention_mask = [data["input_ids"][0].size(0)]
            return [{
                "input_ids": data["input_ids"][0],
                "labels":    data["labels"][0],
                "position_ids": pos_ids,
                "pixel_values": torch.cat(pixel_values, dim=0),
                "image_grid_thw": torch.cat([thw.unsqueeze(0) for thw in grid_thw], dim=0),
                "sft_format": qwen_conv,
                "attention_mask": attention_mask,
            }]

    def _process_deepseek(self, img_path, conv_src: List[Dict]) -> Dict:
        """Run the DeepSeek preprocessing pipeline when conv is not None."""            
        conversation = [
            {"role": "user", 
              "content": conv_src[0]["value"], 
              "images": [str(img_path)]},
            {"role": "assistant", "content": ''}
        ]
        if isinstance(img_path, Image.Image):
            # vqa-rad dataset directly load PIL images
            pil_images = [img_path]
        else:
            pil_images = load_pil_images(conversation)
        
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=False,
            inference_mode=self.valid,
        )
        prepare_inputs["sft_format"] = conv_src[1]["value"]
        return [prepare_inputs]
    
    def _data_augmentation(self, pil_img: Image.Image, name) -> Image.Image:
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

        # 1. flips
        # if random.random() < 0.5:
        #     pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.1:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

        # 2. small rotation
        if random.random() < 0.5:
            angle = random.uniform(-12, 12)
            # rotate with expand=False to keep size, fill with black
            pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0))

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

        # 5. random slight crop + resize (preserve medical semantics; crop only 0~8%)
        if random.random() < 0.5:
            max_crop = 0.08
            dw = int(w * random.uniform(0, max_crop))
            dh = int(h * random.uniform(0, max_crop))
            left = random.randint(0, dw)
            top = random.randint(0, dh)
            right = w - (dw - left)
            bottom = h - (dh - top)
            if right - left > 10 and bottom - top > 10:
                pil_img = pil_img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)
        
        if name == "isic2018" or name == "isic2019" or name == "cxr-lt":
            # Flips.
            if random.random() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

        return pil_img
    
    def __getitem__(self, idx: int) -> Tuple:
        
        """Main pipeline: sample a dataset, load a sample, preprocess image and dialogue,
        apply the corresponding processor, and return training data."""
        for attempt in range(10):
            # ds = random.choice(self.datasets) if not self.valid else self.datasets[0]
            ds = self._weighted_sample_dataset()
            self._current_ds = ds
            if ds.startswith("PubMedVision"):
                ds_len = self.pubmed_len
            elif ds == "MIMIC":
                ds_len = self.mimic_len
            elif ds == "SLAKE":
                ds_len = len(self.slake_data)
            elif ds == "PMC_VQA":
                ds_len = len(self.pmc_data)
            elif ds == "vqa-rad":
                ds_len = len(self.vqa_rad)
            elif ds == "path-vqa":
                ds_len = len(self.path_vqa)
            elif ds == "cxr-lt":                 
                ds_len = len(self.cxr_lt)
            elif ds == "isic2018":
                ds_len = len(self.isic2018)
            elif ds == "isic2019":
                ds_len = len(self.isic2019)
            elif ds == "vqamed2019":
                ds_len = len(self.vqamed2019)
            elif ds == "kvasir_vqa":
                ds_len = len(self.kvasir_vqa)
            else:
                ds_len = self.length
            idx_local = idx % ds_len
            
            
            try:
                q, a, img_path, cls_label = self._pick_item(ds, idx_local)
                pil_img, conv_src = self._prepare_pil_and_conv(q, a, img_path)

                seg_in, pil_img = self._full_transform(pil_img)
                if not self.test and not self.valid:
                    pil_img = self._data_augmentation(pil_img, name=ds)
                
                image_label = torch.zeros_like(seg_in)[0]
                tasks = ["vqa"]

                if self.conv is None:
                    # print("Using Qwen processor...")
                    prepare_inputs = self._process_qwen(pil_img, conv_src)
                else:
                    # print("Using Deepseek processor...")
                    prepare_inputs = self._process_deepseek(img_path, conv_src)
                
                promptype = 't'
                bboxs = torch.zeros(1, 4)
                points = torch.zeros(1, 1, 2)
                point_labels = torch.zeros(1, 1, dtype=torch.int32)
                # class_list = ["None"]
                class_list = ["None"]
                # transfer cls_label to tensor
                class_tail_label = cls_label

                
                
                if ds == "vqa-rad":
                    img_path = ["vqa-rad None"]
                elif ds == "path-vqa":
                    img_path = ["path-vqa None"]
                else:
                    img_path = [str(img_path)]
                return (seg_in.unsqueeze(0), image_label.unsqueeze(0),
                        bboxs.unsqueeze(0), points, point_labels,
                        prepare_inputs, img_path, tasks, promptype, class_list, [ds], class_tail_label)

            except FileNotFoundError as e:
                idx = random.randint(0, self.length - 1)
                print(f"[Warning] {e}; retry with idx={idx} (attempt {attempt+1}/10)")

        raise RuntimeError("Failed to load valid sample after multiple attempts.")




# PubMedVision modality_list = ['computed tomography', 'dermoscopy', 'digital photography', 'endoscopy', 'fundus photography', 'infrared reflectance imaging', 
                            # 'magnetic resonance imaging', 'microfluidic device', 'microscopy images', 'optical coherence tomography', 'others', 'ultrasound']
                            
#  pmc-vqa modality_list =         
# ["computed tomography", "others", "magnetic resonance imaging", "microscopy images","ultrasound","PET",d
# "ANGIOGRAPHY",  "PET_CT",  "fundus photography", "optical coherence tomography",  "endoscopy",  
# "infrared reflectance imaging",  "digital photography",  "MAMMOGRAPHY",  "FLUOROSCOPY",  "DSA",  
# "SCINTIGRAPHY",  "PHOTOACOUSTIC",  "PET_MRI",  "ELASTOGRAPHY",  "dermoscopy","x-ray"]
