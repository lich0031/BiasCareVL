from enum import Enum
try:
    from peft import LoraConfig, get_peft_model
except:
    pass
import numpy as np
import torch
import torch.distributed as dist
import random
import math
import os

from functools import partial

# GLOBAL_SEED = 46
GLOBAL_SEED = 42
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_SEG_Task_TOKEN = "<|seg|>"
DEFAULT_DET_Task_TOKEN = "<|det|>"
DEFAULT_CLS_Task_TOKEN = "<|cls|>"

DEFAULT_SEG_OUT_TOKEN = "<|seg_output|>"

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_PROMPT_TOKEN = "<|prompt|>"
DEFAULT_POINT_TOKEN = "<|point|>"
DEFAULT_PLABEL_TOKEN = "<|plabel|>"
DEFAULT_BBOX_TOKEN = "<|bbox|>"


Disease_name = ["tumor","glioma","neuroblastoma","edema","infarction",
                "aneurysm","infection","effusion","cyst","lesion","necrosis",
                "polyp","reflow"]

Disease_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "{task_name}Please locate the possible disease in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Can you locate the possible disease in this image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Identify the region that shows signs of disease in the image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Is there any disease located in this image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Highlight the areas that might exhibit disease symptoms.",
]

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "{task_name}Can you locate the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Please locate the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}What is {class_name} in this image? Please respond with its location.",
    DEFAULT_IMAGE_TOKEN + "{task_name}What is {class_name} in this image? Please output the location.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Can you locate the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Where is the {class_name} located in this picture?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Please find the {class_name} you can see in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Identify and point out the {class_name} in the image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Locate the {class_name} and show its boundaries.",
]

GENERATE_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "{task_name}Please describe the abnormalities visible in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}What findings can you observe in this image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Summarize the pathological features present in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Provide a brief report of the notable findings in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Describe all visible signs of pathology in the given image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Can you interpret the key findings from this medical image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Explain what abnormalities you can detect in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Generate a diagnostic summary based on this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}What potential issues or irregularities do you notice in this image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Please analyze the image and describe any clinical findings.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Describe any visual indicators of disease in this image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}What does this image reveal about the patient's condition?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Provide a radiological interpretation of the image.",
    DEFAULT_IMAGE_TOKEN + "{task_name}Highlight and explain any unusual features visible in the scan.",
    DEFAULT_IMAGE_TOKEN + "{task_name}What are the notable differences from normal anatomy in this image?",
    DEFAULT_IMAGE_TOKEN + "{task_name}What pathological patterns do you observe in this scan?",
    DEFAULT_IMAGE_TOKEN + "{task_name}Give a detailed description of the image findings relevant to diagnosis.",
]


LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST_SEG = [
    "For {class_name}, the segmentation result is here.",
    "The mask for {class_name} has been computed.",
    "Segmentation for {class_name} is provided here.",
    "Here is the segmentation of {class_name}.",
    "The segmentation mask of {class_name} is saved.",
    "{class_name} has been segmented.",
    "Please find the segmentation of {class_name} here.",
    "The generated segmentation for {class_name} is here.",
]

ANSWER_LIST_DET = [
    "Sure, {class_name}.",
    "{class_name} is present.",
    "Sure, {class_name} is here.",
    "the object {class_name} is detected.",
    "{class_name} has been located.",
    "{class_name} is clearly visible.",
    "I have detected {class_name} in the image.",
    "Sure, {class_name} appears in the image.",
]


def log_parameter_info(model, full_weight_names, lora_except_weight, args, log_dir):
    """
    Logs information about model parameters (trainable full, LoRA, frozen non-LoRA)
    to a text file in the specified log directory.

    Args:
        model: The PyTorch model.
        full_weight_names: A list of substrings identifying parameters to be fully trained.
        args: The command line arguments, used for local_rank and lora_target_modules.
        log_dir: The directory where the log file should be saved.
    """
    if args.local_rank == 0 and args.test == False:
        param_info_filepath = os.path.join(log_dir, "parameter_info.txt")
        try:
            param_info_file = open(param_info_filepath, 'w')
            print(f"Saving parameter info to {param_info_filepath}")
            param_info_file.write(f"Full Weight Names for Full Training: {full_weight_names}\n\n")
            print(f"Saving parameter info to {param_info_filepath}")
            param_info_file.write(f"lora_except_weight: {lora_except_weight}\n\n")
        except IOError as e:
            print(f"Error opening parameter info file: {e}")
            param_info_file = None
    else:
        param_info_file = None

    trainable_full_params = []
    lora_params = []
    frozen_non_lora_params = []
    
    # Get the target modules for LoRA
    lora_targets = args.lora_target_modules.split(',') if args.lora_target_modules else []

    for name, param in model.named_parameters():
        is_trainable_full = False
        is_lora = False

        # 1. Check for full weight names and set trainable
        if any(x in name for x in full_weight_names):
            # This needs to run on all ranks to set requires_grad correctly
            param.requires_grad = True
            is_trainable_full = True
            # Log info only on rank 0
            if param_info_file:
                trainable_full_params.append(f"Trainable Full Param: {name} {param.shape} Requires Grad: {param.requires_grad}")

        # 2. Check for LoRA parameters (parameters added by PEFT)
        if "lora" in name:
            is_lora = True
            # LoRA parameters should already have requires_grad=True from lora_assemble
            # Log info only on rank 0
            if param_info_file:
                lora_params.append(f"LoRA Param: {name} {param.shape} Requires Grad: {param.requires_grad}")

        # 3. Check for frozen non-LoRA parameters, excluding LoRA target modules
        # Ensure this check happens *after* potentially setting requires_grad for full weights
        is_lora_target = any(target in name for target in lora_targets)
        if not is_trainable_full and not is_lora and not param.requires_grad and not is_lora_target:
             # Log info only on rank 0
             if param_info_file:
                 frozen_non_lora_params.append(f"Frozen Non-LoRA Param: {name} {param.shape} Requires Grad: {param.requires_grad}")

    # Write collected info to file on rank 0
    if param_info_file:
        param_info_file.write("--- Trainable Full Parameters ---\n")
        for line in trainable_full_params:
            param_info_file.write(line + "\n")

        param_info_file.write("\n--- LoRA Parameters ---\n")
        for line in lora_params:
            param_info_file.write(line + "\n")

        param_info_file.write("\n--- Frozen Non-LoRA Parameters (excluding LoRA targets) ---\n")
        for line in frozen_non_lora_params:
            param_info_file.write(line + "\n")

        param_info_file.close()
        if args.local_rank == 0 and args.test == False:
            print(f"Finished saving parameter info to {param_info_filepath}")

def gather_all(data_list, device):
    import pickle
    import torch.distributed as dist

    # Serialize the object to bytes, then convert to tensor
    byte_tensor = torch.tensor(list(pickle.dumps(data_list)), dtype=torch.uint8, device=device)
    size = torch.tensor([len(byte_tensor)], device=device)

    sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)

    max_size = max([s.item() for s in sizes])
    padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
    padded[:len(byte_tensor)] = byte_tensor

    gathered = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)

    all_data = []
    for tensor, sz in zip(gathered, sizes):
        raw = tensor.cpu().numpy().tobytes()[:sz.item()]
        all_data.extend(pickle.loads(raw))
    return all_data


def parse_bbox(bbox_str):
    """Convert a bbox string into [x1, y1, x2, y2] format."""
    if bbox_str is None:
        return None
    try:
        coords = [float(x.strip()) for x in bbox_str.split(',')]
        if len(coords) == 4:
            return coords
        return None
    except:
        return None

def calculate_iou(box1, box2):
    """Compute the IoU between two bounding boxes."""
    if box1 is None or box2 is None:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Compute the intersection.
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Compute the union.
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
                
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def image_reverse_normalize(image):
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=image.device).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=image.device).view(-1, 1, 1)
    image = image * pixel_std + pixel_mean
    image /= 255.0
    return image


def shape_input(input_dict, device, precision):
    """Recursively move tensors in a dictionary or list to the specified device and dtype."""
    # Determine the target dtype.
    if precision == "fp16":
        target_dtype = torch.float16
    elif precision == "bf16":
        target_dtype = torch.bfloat16
    else:
        target_dtype = torch.float32

    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            # only cast floating tensors — keep integer ones (e.g. indices) as is
            if v.dtype.is_floating_point:
                input_dict[k] = v.to(device=device, dtype=target_dtype)
            else:
                input_dict[k] = v.to(device=device)
        elif isinstance(v, dict):
            shape_input(v, device, precision)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, torch.Tensor):
                    if item.dtype.is_floating_point:
                        input_dict[k][i] = item.to(device=device, dtype=target_dtype)
                    else:
                        input_dict[k][i] = item.to(device=device)
                elif isinstance(item, dict):
                    shape_input(item, device, precision)
    return input_dict


def lora_assemble(model, except_weight, args):
    def find_linear_layers(model, target_modules, exclude_names):
        return sorted([name for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and all(x not in name for x in exclude_names)
                    and any(x in name for x in target_modules)])
    
    target_modules = args.lora_target_modules.split(",")
    lora_target_modules = find_linear_layers(model, target_modules, except_weight)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0: # Ensure count is not zero before division
            self.avg = self.sum / self.count
        else:
            self.avg = 0 # Or float('nan') or appropriate default
            
    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure self.sum is a scalar float for reduction.
        # If self.sum can be a numpy array (e.g. for multi-metric tracking),
        # this part needs to be handled more carefully, possibly by ensuring
        # it's converted to a tensor of fixed size or reduced differently.
        # For typical loss/metric averaging, sum is usually scalar.
        current_sum_val = float(self.sum) if isinstance(self.sum, np.ndarray) else self.sum
        current_count_val = float(self.count) # Ensure count is float for tensor creation

        total = torch.tensor(
            [current_sum_val, current_count_val], dtype=torch.float32, device=device
        )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        
        if total.shape[0] == 2:
            reduced_sum, reduced_count = total.tolist()
            self.sum = reduced_sum 
            self.count = int(reduced_count) # count should be an integer
        else:
            # This case indicates an unexpected tensor shape after all_reduce
            print(f"Warning: AverageMeter all_reduce received tensor of unexpected shape: {total.shape} on rank {dist.get_rank() if dist.is_initialized() else 'N/A'}. Original sum: {current_sum_val}, count: {current_count_val}")
            # Attempt to gracefully handle or set to a default state if recovery isn't straightforward
            # For now, we'll leave sum and count as they were before this attempt if shape is wrong
            # Or, you might want to set them to an error indicator or re-raise

        if self.count > 0: # Avoid division by zero
            self.avg = self.sum / self.count
        else:
            self.avg = 0.0 # Or float('nan') or handle as appropriate


    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def seed_worker(worker_id):
    worker_seed = GLOBAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def compute_flops_with_profiler(model, sample_batch, args):
    from torch.profiler import profile, ProfilerActivity
    """
    Use torch.profiler to estimate FLOPs for a single forward pass.
    Fix: if sample_batch and model parameters use different floating-point dtypes
    (for example, bf16 inputs with fp32 weights), execution may fail.
          'Input type (c10::BFloat16) and bias type (float) should be the same'。
          All floating-point input tensors are automatically cast to the model's
          main parameter dtype here.
    Notes:
      1) Only a single forward pass is profiled; backward is not included.
      2) Training-step FLOPs can be approximated as roughly *3
         (forward + backward + optimizer).
    """
    if profile is None:
        if is_main_process():
            print("[FLOPs] torch.profiler is unavailable; skipping.")
        return None

    # Infer the model's main dtype from the first floating-point parameter.
    try:
        param_dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    except StopIteration:
        param_dtype = torch.float32

    # First move inputs to the device following the original logic,
    # then force-cast them to param_dtype.
    sample_batch = shape_input(sample_batch, args.device, args.precision)

    def _align_dtype(d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                if v.dtype != param_dtype:
                    d[k] = v.to(dtype=param_dtype)
        return d

    sample_batch = _align_dtype(sample_batch)

    # If the model itself is not already in param_dtype (rare), cast it as well.
    if any(p.is_floating_point() and p.dtype != param_dtype for p in model.parameters()):
        model.to(param_dtype)

    model_was_training = model.training
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_flops=True, record_shapes=False) as prof:
            _ = model(**sample_batch)

    total_flops = 0
    for evt in prof.key_averages():
        if hasattr(evt, "flops") and evt.flops is not None:
            total_flops += evt.flops

    if is_main_process():
        forward_tflops = total_flops / 1e12
        print(f"[FLOPs] Forward: {forward_tflops:.3f} TFLOPs")
        print(f"[FLOPs] Approx train step ≈ {forward_tflops*3:.3f} TFLOPs (x3 heuristic)")

    if model_was_training:
        model.train()
    return total_flops
    
