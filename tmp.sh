deepspeed --include localhost:0,1,6,7 --master_port=25001 run.py \
  --version weight/HuatuoGPT-Vision-7B-Qwen2.5VL \
  --vision_pretrained weight/sam_vit_b_01ec64.pth \
  --dataset_dir /mnt/disk2/hwj/Tails/dataset \
  --dataset "refer_seg" \
  --refer_seg_data "IMed361M" \
  --val_dataset "refer_seg" \
  --sample_rates "1" \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --image_size 256 \
  --lr 1e-5 \
  --lr_vl 5e-6 \
  --lora_r 8 \
  --lora_alpha 16 \
  --exp_name temp

deepspeed --include localhost:0,1,6,7 --master_port=25001 run.py \
  --vision_pretrained /mnt/disk2/hwj/Tails/weight/IMISNet-B.pth \
  --dataset_dir /mnt/disk2/hwj/Tails/dataset \
  --dataset "refer_seg" \
  --refer_seg_data "IMed361M" \
  --val_dataset "refer_seg" \
  --sample_rates "1" \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --image_size 256 \
  --lr 1e-5 \
  --lr_vl 5e-6 \
  --lora_r 8 \
  --lora_alpha 16 \
  --exp_name pretrainseg

# 第二阶段训练 成功
deepspeed --include localhost:0,1,6,7 --master_port=25002 run.py \
  --version weight/HuatuoGPT-Vision-7B-Qwen2.5VL \
  --vision_pretrained /mnt/disk2/hwj/Tails/Chat2Tail/runs/pretrainseg4_bp_224_1e-05_BG62_RA816_lmw1.0_sr1_mpT/hf_model/pytorch_model.bin \
  --dataset_dir /mnt/disk2/hwj/Tails/dataset \
  --dataset "refer_seg||vqa" \
  --refer_seg_data "IMed361M" \
  --vqa_data "PubMedVision||MIMIC||SLAKE||PMC_VQA||vqa-rad||path-vqa" \
  --val_dataset "refer_seg||vqa" \
  --sample_rates "1,1" \
  --batch_size 4 \
  --grad_accumulation_steps 2 \
  --image_size 256 \
  --lr 1e-5 \
  --lr_vl 5e-6 \
  --epochs 5 \
  --lora_r 8 \
  --lora_alpha 16 \
  --exp_name temp_ \
  --seed 58


deepspeed --include localhost:2,3,4,5 --master_port=24002 runs.py \
--epochs 3 \
--divided_save 0.025 \
--batch_size 4 \
--lm_weight 1.0 \
--lr 1e-5 \
--lr_vl 5e-6 \
--sample_rates "1,1" \
--promptype tbp \
--exp_name NMI2 \
--image_size 256 \
--test \
--val_dataset refer_seg \
--val_batch_size 1
