#!/bin/bash
MASTER_PORT=29500
PROMPT_VERSION=v1
MODEL_VERSION="llava-v1.5-7b"
exp_name=llava_v1.5_7b_crello_v5.2_1e
#exp_name=debug
epoch=1
batch_size_per_device=1
acc_step=1
data_version="v5.2"
num_gpu=1
samples_per_epoch=18768
num_tasks=6
 Ensure variables are properly assigned and do not contain zero values
if [ $batch_size_per_device -eq 0 ] || [ $acc_step -eq 0 ] || [ $num_gpu -eq 0 ]; then
  echo "Error: batch_size_per_device, acc_step, or num_gpu cannot be zero."
  exit 1
fi

# Calculate max_steps as an integer
max_steps=$((samples_per_epoch / (batch_size_per_device * acc_step * num_gpu) * epoch * num_tasks))

# Check if max_steps was calculated correctly
if [ -z "$max_steps" ] || [ "$max_steps" -eq 0 ]; then
  echo "Error: max_steps calculated as zero or not valid."
  exit 1
fi

CUDA_VISIBLE_DEVICES=5 python miridih_llava/eval/eval_mem.py \
    --model_name_or_path liuhaotian/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /workspace/data/crello-v5.2/annotations/train_llava_numerical.json \
    --dev_data_path /workspace/data/crello-v5.2/annotations/val_coord_pred.json \
    --image_folder /workspace/data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /data/checkpoints/hugging_face/$MODEL_VERSION/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /data/checkpoints/jjy/$exp_name \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $batch_size_per_device \
    --per_device_eval_batch_size $batch_size_per_device \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --data_version $data_version \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --max_steps $max_steps \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --exp_name $exp_name
