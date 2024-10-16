#!/bin/bash
MASTER_PORT=29500
PROMPT_VERSION=v1
MODEL_VERSION="llava-v1.5-7b"
data_version="v6.4"
exp_name=llava_v1.5_7b_miridih_v6.4_1e_append
#exp_name=debug
batch_size_per_device=16
acc_step=4
samples_per_epoch=72071
num_tasks=6
num_gpu=4

# Ensure variables are properly assigned and do not contain zero values
if [ $batch_size_per_device -eq 0 ] || [ $acc_step -eq 0 ] || [ $num_gpu -eq 0 ]; then
  echo "Error: batch_size_per_device, acc_step, or num_gpu cannot be zero."
  exit 1
fi

deepspeed --include="localhost:0,1,2,3" --master_port=$MASTER_PORT miridih_llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload_v5.json \
    --model_name_or_path liuhaotian/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --ele_cache_path ./train_element_clip_features_miridih.json \
    --eval_ele_cache_path ./eval_element_clip_features_miridih.json \
    --data_path /workspace/data/miridih-v6.4/annotations/train_llava_numerical.json \
    --dev_data_path /workspace/data/miridih-v6.4/annotations/val_llava_numerical.json \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size $batch_size_per_device \
    --per_device_eval_batch_size $batch_size_per_device \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 100 \
    --eval_steps 200 \
    --save_total_limit 2 \
    --data_version $data_version \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --exp_name $exp_name
