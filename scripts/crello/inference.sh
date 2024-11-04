#!/bin/bash
ckpt_name=llava_v1.5_7b_crello_v6.7_2e
num_gpu=1
id=os4xtd41
MASTER_PORT_NUM=29501
# ckpt_name=pretrained

# Array of JSON files
 json_files=("/workspace/data/crello-v6.7/annotations_1/val_coord_pred.json"
            "/workspace/data/crello-v6.7/annotations_1/val_random.json"
             "/workspace/data/crello-v6.7/annotations_1/val_cp2s.json"
             "/workspace/data/crello-v6.7/annotations_1/val_cs2p.json"
             "/workspace/data/crello-v6.7/annotations_1/val_refine.json"
             "/workspace/data/crello-v6.7/annotations_1/val_complete.json")
#json_files=("/workspace/data/crello-v6.7/annotations_1/val_coord_pred.json")
# Output directory
output_dir="/data/checkpoints/jjy/$ckpt_name"

# Make sure the output directory exists
mkdir -p $output_dir

for i in "${!json_files[@]}"; do
    # Extract the base name of the JSON file (without the directory and extension)
    base_name=$(basename ${json_files[$i]} .json)

    # Define the corresponding output file name
    output_file="$output_dir/${base_name}_out.json"

    # Calculate the GPU index (e.g., mod the loop index with the number of available GPUs)
    #gpu_index=$((i % 8))  # Assuming you have 8 GPUs (0, 1, 2, 3, 4, 5, 6, 7)
    gpu_index=$(((i % 2)))
    # gpu_index=3
    # Run the command with the dynamically set GPU index
    CUDA_VISIBLE_DEVICES=$gpu_index  torchrun --nproc_per_node $num_gpu --master_port $MASTER_PORT_NUM miridih_llava/serve/cli_multi_v6_7_crello.py \
    --model-path /data/checkpoints/jjy/$ckpt_name \
    --json-file ${json_files[$i]} \
    --output-file $output_file \
    --exp_name ${ckpt_name} \
    --id ${id} \
    --max-new-tokens 4096 \
    --ele_cache_path ./eval_element_clip_features.json \
    --num-gpus $num_gpu --data-path /workspace/data \
    --image-out &

    MASTER_PORT_NUM=$((MASTER_PORT_NUM + 1))

done
