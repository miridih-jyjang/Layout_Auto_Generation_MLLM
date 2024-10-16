#!/bin/bash
num_gpu=1
ckpt_name=llava_v1.5_7b_miridih_v6.4_1e_append
MASTER_ADDR=127.0.0.1
MASTER_PORT=29510
# ckpt_name=pretrained

# Array of JSON files
# json_files=("data/miridih-max25-v4/annotations/val_llava_c2ps.json"
#             "data/miridih-max25-v4/annotations/val_llava_cp2s.json"
#             "data/miridih-max25-v4/annotations/val_llava_cs2p.json"
#             "data/miridih-max25-v4/annotations/val_llava_random.json"
#             "data/miridih-max25-v4/annotations/val_llava_refine.json"
#             "data/miridih-max25-v4/annotations/val_llava_complete.json")
#json_files=("/workspace/data/miridih-v6.4/annotations/val_coord_pred.json")
# json_files=("/workspace/data/miridih-v6.4/annotations/val_complete.json")
json_files=("/workspace/data/scenarios/annotations/testA_cp2s.json")
# Output directory
#output_dir="output/$ckpt_name"
output_dir=/data/checkpoints/jjy/llava_v1.5_7b_miridih_v6.4_1e_append/output_temp0.2_samp
# Make sure the output directory exists
mkdir -p $output_dir

for i in "${!json_files[@]}"; do
    # Extract the base name of the JSON file (without the directory and extension)
    base_name=$(basename ${json_files[$i]} .json)

    # Define the corresponding output file name
    output_file="$output_dir/${base_name}_output.json"

    # Calculate the GPU index (e.g., mod the loop index with the number of available GPUs)
    #gpu_index=$((i % 8))  # Assuming you have 8 GPUs (0, 1, 2, 3, 4, 5, 6, 7)
    gpu_index=$(((i % 5) + 0))
    # Run the command with the dynamically set GPU index
    CUDA_VISIBLE_DEVICES=$gpu_index torchrun --nproc_per_node=$num_gpu --master_addr $MASTER_ADDR --master_port $MASTER_PORT  miridih_llava/serve/cli_multi_v6_4_scenario_A.py \
    --model-path /data/checkpoints/jjy/$ckpt_name \
    --json-file ${json_files[$i]} \
    --max-new-tokens 4096 \
    --temperature 0.2 \
    --ele_cache_path ./train_element_clip_features_miridih.json \
    --output-file $output_file \
    --num-gpus $num_gpu --data-path /workspace/data \
    --image-out 

done
