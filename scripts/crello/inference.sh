#!/bin/bash
ckpt_name=llava_v1.5_7b_crello_v4_1e
# ckpt_name=pretrained

# Array of JSON files
# json_files=("/data/checkpoints/jjy/$ckpt_name/val_c2ps.json"
#              "/data/checkpoints/jjy/$ckpt_name/val_cp2s.json"
#              "/data/checkpoints/jjy/$ckpt_name/val_cs2p.json"
#              "/data/checkpoints/jjy/$ckpt_name/val_random.json"
#              "/data/checkpoints/jjy/$ckpt_name/val_refine.json"
#              "/data/checkpoints/jjy/$ckpt_name/val_complete.json")
json_files=("/data/checkpoints/jjy/$ckpt_name/val_refine.json")
# Output directory
output_dir="output/$ckpt_name"

# Make sure the output directory exists
mkdir -p $output_dir

for i in "${!json_files[@]}"; do
    # Extract the base name of the JSON file (without the directory and extension)
    base_name=$(basename ${json_files[$i]} .json)

    # Define the corresponding output file name
    output_file="$output_dir/${base_name}_out.json"

    # Calculate the GPU index (e.g., mod the loop index with the number of available GPUs)
    #gpu_index=$((i % 8))  # Assuming you have 8 GPUs (0, 1, 2, 3, 4, 5, 6, 7)
    gpu_index=$(((i % 2) + 6))

    # Run the command with the dynamically set GPU index
    CUDA_VISIBLE_DEVICES=$gpu_index python miridih_llava/serve/cli_multi_v4_crello.py \
    --model-path /data/checkpoints/jjy/$ckpt_name \
    --json-file ${json_files[$i]} \
    --output-file $output_file \
    --num-gpus 1 --data-path ./data \
    --image-out &

done
