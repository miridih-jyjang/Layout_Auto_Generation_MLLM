#!/bin/bash
ckpt_name=llava_v1.5_7b_miridih_v4_1e
# ckpt_name=pretrained

# Array of JSON files
json_files=("data/miridih-max25-v4/annotations/val_llava_c2ps.json"
            "data/miridih-max25-v4/annotations/val_llava_cp2s.json"
            "data/miridih-max25-v4/annotations/val_llava_cs2p.json"
            "data/miridih-max25-v4/annotations/val_llava_random.json"
            "data/miridih-max25-v4/annotations/val_llava_refine.json"
            "data/miridih-max25-v4/annotations/val_llava_complete.json")
#json_files=("data/miridih-max25-v3/annotations/val_llava_c2ps.json")
# Output directory
output_dir="output/$ckpt_name"

# Make sure the output directory exists
mkdir -p $output_dir

for i in "${!json_files[@]}"; do
    # Extract the base name of the JSON file (without the directory and extension)
    base_name=$(basename ${json_files[$i]} .json)

    # Define the corresponding output file name
    output_file="$output_dir/${base_name}_output.json"

    # Calculate the GPU index (e.g., mod the loop index with the number of available GPUs)
    #gpu_index=$((i % 8))  # Assuming you have 8 GPUs (0, 1, 2, 3, 4, 5, 6, 7)
    gpu_index=$(((i % 5) + 2))

    # Run the command with the dynamically set GPU index
    CUDA_VISIBLE_DEVICES=$gpu_index python miridih_llava/serve/cli_multi_v4.py \
    --model-path /data/checkpoints/jjy/$ckpt_name \
    --json-file ${json_files[$i]} \
    --output-file $output_file \
    --num-gpus 1 --data-path ./data \
    --image-out &

done
