#!/bin/bash
ckpt_name=llava_v1.5_7b_crello_v6.0_1e
num_gpu=1
# ckpt_name=pretrained

# Array of JSON files
#json_files=("/workspace/data/crello-v6/annotations/val_coord_pred.json"
#            "/workspace/data/crello-v6/annotations/val_random.json"
#            "/workspace/data/crello-v6/annotations/val_c2ps.json"
#             "/workspace/data/crello-v6/annotations/val_cp2s.json"
#             "/workspace/data/crello-v6/annotations/val_cs2p.json"
#             "/workspace/data/crello-v6/annotations/val_refine.json"
#             "/workspace/data/crello-v6/annotations/val_complete.json")
json_files=("/workspace/data/crello-v6/annotations/val_coord_pred.json")
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
    #gpu_index=$(((i % 8) + 3))
    gpu_index=3
    # Run the command with the dynamically set GPU index
    CUDA_VISIBLE_DEVICES=$gpu_index torchrun --nproc_per_node $num_gpu miridih_llava/serve/cli_multi_v6_crello.py \
    --model-path /data/checkpoints/jjy/$ckpt_name \
    --json-file ${json_files[$i]} \
    --output-file $output_file \
    --max-new-tokens 4096 \
    --num-gpus $num_gpu --data-path /workspace/data #\
 #   --image-out 

done
