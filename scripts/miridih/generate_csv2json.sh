#!/bin/bash
DATASET_NAME="miridih_v6.6"
DATA_PATH="/workspace/data/miridih-v6.6/"
SAVE_PATH="/workspace/data/miridih-v6.6/html_format"
MAX_ELE_NUM=50
MIN_ELE_NUM=1
MODEL_PATH_TO_NAME="codellama/CodeLlama-7b-hf"
BBOX_QUANTIZATION="code"
CONSISTENCY_NUM=1

# Step 1. generate csv2json (layoutnuwa)
#python convertHTML/build_code_v6.6.py --dataset_name ${DATASET_NAME} --dataset_path ${DATA_PATH} \
#        --save_path ${SAVE_PATH} --max_ele_num ${MAX_ELE_NUM} --min_ele_num ${MIN_ELE_NUM} --model_path_or_name ${MODEL_PATH_TO_NAME} \
#        --bbox_quantization ${BBOX_QUANTIZATION} --consistency_num ${CONSISTENCY_NUM} --add_task_instruction

JSON_PATH="/workspace/data/miridih-v6.6/html_format"
TRAIN_IMAGE_ROOT="/workspace/data/miridih-v6.6/images"
VAL_IMAGE_ROOT="/workspace/data/miridih-v6.6/images"
OUTPUT_DIR="/workspace/data/ca_squad/annotations"

# Step 2. generation json2json (posterllava)

python data/get_promptv5-6.py --json-path ${JSON_PATH} --train-image-root ${TRAIN_IMAGE_ROOT} --val-image-root ${VAL_IMAGE_ROOT} --output-dir ${OUTPUT_DIR}
