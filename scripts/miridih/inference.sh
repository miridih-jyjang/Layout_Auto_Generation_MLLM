ckpt_name=llava_v1.5_7b_miridih_v1
# ckpt_name=pretrained
CUDA_VISIBLE_DEVICES=1 python llava/serve/cli_multi.py \
--model-path checkpoints/llava_v1.5_7b_miridih \
--json-file data/miridih/annotations/test_v1.json \
--output-file output/${ckpt_name}_output_val.json \
--num-gpus 1 --data-path ./data/
