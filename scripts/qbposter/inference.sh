ckpt_name=posterllava_v0
CUDA_VISIBLE_DEVICES=0 python llava/serve/cli_multi.py \
--model-path pretrained_model/posterllava_v0 \
--json-file data/miridih/posterllava_small.json \
--output-file output/${ckpt_name}_output_val.json \
--num-gpus 1 --data-path ./data/ \
--debug
