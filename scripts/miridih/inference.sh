ckpt_name=llava_v1.5_7b_miridih_v3_1e
# ckpt_name=pretrained
CUDA_VISIBLE_DEVICES=4 python miridih_llava/serve/cli_multi.py \
--model-path /data/checkpoints/jjy/$ckpt_name \
--json-file data/miridih-max25-v3/annotations/val_llava_numerical.json \
--output-file output/$ckpt_name/output.json \
--num-gpus 1 --data-path ./data --debug
