#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='1464_sup'
ROOT=../../../..

machine_name=YOUR_MACHINE_NAME

CUDA_VISIBLE_DEVICES=0 python $ROOT/eval.py \
    --config=config.yaml \
    --base_size 662 \
    --scales 1.0 \
    --model_path=best.ckpt \
    --machine_name ${machine_name}