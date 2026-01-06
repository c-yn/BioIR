#!/usr/bin/env bash
CONFIG=$1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=4322 basicsr/train.py -opt $CONFIG --launcher pytorch






