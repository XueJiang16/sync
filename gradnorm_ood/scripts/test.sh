#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2

python3 test_ood.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir /mapai/haowenguo/ILSVRC/Data/CLS-LOC/val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_path /mapai/haowenguo/code/SPL/jx/big_transfer/BiT-S-R101x1-LT_a4_2022-01-25_114355/bit.pth.tar  \
--batch 256 \
--logdir checkpoints/bit_LT_a4_log \
--score ${METHOD}