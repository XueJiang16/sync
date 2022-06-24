#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2
CKPT=$3
OUT_DIR=$4
ID_CLS=$5

python3 test_ood_custom.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir /mapai/haowenguo/ILSVRC/Data/CLS-LOC/val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_path ${CKPT}  \
--batch 256 \
--logdir ${OUT_DIR} \
--score ${METHOD} \
--id_cls ${ID_CLS}
