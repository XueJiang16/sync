#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2
CKPT=$3
OUT_DIR=$4
META_FILE=$5
ID_CLS=$6

python3 test_ood_custom_v2.py \
--name test_${METHOD}_${OUT_DATA} \
--in_datadir /mapai/haowenguo/ILSVRC/Data/CLS-LOC/val \
--out_datadir dataset/ood_data/${OUT_DATA} \
--model_path ${CKPT}  \
--batch 256 \
--logdir ${OUT_DIR} \
--score ${METHOD} \
--meta_file ${META_FILE} \
--id_cls ${ID_CLS} \

