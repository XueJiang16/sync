#!/usr/bin/env bash
CKPT=$1
OUT_DIR=$2
ID_FILE=$3

python3 tune_mahalanobis_hyperparameter.py \
--name tune_mahalanobis \
--model_path ${CKPT} \
--logdir ${OUT_DIR} \
--datadir /mapai/haowenguo/ILSVRC/Data/CLS-LOC/train \
--train_list ${ID_FILE} \
--val_list /mapai/haowenguo/ILSVRC/Data/CLS-LOC/meta/val_labeled.txt \
--batch 32
