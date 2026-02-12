#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --expname primary_model_gaussian_sgci_film \
    --model_type muq_roberta_transformer_dist \
    --datadir /share/nas169/jethrowang/MusicEval-full \
    --train_list_path /share/nas169/jethrowang/MusicEval-full/sets/train_mos_list.txt \
    --validation_list_path /share/nas169/jethrowang/MusicEval-full/sets/dev_mos_list.txt \
    --test_list_path /share/nas169/jethrowang/MusicEval-full/sets/test_mos_list.txt \
    --batch_size 32 \
    --lr 5e-5 \
    --optimizer adamw \
    --dist_prediction_score_style gaussian \
    --num_bins 20 \
    --use_sgci