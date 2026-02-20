#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --expname primary_model_gaussian_listwise_mi_lambda_0.01_decoupled_contrastive_ta_lambda_0.01 \
    --model_type muq_roberta_transformer_dist \
    --datadir /share/nas169/jethrowang/MusicEval-full \
    --train_list_path /share/nas169/jethrowang/MusicEval-full/sets/train_mos_list.txt \
    --validation_list_path /share/nas169/jethrowang/MusicEval-full/sets/dev_mos_list.txt \
    --test_list_path /share/nas169/jethrowang/MusicEval-full/sets/test_mos_list.txt \
    --batch_size 24 \
    --valid_batch_size 24 \
    --lr 5e-5 \
    --optimizer adamw \
    --dist_prediction_score_style gaussian \
    --num_bins 20 \
    --ranking_loss_type listwise \
    --ranking_lambda 0.01 \
    --ranking_warmup_epochs 0 \
    --pairwise_margin 0.0 \
    --pairwise_tolerance 0.25 \
    --listwise_temperature 1.0 \
    --contrastive_lambda 0.01 \
    --contrastive_temperature 0.1
