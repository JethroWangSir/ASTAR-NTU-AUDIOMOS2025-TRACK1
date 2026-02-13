#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --expname primary_model_gaussian_qamro_lambda_0.01 \
    --model_type muq_roberta_transformer_dist \
    --datadir /share/nas169/jethrowang/MusicEval-full \
    --train_list_path /share/nas169/jethrowang/MusicEval-full/sets/train_mos_list.txt \
    --validation_list_path /share/nas169/jethrowang/MusicEval-full/sets/dev_mos_list.txt \
    --test_list_path /share/nas169/jethrowang/MusicEval-full/sets/test_mos_list.txt \
    --batch_size 32 \
    --valid_batch_size 32 \
    --lr 5e-5 \
    --optimizer adamw \
    --dist_prediction_score_style gaussian \
    --num_bins 20 \
    --use_ranking_loss \
    --ranking_loss_type qamro \
    --rank_lambda 0.01 \
    --qamro_preference_factor 7.0 \
    --qamro_margin_scale 0.2

python train.py \
    --expname primary_model_gaussian_qamro_lambda_0.05 \
    --model_type muq_roberta_transformer_dist \
    --datadir /share/nas169/jethrowang/MusicEval-full \
    --train_list_path /share/nas169/jethrowang/MusicEval-full/sets/train_mos_list.txt \
    --validation_list_path /share/nas169/jethrowang/MusicEval-full/sets/dev_mos_list.txt \
    --test_list_path /share/nas169/jethrowang/MusicEval-full/sets/test_mos_list.txt \
    --batch_size 32 \
    --valid_batch_size 32 \
    --lr 5e-5 \
    --optimizer adamw \
    --dist_prediction_score_style gaussian \
    --num_bins 20 \
    --use_ranking_loss \
    --ranking_loss_type qamro \
    --rank_lambda 0.05 \
    --qamro_preference_factor 7.0 \
    --qamro_margin_scale 0.2

python train.py \
    --expname primary_model_gaussian_qamro_lambda_0.1 \
    --model_type muq_roberta_transformer_dist \
    --datadir /share/nas169/jethrowang/MusicEval-full \
    --train_list_path /share/nas169/jethrowang/MusicEval-full/sets/train_mos_list.txt \
    --validation_list_path /share/nas169/jethrowang/MusicEval-full/sets/dev_mos_list.txt \
    --test_list_path /share/nas169/jethrowang/MusicEval-full/sets/test_mos_list.txt \
    --batch_size 32 \
    --valid_batch_size 32 \
    --lr 5e-5 \
    --optimizer adamw \
    --dist_prediction_score_style gaussian \
    --num_bins 20 \
    --use_ranking_loss \
    --ranking_loss_type qamro \
    --rank_lambda 0.1 \
    --qamro_preference_factor 7.0 \
    --qamro_margin_scale 0.2