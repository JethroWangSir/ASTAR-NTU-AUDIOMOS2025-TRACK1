#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --expname primary_model_beta \
    --model_type muq_roberta_transformer_dist \
    --datadir /share/nas169/jethrowang/MusicEval-full \
    --train_list_path /share/nas169/jethrowang/MusicEval-full/person_mos/train_person_mos.txt \
    --validation_list_path /share/nas169/jethrowang/MusicEval-full/person_mos/dev_person_mos.txt \
    --test_list_path /share/nas169/jethrowang/MusicEval-full/person_mos/test_person_mos.txt \
    --batch_size 32 \
    --valid_batch_size 32 \
    --lr 5e-5 \
    --optimizer adamw \
    --dist_prediction_score_style beta \
    --num_bins 20