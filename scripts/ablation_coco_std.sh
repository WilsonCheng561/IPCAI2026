#!/bin/bash

# VERSION 1: Run ablation on the results std for COCO few-shot results

# SEEDS=(42 13 27 36 88 33 69 55 77 99)
# SHOTS=(1 2 3 5 10 30)
# CONFIG=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L.yaml
# GPUS=4

# for SEED in "${SEEDS[@]}"; do
#     for SHOT in "${SHOTS[@]}"; do
#         echo "=====> Running few-shot pipeline for $SHOT shot with seed $SEED"
#         CUDA_VISIBLE_DEVICES=0,1,2,3 zsh ./few_shot_full_pipeline.sh $CONFIG $SHOT $GPUS $SEED
#     done
# done



# VERSION 2: Fix missing runs

# SEEDS=(27 36 88)
# SHOTS=(1 2 3 5 10 30)
# CONFIG=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L.yaml
# GPUS=4

# for SEED in "${SEEDS[@]}"; do
#     for SHOT in "${SHOTS[@]}"; do
#         echo "=====> Running few-shot pipeline for $SHOT shot with seed $SEED"
#         CUDA_VISIBLE_DEVICES=0,1,2,3 zsh ./few_shot_full_pipeline.sh $CONFIG $SHOT $GPUS $SEED
#     done
# done

# # Plus two custom runs that got screwed up
# CUDA_VISIBLE_DEVICES=0,1,2,3 zsh ./few_shot_full_pipeline.sh $CONFIG 1 $GPUS 33
# CUDA_VISIBLE_DEVICES=0,1,2,3 zsh ./few_shot_full_pipeline.sh $CONFIG 1 $GPUS 99


# VERSION 3: Run best seed setting for 10 shot,
# for saving json file and convert/compare to semantic segmentation results

CONFIG=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L.yaml
GPUS=4
# CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./few_shot_full_pipeline.sh $CONFIG 10 $GPUS 33
CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./few_shot_full_pipeline.sh $CONFIG 30 $GPUS 33


# CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./few_shot_full_pipeline.sh $CONFIG 1 $GPUS 33
# CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./few_shot_full_pipeline.sh $CONFIG 5 $GPUS 33