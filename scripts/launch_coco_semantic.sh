#! /bin/bash

# First num is the coco semantic split, second num is the number of GPUs
CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./scripts/few_shot_coco_semantic_pipeline.sh 1 4
CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./scripts/few_shot_coco_semantic_pipeline.sh 2 4
CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./scripts/few_shot_coco_semantic_pipeline.sh 3 4
CUDA_VISIBLE_DEVICES=1,0,5,6 zsh ./scripts/few_shot_coco_semantic_pipeline.sh 4 4