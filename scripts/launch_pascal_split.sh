#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,4,5,6 ./scripts/pascal_split.sh 1 5
CUDA_VISIBLE_DEVICES=0,1,4,5,6 ./scripts/pascal_split.sh 2 5
CUDA_VISIBLE_DEVICES=0,1,4,5,6 ./scripts/pascal_split.sh 3 5