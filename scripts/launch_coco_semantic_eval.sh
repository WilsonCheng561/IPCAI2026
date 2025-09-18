#! /bin/bash

SPLITS=(1 2 3 4)
SHOTS=(1 5)
SEED=99

for SPLIT in ${SPLITS[@]}; do
    for SHOT in ${SHOTS[@]}; do
        echo " ==> Evaluating split ${SPLIT} shot ${SHOT} seed ${SEED}"
        python3 no_time_to_train/dataset/coco_inst_to_segm.py \
            --pred_json inst_to_segm/coco_inst_semantic_split_${SPLIT}_${SHOT}shot_${SEED}seed_results.json \
            --class_split coco_semantic_split_${SPLIT}
    done
done