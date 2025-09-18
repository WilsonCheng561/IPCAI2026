#!/usr/bin/env bash

CONFIG=$1
SHOTS=$2
GPUS=$3
SEED=$4

CLASS_SPLIT="few_shot_classes"
RESULTS_DIR=work_dirs/few_shot_results
mkdir -p $RESULTS_DIR

FILENAME=few_shot_ann_${SHOTS}shot_seed${SEED}_fixed.pkl
echo "Generating few-shot annotation file with $SHOTS shots and seed $SEED"

# Generated file will have the format of <out-path>_<n_shot>shot_seed<seed>.pkl
# python no_time_to_train/dataset/few_shot_sampling.py --n-shot $SHOTS --out-path ${RESULTS_DIR}/${FILENAME} --seed $SEED --dataset $CLASS_SPLIT

# python run_lightening.py test --config $CONFIG \
#                               --model.test_mode fill_memory \
#                               --out_path ${RESULTS_DIR}/memory.ckpt \
#                               --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
#                               --model.init_args.dataset_cfgs.fill_memory.memory_pkl ${RESULTS_DIR}/${FILENAME} \
#                               --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOTS \
#                               --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
#                               --trainer.logger.save_dir ${RESULTS_DIR}/ \
#                               --trainer.devices $GPUS

# python run_lightening.py test --config $CONFIG \
#                               --model.test_mode postprocess_memory \
#                               --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
#                               --ckpt_path ${RESULTS_DIR}/memory.ckpt \
#                               --out_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
#                               --trainer.devices 1

python run_lightening.py test --config $CONFIG  \
                              --ckpt_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                              --model.init_args.test_mode test \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.dataset_cfgs.test.class_split $CLASS_SPLIT \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS