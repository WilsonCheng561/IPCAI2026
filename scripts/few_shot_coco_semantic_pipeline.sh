#! /bin/bash

## LAUNCH COMMAND EXAMPLE:
## CUDA_VISIBLE_DEVICES=0,1 ./scripts/few_shot_coco_semantic_pipeline.sh 1 2  # split 1, using 2 GPUs

# COCO split to evaluate on. Either 1, 2, 3, or 4
COCO_SPLIT=$1
# GPUS is the number of GPUs to use
GPUS=$2
# SHOTS can be an array of numbers
SHOTS=(1 5)
# SEED is the seed to use for the random number generator
SEED=42
# YAML file to use for the config
YAML_FILE=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L.yaml

# Novel classes are always 20
CATEGORY_NUM=20

for SHOT in "${SHOTS[@]}"; do
    # Echo in red color "We are starting pipeline for $SHOT shot"
    echo -e "\033[31mCOCO SEMANTIC SPLIT $COCO_SPLIT, SHOT $SHOT\033[0m"

    RESULTS_DIR=work_dirs/coco_semantic_${COCO_SPLIT}_seed${SEED}/${SHOT}shot
    # If the results directory does not exist, create it
    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir -p $RESULTS_DIR
    fi
    FILENAME=coco_semantic_few_shot_ann_${SHOT}shot_seed${SEED}_fixed.pkl

    mkdir -p $RESULTS_DIR


    # Generated file will have the format of <out-path>_<n_shot>shot_seed<seed>.pkl
    python no_time_to_train/dataset/few_shot_sampling.py --n-shot $SHOT \
                                                --out-path $RESULTS_DIR/${FILENAME} \
                                                --seed $SEED \
                                                --dataset coco_semantic_split_${COCO_SPLIT} \
                                                --img-dir ./data/coco/train2017 \
                                                --plot

    python run_lightening.py test --config $YAML_FILE \
                                --model.test_mode fill_memory \
                                --out_path $RESULTS_DIR/memory.ckpt \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --model.init_args.dataset_cfgs.fill_memory.memory_pkl $RESULTS_DIR/${FILENAME} \
                                --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
                                --model.init_args.dataset_cfgs.fill_memory.class_split coco_semantic_split_${COCO_SPLIT} \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices $GPUS

    python run_lightening.py test --config $YAML_FILE \
                                --model.test_mode postprocess_memory \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --ckpt_path $RESULTS_DIR/memory.ckpt \
                                --out_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices 1

    python run_lightening.py test --config $YAML_FILE  \
                                --ckpt_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --model.init_args.test_mode test \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --model.init_args.dataset_cfgs.test.class_split coco_semantic_split_${COCO_SPLIT} \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices $GPUS \
                                --coco_semantic_split $COCO_SPLIT \
                                --n_shot $SHOT \
                                --seed $SEED
done