#! /bin/bash

# Parse arguments
DATASET_NAME=$1
SHOT=$2
DEVICES=$3
CAT_NAMES=$4
CATEGORY_NUM=$5

# Additional settings
NUM_DEVICES=$((1+${#DEVICES//[^,]/})) # Count the number of commas in DEVICES
ALL_DATASETS_PATH=/localdisk/data2/Users/s2254242/datasets
DATASET_PATH=$ALL_DATASETS_PATH/$DATASET_NAME
YAML_PATH=no_time_to_train/pl_configs/matching_cdfsod_template.yaml
PATH_TO_SAVE_CKPTS=./tmp_ckpts/cd_fsod/matching
mkdir -p $PATH_TO_SAVE_CKPTS
FIRST_DEVICE=${DEVICES%%,*}

# FIFTH STEP: Testing on the target set. With visualisation enabled in the .py file
# --------------------
echo -e "\033[31mEVALUATING $SHOT SHOT FOR DATASET $DATASET_NAME\033[0m"
CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
    --model.test_mode test \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --model.init_args.dataset_cfgs.test.root $DATASET_PATH/test \
    --model.init_args.dataset_cfgs.test.json_file $DATASET_PATH/annotations/test\_with_segm.json \
    --model.init_args.dataset_cfgs.test.cat_names $CAT_NAMES \
    --trainer.devices $NUM_DEVICES
echo "Fifth step done: Testing on the target set"