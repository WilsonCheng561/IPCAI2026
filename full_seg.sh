#!/usr/bin/env bash

# Launch command 
# 10 shot, 4 GPUs, seed 33
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./full_seg.sh ./no_time_to_train/new_exps/coco_allclasses_10shot_Sam2L.yaml 10 4 33

CONFIG=$1
SHOTS=$2
GPUS=$3
SEED=$4

CLASS_SPLIT="default_classes"
RESULTS_DIR=work_dirs/coco_all_results
mkdir -p $RESULTS_DIR

FILENAME=coco_all_${SHOTS}shot_seed${SEED}_fixed.pkl
echo "Generating few-shot annotation file with $SHOTS shots and seed $SEED"

# Generated file will have the format of <out-path>_<n_shot>shot_seed<seed>.pkl
# python no_time_to_train/dataset/few_shot_sampling.py --n-shot $SHOTS --out-path ${RESULTS_DIR}/${FILENAME} --seed $SEED --dataset $CLASS_SPLIT

# python run_lightening.py test --config $CONFIG \
#                               --model.test_mode fill_memory \
#                               --out_path ${RESULTS_DIR}/memory_${SHOTS}shot_seed${SEED}.ckpt \
#                               --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
#                               --model.init_args.dataset_cfgs.fill_memory.memory_pkl ${RESULTS_DIR}/${FILENAME} \
#                               --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOTS \
#                               --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
#                               --trainer.logger.save_dir ${RESULTS_DIR}/ \
#                               --trainer.devices $GPUS

# python run_lightening.py test --config $CONFIG \
#                               --model.test_mode postprocess_memory \
#                               --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
#                               --ckpt_path ${RESULTS_DIR}/memory_${SHOTS}shot_seed${SEED}.ckpt \
#                               --out_path ${RESULTS_DIR}/memory_postprocessed_${SHOTS}shot_seed${SEED}.ckpt \
#                               --trainer.devices 1

python run_lightening.py test --config $CONFIG  \
                              --ckpt_path ${RESULTS_DIR}/memory_postprocessed_${SHOTS}shot_seed${SEED}.ckpt \
                              --model.init_args.test_mode test \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.dataset_cfgs.test.class_split $CLASS_SPLIT \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS


# For plotting the reference images, we need to adapt this command:
# python no_time_to_train/make_plots/plot_reference_images.py \
#         --json_path ./data/$1/annotations/1_shot_with_segm.json \
#         --image_dir ./data/$1/train \
#         --output_dir ./data/$1/annotations/1_shot_with_segm_vis \
#         --dataset_name $1


# Old simpler version
# #!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2

# # Fill memory with references
# python run_lightening.py test --config $CONFIG --model.test_mode fill_memory --out_path ./tmp_ckpts/0000.ckpt --trainer.devices $GPUS

# #  Postprocess memories, e.g., computer averages and run kmeans
# python run_lightening.py test --config $CONFIG --model.test_mode postprocess_memory --ckpt_path ./tmp_ckpts/0000.ckpt --out_path ./tmp_ckpts/1111.ckpt --trainer.devices 1

# # testing on the target set
# python run_lightening.py test --config $CONFIG --model.test_mode test --ckpt_path ./tmp_ckpts/1111.ckpt --trainer.devices $GPUS