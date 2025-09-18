#!/usr/bin/env bash
# 用法:  ./run.sh 1 5 10 30

set -e

# ---- 关键修复：移除错误的等号写法的环境变量（Torch 只认冒号语法）----
unset PYTORCH_CUDA_ALLOC_CONF || true
# ---------------------------------------------------------------------

YAML_PATH=cholec/pl_configs/cholec80.yaml
DATA_ROOT=/mnt/disk0/haoding/no-time-to-train/data

# 参考集（前 1-20 视频）——用于 fill_memory/pkl
REF_JSON=$DATA_ROOT/annotations/custom_references_with_SAM_segm.json
# 目标集（21-30 视频）——用于 test
TAR_JSON=$DATA_ROOT/annotations/custom_targets_with_SAM_segm.json  # 若还没有 _with_SAM_segm，可先用 custom_targets.json

for K in "$@"; do
  echo -e "\n=====  RUN ${K}-shot  ====="
  SAVE_DIR=work_dirs/cholec80_${K}shot
  mkdir -p "$SAVE_DIR"

  # 1) pkl 一定用 “同一个 REF_JSON” 生成
  PKL=$DATA_ROOT/annotations/custom_refs_${K}shot.pkl
  python /home/haoding/Wenzheng/no-time-to-train/cholec/coco_to_pkl_k.py "$REF_JSON" "$PKL" "$K"

  # 2) fill_memory —— 强制把 root/json_file 对齐到参考集 + 传同一个 pkl
  python run_lightening.py test --config $YAML_PATH \
      --model.test_mode fill_memory \
      --out_path $SAVE_DIR/cholec80_${K}shot_refs_memory.pth \
      --model.init_args.dataset_cfgs.fill_memory.root $DATA_ROOT/images \
      --model.init_args.dataset_cfgs.fill_memory.json_file $REF_JSON \
      --model.init_args.dataset_cfgs.fill_memory.memory_pkl $PKL \
      --model.init_args.dataset_cfgs.fill_memory.memory_length $K \
      --model.init_args.model_cfg.memory_bank_cfg.length $K \
      --trainer.devices 1

  # 3) postprocess
  python run_lightening.py test --config $YAML_PATH \
      --model.test_mode postprocess_memory \
      --ckpt_path $SAVE_DIR/cholec80_${K}shot_refs_memory.pth \
      --out_path $SAVE_DIR/cholec80_${K}shot_refs_memory_post.pth \
      --model.init_args.model_cfg.memory_bank_cfg.length $K \
      --trainer.devices 1

  # 4) test —— 指定 test 用的 root/json_file 到“目标集”
  python run_lightening.py test --config $YAML_PATH \
      --model.test_mode test \
      --ckpt_path $SAVE_DIR/cholec80_${K}shot_refs_memory_post.pth \
      --model.init_args.model_cfg.memory_bank_cfg.length $K \
      --model.init_args.dataset_cfgs.test.root $DATA_ROOT/images \
      --model.init_args.dataset_cfgs.test.json_file $TAR_JSON \
      --trainer.devices 1
done
