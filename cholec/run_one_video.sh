#!/usr/bin/env bash
# 用法:
#   bash cholec/run_one_video.sh 21 5
#   第一个参数: 视频号(如 21 或 21-23)
#   第二个参数: K-shot (默认 5)

set -e

VIDEOS_SPEC="${1:?请给视频号, 如 21 或 21-23}"
K="${2:-5}"

YAML_PATH=cholec/pl_configs/cholec80.yaml
DATA_ROOT=/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/data

# 参考集（前 1-20 视频）的 json + K-shot pkl（你已有）
REF_JSON=$DATA_ROOT/annotations/custom_references_with_SAM_segm.json
PKL=$DATA_ROOT/annotations/custom_refs_${K}shot.pkl
if [ ! -f "$PKL" ]; then
  echo "[INFO] 生成 ${K}-shot pkl ..."
  python cholec/coco_to_pkl_k.py "$REF_JSON" "$PKL" "$K"
fi

# 目标集总 JSON（包含 21-30）
ALL_TAR_JSON=$DATA_ROOT/annotations/custom_targets_with_SAM_segm.json

# 先把目标集切成单/少量视频的子 JSON
SLICE_JSON=$DATA_ROOT/annotations/custom_targets_with_SAM_segm_v${VIDEOS_SPEC}.json
python cholec/slice_coco_by_video.py --in "$ALL_TAR_JSON" --videos "$VIDEOS_SPEC" --out "$SLICE_JSON"

SAVE_DIR=/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/work_dirs/cholec80_${K}shot_v${VIDEOS_SPEC}
mkdir -p "$SAVE_DIR"

# 1) fill_memory（只做一次也行，这里为自洽起见仍跑，但很快）
python run_lightening.py test --config $YAML_PATH \
  --model.test_mode fill_memory \
  --out_path $SAVE_DIR/cholec80_${K}shot_refs_memory.pth \
  --model.init_args.dataset_cfgs.fill_memory.root $DATA_ROOT/images \
  --model.init_args.dataset_cfgs.fill_memory.json_file $REF_JSON \
  --model.init_args.dataset_cfgs.fill_memory.memory_pkl $PKL \
  --model.init_args.dataset_cfgs.fill_memory.memory_length $K \
  --model.init_args.model_cfg.memory_bank_cfg.length $K \
  --trainer.devices 1 \
  --trainer.precision 16

# 2) postprocess
python run_lightening.py test --config $YAML_PATH \
  --model.test_mode postprocess_memory \
  --ckpt_path $SAVE_DIR/cholec80_${K}shot_refs_memory.pth \
  --out_path $SAVE_DIR/cholec80_${K}shot_refs_memory_post.pth \
  --model.init_args.model_cfg.memory_bank_cfg.length $K \
  --trainer.devices 1 \
  --trainer.precision 16

# 3) test（只在切出来的子 JSON 上评估，显著加速）
python run_lightening.py test --config $YAML_PATH \
  --model.test_mode test \
  --ckpt_path $SAVE_DIR/cholec80_${K}shot_refs_memory_post.pth \
  --model.init_args.model_cfg.memory_bank_cfg.length $K \
  --model.init_args.dataset_cfgs.test.root $DATA_ROOT/images \
  --model.init_args.dataset_cfgs.test.json_file $SLICE_JSON \
  --trainer.devices 1 \
  --trainer.precision 16 \
  --trainer.limit_test_batches 1.0 \
  --test.debug_check \
  --test.debug_outdir $SAVE_DIR/debug_vis
