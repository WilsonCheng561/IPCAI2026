#!/usr/bin/env bash
set -Eeuo pipefail

# 用法示例：
#   bash run_one_video.sh 21 22 23 24

usage() {
  echo "Usage: $0 <video_id ...>   # e.g. $0 21 22 23 24"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

# -------- 可改参数（与你原脚本一致） --------
DEPTH_SCRIPT="/home/wcheng31/Depth-Anything-V2/depth.py"

DINO_HEATMAP="/home/wcheng31/dinov2/heatmap.py"
DINO_CKPT="facebook/dino-vitb16"

COMBINE_POINTS="/home/wcheng31/dinov2/generate_point4sam2.py"

SAM2_INFER="/home/wcheng31/SAM2/run_sam2_point_infer.py"
SAM2_CKPT="/projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt"
SAM2_CFG="sam2_hiera_l.yaml"

# -------- 小工具函数 --------
validate_vid() {
  local v="$1"
  [[ "$v" =~ ^[0-9]+$ ]] && (( v >= 21 && v <= 30 ))
}

run_one_video() {
  local VID="$1"

  # 路径配置（每个视频独立）
  local BASE="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data/video${VID}/ws_0"
  local IMG_DIR="${BASE}/images"
  local PROMPTS_JSON="${BASE}/prompts.json"

  # 中间文件与可视化输出目录（迁移到 ours）
  local OURS_BASE="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/cholec80_video${VID}"
  local VIZ_DIR="${OURS_BASE}/figures_attn_depthv2"
  local DINO_DIR="${OURS_BASE}/figures_attn_dino1"
  local PTS_DIR="${OURS_BASE}/points"
  local SAM2_OUT="${OURS_BASE}/figures_points_sam2"

  echo ""
  echo "=== Video ${VID} | BASE = ${BASE} ==="

  mkdir -p "$VIZ_DIR" "$DINO_DIR" "$PTS_DIR" "$SAM2_OUT"

  # 1) Depth-Anything 只跑 prompts.json 涉及的帧
  python "$DEPTH_SCRIPT" \
    --img_dir "$IMG_DIR" \
    --out_dir "$VIZ_DIR" \
    --prompts_json "$PROMPTS_JSON"

  # 2) DINO 热图同样只跑 prompts.json 里的帧
  python "$DINO_HEATMAP" \
    --image_path "$IMG_DIR" \
    --output_dir "$DINO_DIR" \
    --ckpt "$DINO_CKPT" \
    --prompts_json "$PROMPTS_JSON"

  # 3) 合并两路点到 points 目录（obj_id 顺延相加）
  python "$COMBINE_POINTS" \
    --img_dir "$IMG_DIR" \
    --depth_dir "$VIZ_DIR" \
    --dino_dir "$DINO_DIR" \
    --out_dir "$PTS_DIR" \
    --max_regions 9 --pos_max 3 \
    --prompts_json "$PROMPTS_JSON"

  # 4) SAM-2 推理：对该视频所有 *_objects.json 推理并可视化
  python "$SAM2_INFER" \
    --img_dir "$IMG_DIR" \
    --pts_dir "$PTS_DIR" \
    --out_dir "$SAM2_OUT" \
    --ckpt "$SAM2_CKPT" \
    --cfg "$SAM2_CFG"

  echo "=== Done video ${VID}. Outputs in:"
  echo "    Depth viz : ${VIZ_DIR}"
  echo "    DINO attn : ${DINO_DIR}"
  echo "    Points    : ${PTS_DIR}"
  echo "    SAM2 viz  : ${SAM2_OUT}"
}

# -------- 主循环：逐个视频执行，失败不影响后续 --------
ok_list=()
fail_list=()
invalid_list=()

for VID in "$@"; do
  if ! validate_vid "$VID"; then
    echo "[SKIP] invalid video id: ${VID} (expect 21-30)"
    invalid_list+=("$VID")
    continue
  fi

  # 不让单个失败中断整体
  set +e
  run_one_video "$VID"
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    ok_list+=("$VID")
  else
    echo "[FAIL] video ${VID} (exit code ${rc})"
    fail_list+=("$VID")
  fi
done

# -------- 汇总 --------
echo ""
echo "========== SUMMARY =========="
[[ ${#ok_list[@]}    -gt 0 ]] && echo "OK     : ${ok_list[*]}"
[[ ${#fail_list[@]}  -gt 0 ]] && echo "FAILED : ${fail_list[*]}"
[[ ${#invalid_list[@]} -gt 0 ]] && echo "SKIPPED: ${invalid_list[*]}"
echo "============================="
