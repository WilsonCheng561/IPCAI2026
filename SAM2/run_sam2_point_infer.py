#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SAM-2 inference with **point prompts** (postproc: dedup + clean + shape/brightness filter)
最小增量修改：
- 过滤过大的掩膜（> max_area_frac * 全图面积）
- 提高过小阈值
- 细长判定更严格

作者：WZC  2025-08-07
"""

import os, json, shutil, argparse
import numpy as np
import cv2, torch
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from sam2.backup.build_sam import build_sam2_video_predictor


# ------------------------------------------------------------------ #
# 0. Hydra / Device helpers
# ------------------------------------------------------------------ #
def setup_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

def setup_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("⚠  MPS support is still experimental for SAM-2.")
    else:
        dev = torch.device("cpu")
    print("Running on", dev)
    return dev


# ------------------------------------------------------------------ #
# A. 通用 & 后处理工具函数
# ------------------------------------------------------------------ #
def to_bin2d(mask) -> np.ndarray:
    """将任意形状的 mask 压成 HxW 的 0/1 uint8。"""
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[0] == 1:
            m = m[0]
        elif m.shape[-1] == 1:
            m = m[..., 0]
        else:
            m = np.squeeze(m)
    elif m.ndim > 3:
        m = np.squeeze(m)
    if m.ndim != 2:
        h, w = m.shape[:2]
        m = m.reshape(h, w)
    return (m > 0).astype(np.uint8)

def mask_area(mask) -> int:
    return int(to_bin2d(mask).sum())

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = to_bin2d(a).astype(bool); b = to_bin2d(b).astype(bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0: return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(max(union, 1))

def remove_small_components(mask: np.ndarray, min_area: int = 180) -> np.ndarray:
    """提高 min_area：更小的碎片被清理掉。"""
    m = to_bin2d(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for lab in range(1, num):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 1
    return out

def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = to_bin2d(mask)
    h, w = m.shape
    ff = (m * 255).copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)  # floodFill 需要额外边框
    cv2.floodFill(ff, pad, (0, 0), 255)
    inv = cv2.bitwise_not(ff)
    filled = cv2.bitwise_or(m * 255, inv)
    return (filled > 0).astype(np.uint8)

def smooth_mask(mask: np.ndarray, k: int = 5, it_close: int = 1, it_open: int = 1) -> np.ndarray:
    m = to_bin2d(mask) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if it_close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=it_close)
    if it_open > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  ker, iterations=it_open)
    return (m > 0).astype(np.uint8)

def clean_mask(mask: np.ndarray,
               min_area: int = 180,
               k: int = 5,
               it_close: int = 1,
               it_open: int = 1) -> np.ndarray:
    """组合清理：去小块 → 填洞 → 平滑。"""
    m = remove_small_components(mask, min_area=min_area)
    if m.any():
        m = fill_holes(m)
        m = smooth_mask(m, k=k, it_close=it_close, it_open=it_open)
    return m

def tool_like(mask: np.ndarray,
              aspect_thr: float = 2.4,
              ecc_thr: float = 0.90) -> bool:
    """
    细长度判定（更严格）：
    aspect = sqrt(λ1/λ2), ecc = sqrt(1 - λ2/λ1)
    满足 (aspect≥aspect_thr) 或 (ecc≥ecc_thr) 视为器械状。
    """
    m = to_bin2d(mask)
    y, x = np.nonzero(m)
    if x.size < 20:
        return False  # 太小/太少点，默认不通过（后面还有亮度兜底）
    xy = np.stack([x, y], 1).astype(np.float32)
    cov = np.cov(xy, rowvar=False)
    vals, _ = np.linalg.eigh(cov)
    vals = np.sort(np.maximum(vals, 1e-6))
    l1, l2 = vals[1], vals[0]
    aspect = float(np.sqrt(l1 / l2))
    ecc    = float(np.sqrt(1.0 - (l2 / l1)))
    return (aspect >= aspect_thr) or (ecc >= ecc_thr)

def keep_if_bright(img_bgr: np.ndarray,
                   mask: np.ndarray,
                   mean_thr: float = 0.72,
                   p95_thr: float = 0.90) -> bool:
    """
    亮度兜底：器械（尤其白色 hook）通常偏亮。
    如果掩膜内灰度均值 ≥ mean_thr 或 95 分位 ≥ p95_thr（0~1），则保留。
    """
    m = to_bin2d(mask).astype(bool)
    if m.sum() < 20:
        return False
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    vals = gray[m]
    return (vals.mean() >= mean_thr) or (np.percentile(vals, 95) >= p95_thr)

def dedup_masks_by_iou(masks_dict: dict, iou_thr: float = 0.80) -> dict:
    items = []
    for oid, m in masks_dict.items():
        area = int(to_bin2d(m).sum())
        items.append((oid, to_bin2d(m), area))
    # 大面积优先
    items.sort(key=lambda t: t[2], reverse=True)

    kept = []
    kept_dict = {}
    for oid, m, _ in items:
        ok = True
        for _, km in kept:
            if mask_iou(m, km) > iou_thr:
                ok = False
                break
        if ok:
            kept.append((oid, m))
            kept_dict[oid] = m
    return kept_dict


# ------------------------------------------------------------------ #
# 1. 可视化工具
# ------------------------------------------------------------------ #
def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.45):
    m = to_bin2d(mask).astype(bool)
    out  = img_bgr.copy()
    out[m] = (out[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out


# ------------------------------------------------------------------ #
# 2. 主流程
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    # python /home/haoding/Wenzheng/SAM2/run_sam2_point_infer_new.py \
    #   --pts_dir /home/haoding/Wenzheng/dinov2/figures_depthv2_points \
    #   --out_dir /home/haoding/Wenzheng/dinov2/figures_sam2_points_depthv2
    parser.add_argument("--img_dir", default="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data/video21/ws_0/images")
    parser.add_argument("--pts_dir", default="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/points")
    parser.add_argument("--out_dir", default="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/figures_points_sam2")
    parser.add_argument("--ckpt",    default="/projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt")
    parser.add_argument("--cfg",     default="sam2_hiera_l.yaml")

    # 后处理参数（更新默认值）
    parser.add_argument("--iou_dedup", type=float, default=0.80)
    parser.add_argument("--min_mask_area", type=int, default=180, help="清理阶段去小块阈值")
    parser.add_argument("--morph_kernel", type=int, default=5)
    # 形状 & 保留规则（更严格）
    parser.add_argument("--aspect_thr", type=float, default=2)
    parser.add_argument("--ecc_thr",    type=float, default=0.85)
    # 太小/太大过滤
    parser.add_argument("--keep_min_area", type=int, default=3000, help="整体掩膜过小则丢弃")
    parser.add_argument("--max_area_frac", type=float, default=0.30, help="整体掩膜过大(占比)则丢弃")
    # 亮度兜底
    parser.add_argument("--bright_mean_thr", type=float, default=0.72)
    parser.add_argument("--bright_p95_thr",  type=float, default=0.90)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_hydra()
    device    = setup_device()
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)
    print("SAM-2 predictor ready.\n")

    # 临时“单帧视频”文件夹
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 遍历 *_objects.json
    json_files = sorted(f for f in os.listdir(args.pts_dir) if f.endswith("_objects.json"))
    colors     = [(0,255,0), (0,128,255), (255,0,0), (255,0,255), (0,255,255),
                  (255,128,0), (128,0,255), (0,255,128), (255,0,128)]

    for jf in json_files:
        with open(os.path.join(args.pts_dir, jf)) as fp:
            data = json.load(fp)

        img_name = data["frame_file"]
        img_path = os.path.join(args.img_dir, img_name)
        if not os.path.exists(img_path):
            print("⚠  image missing:", img_path)
            continue

        # 提前读原图（用于亮度兜底判定 & 面积占比计算）
        img_bgr = cv2.imread(img_path)
        H, W = img_bgr.shape[:2]
        img_area = H * W
        max_area_abs = int(args.max_area_frac * img_area)

        # 将原图复制为 temp/0000000.jpg
        dst_name = "0000000.jpg"
        shutil.copy(img_path, os.path.join(tmp_dir, dst_name))

        # 初始化 SAM-2 状态
        state = predictor.init_state(video_path=tmp_dir)
        predictor.reset_state(state)

        # 加入 point prompts
        valid_oids = []
        for obj in data["objects"]:
            if len(obj.get("points", [])) == 0:
                continue
            pts_list = obj["points"]
            pts  = np.array(pts_list,  np.float32)
            # ★ 最小修改：labels 缺省兜底；并兼容不同 predictor 版本，避免 object_score_logits KeyError
            labs_list = obj.get("labels", [1] * len(pts_list))
            labs = np.array(labs_list, dtype=np.int64)

            try:
                if hasattr(predictor, "add_new_points"):
                    predictor.add_new_points(state, 0, obj["obj_id"], pts, labs)
                else:
                    predictor.add_new_points_or_box(state, 0, obj["obj_id"], pts, labs)
            except KeyError as e:
                print(f"[WARN] {e} on add_new_points_or_box → fallback to add_new_points for obj_id={obj['obj_id']}")
                if hasattr(predictor, "add_new_points"):
                    predictor.add_new_points(state, 0, obj["obj_id"], pts, labs)
                else:
                    raise
            valid_oids.append(obj["obj_id"])

        # Propagate - 单帧
        raw_masks = {}
        for _, obj_ids, logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(obj_ids):
                if oid in valid_oids:
                    raw_masks[oid] = (logits[i] > 0).cpu().numpy()

        # === 后处理 ===
        # 1) IoU 去重
        masks = dedup_masks_by_iou(raw_masks, iou_thr=args.iou_dedup)

        # 2) 清理每个掩膜
        for oid in list(masks.keys()):
            masks[oid] = clean_mask(masks[oid],
                                    min_area=args.min_mask_area,
                                    k=args.morph_kernel,
                                    it_close=1, it_open=1)

        # 2.5) 过小/过大直接丢弃（新增 max_area_frac）
        for oid in list(masks.keys()):
            area = mask_area(masks[oid])
            if area < args.keep_min_area or area > max_area_abs:
                del masks[oid]

        # 3) 形状筛选（更严格）+ 亮度兜底（保住白色 hook）
        for oid in list(masks.keys()):
            ok_shape = tool_like(masks[oid],
                                 aspect_thr=args.aspect_thr,
                                 ecc_thr=args.ecc_thr)
            if not ok_shape:
                ok_bright = keep_if_bright(img_bgr, masks[oid],
                                           mean_thr=args.bright_mean_thr,
                                           p95_thr=args.bright_p95_thr)
                if not ok_bright:
                    del masks[oid]

        # 叠加可视化
        combined = img_bgr.copy()
        kept_ids = sorted(masks.keys())
        for i, oid in enumerate(kept_ids):
            combined = overlay_mask(combined, masks[oid], colors[i % len(colors)])
            for obj in data["objects"]:
                if obj["obj_id"] == oid:
                    for (x, y) in obj["points"]:
                        cv2.circle(combined, (int(x), int(y)), 4, colors[i % len(colors)], -1)
                    break

        out_path = os.path.join(args.out_dir, img_name)
        cv2.imwrite(out_path, combined)
        print(f"✓ saved {out_path} | kept {len(kept_ids)}/{len(raw_masks)} masks")

        # 清理 temp
        os.remove(os.path.join(tmp_dir, dst_name))

    os.rmdir(tmp_dir)
    print("\nAll done, results in", args.out_dir)


if __name__ == "__main__":
    main()
