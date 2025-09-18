#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script for no-time-to-train predictions.
增加功能: 可视化结果保存到对应 video/K 文件夹
"""

import os, json, random, re
from pathlib import Path
import numpy as np
import cv2
from collections import Counter
from pycocotools import mask as maskUtils

# ==== 配置路径 ====
SAVE_DIR = Path("/home/haoding/Wenzheng/no-time-to-train/work_dirs/cholec80_30shot_v26")
GT_JSON = Path("/mnt/disk0/haoding/no-time-to-train/data/annotations/custom_targets_with_SAM_segm_v26.json")
PRED_JSON = SAVE_DIR / "export_result.json"
MEM_CKPT = SAVE_DIR / "cholec80_30shot_refs_memory_post.pth"
IMG_ROOT = Path("/mnt/disk0/haoding/no-time-to-train/data/images")
OUTDIR = Path("/home/haoding/Wenzheng/no-time-to-train/pred_vis")
OUTDIR.mkdir(exist_ok=True)

# ==== 解析 K ====
m = re.search(r"(\d+)shot", str(SAVE_DIR))
K_SHOT = int(m.group(1)) if m else -1

# ==== 1. Check memory checkpoint ====
print("\n=== [1] Memory Checkpoint ===")
if MEM_CKPT.exists():
    print(f"✅ Found memory checkpoint: {MEM_CKPT}")
    size = MEM_CKPT.stat().st_size / 1024 / 1024
    print(f"   File size: {size:.2f} MB")
else:
    print(f"❌ Memory checkpoint missing: {MEM_CKPT}")

# ==== 2. Load GT ====
print("\n=== [2] Ground Truth ===")
gt = json.loads(GT_JSON.read_text())
print(f"GT: {len(gt['images'])} images, {len(gt['annotations'])} annotations")
gt_ids = {im["id"] for im in gt["images"]}
gt_cats = Counter([ann["category_id"] for ann in gt["annotations"]])
print("GT 类别覆盖:", gt_cats)

# ==== 3. Load Predictions ====
print("\n=== [3] Predictions ===")
if not PRED_JSON.exists():
    print(f"❌ PRED_JSON 不存在: {PRED_JSON}")
    exit(1)

preds = json.loads(PRED_JSON.read_text())
print(f"Predictions: {len(preds)} entries")
if len(preds) > 0:
    print("示例预测:", preds[0])

# ==== 4. Category ID check ====
print("\n=== [4] Category ID check ===")
pred_cats = Counter([p["category_id"] for p in preds])
print("Pred categories:", pred_cats)
if pred_cats and min(pred_cats.keys()) == 0:
    print("⚠️ Predicted categories start from 0 (可能需要 +1)")

# ==== 5. Segmentation check ====
print("\n=== [5] Segmentation check ===")
empty_seg = sum([1 for p in preds if not p.get("segmentation")])
print(f"Empty segmentation: {empty_seg}/{len(preds)} ({empty_seg/len(preds)*100:.1f}%)")

short_seg = 0
for p in preds:
    seg = p.get("segmentation")
    if isinstance(seg, list):
        if all(len(s) < 6 for s in seg):
            short_seg += 1
print(f"Too-short segmentation (<3 pts): {short_seg}")

# ==== 6. Image ID coverage ====
print("\n=== [6] Image ID coverage ===")
pred_ids = {p["image_id"] for p in preds}
inter = gt_ids & pred_ids
print(f"Pred IDs={len(pred_ids)}, GT IDs={len(gt_ids)}, Intersection={len(inter)}")
print(f"Coverage ratio: {len(inter)/len(gt_ids):.3f}")

# ==== 7. Recall (IoU>=0.5) ====
print("\n=== [7] Recall (IoU>=0.5, per-class) ===")

def iou_bbox(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    inter = max(0, xB-xA+1) * max(0, yB-yA+1)
    union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter/union if union > 0 else 0

gt_by_img = {}
for ann in gt["annotations"]:
    gt_by_img.setdefault(ann["image_id"], []).append(ann)

TP, TOTAL = Counter(), Counter()
for img_id, anns in gt_by_img.items():
    preds_img = [p for p in preds if p["image_id"] == img_id]
    for ann in anns:
        TOTAL[ann["category_id"]] += 1
        found = False
        for p in preds_img:
            if p["category_id"] == ann["category_id"]:
                if iou_bbox(ann["bbox"], p["bbox"]) >= 0.5:
                    found = True
                    break
        if found:
            TP[ann["category_id"]] += 1

for cid in sorted(TOTAL.keys()):
    recall = TP[cid] / TOTAL[cid] if TOTAL[cid] > 0 else 0
    print(f"Class {cid}: Recall={recall:.3f} ({TP[cid]}/{TOTAL[cid]})")

overall_recall = sum(TP.values()) / sum(TOTAL.values())
print(f"Overall Recall: {overall_recall:.3f}")

# ==== 8. Visualization ====
print("\n=== [8] Visualization (5 random samples) ===")
id2img = {im["id"]: im for im in gt["images"]}

for p in random.sample(preds, min(5, len(preds))):
    img_id = p["image_id"]
    if img_id not in id2img:
        continue
    img_info = id2img[img_id]
    img_path = IMG_ROOT / img_info["file_name"]
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))

    # draw prediction (red)
    seg = p.get("segmentation")
    if seg:
        if isinstance(seg, list):
            for s in seg:
                pts = np.array(s).reshape(-1, 2).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        elif isinstance(seg, dict):
            m = maskUtils.decode(seg)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cv2.polylines(img, [c], True, (0, 0, 255), 2)
    else:
        x, y, w, h = p["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)

    # draw GT (green)
    ganns = gt_by_img.get(img_id, [])
    for g in ganns:
        x, y, w, h = g["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

    # === 新增: 按 video/K 保存 ===
    vid = str(img_id)[:2]  # e.g., "21"
    subdir = OUTDIR / f"video{vid}_K{K_SHOT}"
    subdir.mkdir(parents=True, exist_ok=True)

    save_p = subdir / f"compare_{img_id}.jpg"
    cv2.imwrite(str(save_p), img)
    print(f"[VIS] saved {save_p.resolve()}")

