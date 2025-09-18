#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug 脚本：检查 GT JSON、PKL 和 Pred JSON
"""

import json, pickle, random
from pathlib import Path
import numpy as np
import cv2
from collections import Counter
from pycocotools import mask as mask_utils

# ===== 配置 =====
DATA_ROOT = Path("/mnt/disk0/haoding/no-time-to-train/data")
GT_JSON = DATA_ROOT / "annotations/custom_targets_with_SAM_segm.json"
PKL_FILE = DATA_ROOT / "annotations/custom_refs_5shot.pkl"
PRED_JSON = Path("export_result.json")   # 你 test 导出的预测结果
IMG_DIR = DATA_ROOT / "images"
OUT_DIR = Path("./debug_vis")
OUT_DIR.mkdir(exist_ok=True)


# ===== 工具函数 =====
def show_mask(img, segms, color=(0, 0, 255)):
    """画 segmentation (polygon or RLE)"""
    if isinstance(segms, list) and len(segms) > 0:  # polygon
        for seg in segms:
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [pts], True, color, 2)
    elif isinstance(segms, dict) and "counts" in segms:  # RLE
        m = mask_utils.decode(segms)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.polylines(img, [c], True, color, 2)


def show_bbox(img, bbox, color=(0, 255, 0)):
    x, y, w, h = bbox
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)


# ===== 检查 GT JSON =====
print("\n=== [1] Ground Truth JSON ===")
gt = json.loads(GT_JSON.read_text())
print(f"GT: {len(gt['images'])} images, {len(gt['annotations'])} annotations")

# 打印前 5 个标注
for ann in gt["annotations"][:5]:
    print(f"ann_id={ann['id']} cat={ann['category_id']} bbox={ann['bbox']} seg_len={len(ann['segmentation'])}")

# 类别分布
cats = [a["category_id"] for a in gt["annotations"]]
print("GT 类别覆盖:", Counter(cats))

# segmentation 点数统计
seg_lens = [len(a["segmentation"][0]) if a.get("segmentation") else 0 for a in gt["annotations"]]
print("Segmentation 点数分布（前 20）:", seg_lens[:20])


# ===== 检查 PKL =====
print("\n=== [2] Reference PKL ===")
with open(PKL_FILE, "rb") as f:
    pkl = pickle.load(f)

print(f"PKL 类别数: {len(pkl)}")
for cid, samples in list(pkl.items())[:3]:
    print(f"Category {cid}: {len(samples)} samples")
    for s in samples[:2]:
        print(" ", s)


# ===== 检查预测结果 =====
print("\n=== [3] Predictions JSON ===")
if PRED_JSON.exists():
    preds = json.loads(PRED_JSON.read_text())
    print(f"Predictions: {len(preds)} entries")
    for p in preds[:5]:
        seg_type = "RLE" if isinstance(p.get("segmentation"), dict) else "Polygon"
        print(f"pred: cat={p.get('category_id')} bbox={p.get('bbox')} seg={seg_type}")
else:
    print("⚠️ PRED_JSON 不存在")


# ===== 可视化对比 =====
print("\n=== [4] 可视化 GT vs Pred ===")
id2img = {im["id"]: im for im in gt["images"]}
sample_anns = random.sample(gt["annotations"], min(5, len(gt["annotations"])))

for ann in sample_anns:
    img_info = id2img[ann["image_id"]]
    img_path = IMG_DIR / img_info["file_name"]
    if not img_path.exists():
        continue

    img_gt = cv2.imread(str(img_path))
    img_pred = img_gt.copy()

    # 画 GT
    show_bbox(img_gt, ann["bbox"], color=(0, 255, 0))
    if ann.get("segmentation"):
        show_mask(img_gt, ann["segmentation"], color=(0, 0, 255))

    # 找对应预测
    pred_match = None
    if PRED_JSON.exists():
        for p in preds:
            if p.get("image_id") == ann["image_id"]:
                pred_match = p
                break
    if pred_match:
        show_bbox(img_pred, pred_match.get("bbox", []), color=(255, 0, 0))
        if pred_match.get("segmentation"):
            show_mask(img_pred, pred_match["segmentation"], color=(255, 0, 0))

    # 拼图保存
    combined = np.concatenate([img_gt, img_pred], axis=1)
    save_p = OUT_DIR / f"compare_{ann['id']}.jpg"
    cv2.imwrite(str(save_p), combined)
    print(f"[VIS] saved {save_p}")
