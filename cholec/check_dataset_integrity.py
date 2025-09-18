#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查 no-time-to-train 自定义数据集的完整性
- 确认 references/targets JSON segmentation 是否非空
- 检查 PKL 是否正确对应 JSON annotations
- 随机保存可视化 (bbox + segm) 到 debug_vis/
"""

import json
import pickle
import random
from pathlib import Path
import cv2
import numpy as np

# ====== 路径配置 ======
DATA_ROOT = Path("/mnt/disk0/haoding/no-time-to-train/data/annotations")
REF_JSON = DATA_ROOT / "custom_references_with_SAM_segm.json"
TAR_JSON = DATA_ROOT / "custom_targets_with_SAM_segm.json"
PKL = DATA_ROOT / "custom_refs_5shot.pkl"   # 可以改成 1shot/10shot 的

IMG_ROOT = Path("/mnt/disk0/haoding/no-time-to-train/data/images")
DEBUG_OUTDIR = Path("./debug_vis")
DEBUG_OUTDIR.mkdir(exist_ok=True)


def check_json(json_path, name="refs"):
    if not json_path.exists():
        print(f"❌ {json_path} 不存在")
        return {}, []

    data = json.loads(json_path.read_text())
    anns = data.get("annotations", [])
    imgs = {im["id"]: im for im in data.get("images", [])}
    print(f"✅ {name}: {len(imgs)} images, {len(anns)} annotations")

    # 打印前 5 个 annotation
    for ann in anns[:5]:
        seg = ann.get("segmentation", [])
        print(f"  ann_id={ann['id']} cat={ann['category_id']} "
              f"bbox={ann['bbox']} seg_len={len(seg)}")

    return imgs, anns


def check_pkl(pkl_path, ref_anns):
    if not pkl_path.exists():
        print(f"❌ {pkl_path} 不存在")
        return
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    id2ann = {a["id"]: a for a in ref_anns}
    print(f"✅ PKL 类别数: {len(data)}")
    for cid, samples in list(data.items())[:3]:
        print(f"  Category {cid}: {len(samples)} samples")
        for s in samples[:2]:
            anns = [id2ann[aid] for aid in s["ann_ids"] if aid in id2ann]
            for ann in anns:
                seg = ann.get("segmentation", [])
                print(f"    ann_id={ann['id']} seg_len={len(seg)}")


def visualize_samples(anns, imgs, name="refs"):
    """随机可视化 5 张图像，保存到 debug_vis"""
    for ann in random.sample(anns, min(5, len(anns))):
        img_info = imgs.get(ann["image_id"])
        if not img_info:
            continue
        img_path = IMG_ROOT / img_info["file_name"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))

        segms = ann.get("segmentation", [])
        if segms:
            for seg in segms:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        else:
            x, y, w, h = ann["bbox"]
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                          (0, 255, 0), 2)

        save_p = DEBUG_OUTDIR / f"{name}_ann_{ann['id']}.jpg"
        cv2.imwrite(str(save_p), img)
        print(f"[VIS] saved {save_p}")


def main():
    # 1. 检查 references
    ref_imgs, ref_anns = check_json(REF_JSON, "references")

    # 2. 检查 targets
    tar_imgs, tar_anns = check_json(TAR_JSON, "targets")

    # 3. 检查 PKL 是否对应
    check_pkl(PKL, ref_anns)

    # 4. 可视化
    if ref_anns:
        visualize_samples(ref_anns, ref_imgs, "refs")
    if tar_anns:
        visualize_samples(tar_anns, tar_imgs, "tars")


if __name__ == "__main__":
    main()
