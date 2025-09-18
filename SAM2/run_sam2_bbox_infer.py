#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SAM‑2 inference on single images using previously generated bbox prompts.
作者：WZC  2025‑07‑28
"""

import os, json, shutil, argparse
import numpy as np
import cv2
from PIL import Image
import torch, matplotlib.pyplot as plt

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from omegaconf import OmegaConf
from sam2.backup.build_sam import build_sam2_video_predictor


# ----------------------------------------------------------------------
# 0. Hydra / Device helpers (同你之前脚本，稍微简化)
# ----------------------------------------------------------------------
def setup_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚠  MPS support is still experimental for SAM‑2.")
    else:
        device = torch.device("cpu")
    print("Running on", device)
    return device

# ----------------------------------------------------------------------
# 1. 提示构建：bbox → 中心点正样
# ----------------------------------------------------------------------
def bbox_json_to_objects(bbox_json):
    """
    Convert one bbox.json file to list of objects:
    [{"obj_id":1,"points":[[cx,cy]],"labels":[1]}…]
    """
    objects = []
    for i, item in enumerate(bbox_json["bboxes"]):
        x0,y0,x1,y1 = item["box"]
        cx, cy = (x0+x1)/2.0, (y0+y1)/2.0
        objects.append({
            "obj_id": i+1,
            "points": [[cx, cy]],
            "labels": [1],          # positive point
            "score": item["score"]
        })
    # 置信度降序
    objects.sort(key=lambda o: o["score"], reverse=True)
    return objects

# ----------------------------------------------------------------------
# 2. 可视化工具
# ----------------------------------------------------------------------
def overlay_mask(img_bgr, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay a boolean mask onto BGR image.
    mask shape can be (H,W) or (1,H,W) or (H,W,1).
    """
    # squeeze to (H,W)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0) if mask.shape[0] == 1 else np.squeeze(mask, axis=-1)
    mask = mask.astype(bool)

    overlay = img_bgr.copy()
    # broadcast color to selected pixels
    overlay[mask] = (
        overlay[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    return overlay


# ----------------------------------------------------------------------
# 3. 主函数
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",   default="/home/haoding/Wenzheng/dinov2/figures")
    parser.add_argument("--bbox_dir",  default="/home/haoding/Wenzheng/dinov2/figures_bbox")
    parser.add_argument("--out_dir",   default="/home/haoding/Wenzheng/dinov2/figures_sam2")
    parser.add_argument("--ckpt", default="/home/haoding/DT_SPR_utils/sam2/SAM2/checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--cfg",       default="sam2_hiera_l.yaml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_hydra()
    device = setup_device()

    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)
    print("SAM‑2 predictor ready.\n")

    # temp dir (单帧伪“视频”)
    temp_video = os.path.join(args.out_dir, "_tmp")
    os.makedirs(temp_video, exist_ok=True)

    # loop over bbox json files
    for bbox_file in sorted(os.listdir(args.bbox_dir)):
        if not bbox_file.endswith("_bbox.json"):
            continue
        with open(os.path.join(args.bbox_dir, bbox_file)) as f:
            bbox_json = json.load(f)

        img_name = bbox_json["image"]
        img_path = os.path.join(args.img_dir, img_name)
        if not os.path.exists(img_path):
            print("⚠ img not found:", img_path); continue

        # copy image into temp folder as "0000000.jpg"
        dst_name = "0000000.jpg"
        shutil.copy(img_path, os.path.join(temp_video, dst_name))

        # init SAM‑2 state
        state = predictor.init_state(video_path=temp_video)
        predictor.reset_state(state)

        # build objects prompt
        objects = bbox_json_to_objects(bbox_json)
        for obj in objects:
            pts  = np.array(obj["points"], np.float32)
            labs = np.array(obj["labels"], np.int32)

            # --- 兼容不同版本的 predictor 接口 ---
            if hasattr(predictor, "add_new_points_or_box"):
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj["obj_id"],
                    points=pts,
                    labels=labs
                )
            elif hasattr(predictor, "add_new_points"):
                predictor.add_new_points(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj["obj_id"],
                    points=pts,
                    labels=labs
                )
            else:
                raise AttributeError(
                    "Predictor has neither 'add_new_points_or_box' "
                    "nor 'add_new_points' methods.")


        # propagate (单帧) 获取 mask
        masks_dict = {}
        for _, obj_ids, logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(obj_ids):
                masks_dict[oid] = (logits[i] > 0.0).cpu().numpy()

        # 读取原图并叠加所有 mask
        img_bgr = cv2.imread(img_path)
        combined = img_bgr.copy()
        colors = [(0,255,0), (0,128,255), (255,0,0), (255,0,255), (0,255,255)]
        for i,obj in enumerate(objects):
            mask = masks_dict.get(obj["obj_id"])
            if mask is None: continue
            combined = overlay_mask(combined, mask, color=colors[i%len(colors)], alpha=0.45)
            x0,y0,x1,y1 = map(int, obj["points"][0] + obj["points"][0])  # dummy to silence linter

        # 画框+分数
        for i,obj in enumerate(objects):
            x0,y0,x1,y1 = map(int, bbox_json["bboxes"][i]["box"])
            cv2.rectangle(combined, (x0,y0), (x1,y1), colors[i%len(colors)], 2)
            cv2.putText(combined, f"{obj['score']:.2f}", (x0, y0-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i%len(colors)], 1)

        # 保存
        out_path = os.path.join(args.out_dir, img_name)
        cv2.imwrite(out_path, combined)
        print("✓ saved", out_path)

        # 清理 temp
        os.remove(os.path.join(temp_video, dst_name))

    os.rmdir(temp_video)
    print("\nAll done, results in", args.out_dir)

if __name__ == "__main__":
    main()
