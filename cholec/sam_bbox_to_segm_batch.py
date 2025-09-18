#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard-coded version: 直接运行即可，无需再传 --input_json 等参数
✅ saved to /mnt/disk0/haoding/no-time-to-train/data/annotations/custom_references_with_SAM_segm.json
✅ visualisations -> /mnt/disk0/haoding/no-time-to-train/data/annotations/references_visualisations
"""

# ───────────────────────── ❶ 手动配置区 ─────────────────────────
# INPUT_JSON  = "/mnt/disk0/haoding/no-time-to-train/data/annotations/custom_references.json"
INPUT_JSON  = "/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/data/annotations/custom_targets.json"
IMAGE_DIR   = "/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/data/images"

SAM_CFG     = "/home/wcheng31/no-time-to-train/cholec/configs/sam2/sam2_hiera_l.yaml"
SAM_CKPT    = "/projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt"
DEVICE      = "cuda"        # "cpu" 亦可
VISUALIZE   = True          # 生成 PNG 可视化
# ───────────────────────────────────────────────────────────────

import json, os, shutil, sys
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from sam2.build_sam import build_sam2_video_predictor

# ========== Hydra & SAM-2 加载 ==========
def _setup_hydra(repo_root: Path):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    os.chdir(repo_root)
    initialize(config_path="configs/sam2", version_base="1.2")

def load_predictor(cfg_yaml: str, ckpt: str, device: str = "cuda"):
    repo_root = Path(cfg_yaml).parent.parent      # 假设 .../configs/sam2/xxx.yaml
    _setup_hydra(repo_root)
    return build_sam2_video_predictor(Path(cfg_yaml).name, ckpt, device=device)

# ========== 工具函数 ==========
def coco_bbox_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def mask_to_polygon(mask: np.ndarray, fallback_box=None):
    """
    二值掩码 ➜ COCO polygon（list[list[float]]）
    • 若掩码尺寸为 0、findContours 抛错，或得到的轮廓 < 1，
      则退化为 bbox 四边形。
    """
    if mask.ndim < 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        pass
    else:
        try:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            polys = [c.flatten().tolist() for c in contours if len(c) >= 6]
            if polys:
                return polys
        except cv2.error as e:
            print(f"\033[93m[mask_to_polygon] OpenCV error → fall back to bbox: {e}\033[0m")

    if fallback_box is not None:
        x, y, w, h = fallback_box
        return [[x, y, x + w, y, x + w, y + h, x, y + h]]
    else:
        return []

def show_vis(img_bgr, masks, polys, boxes, labels, save_p):
    """
    半透明 mask + polygon + bbox + 字符标签
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    for m in masks:
        m2 = np.squeeze(m)
        if m2.max() == 0:
            continue
        color = np.concatenate([np.random.rand(3), [0.45]])  # RGBA
        rgba  = np.zeros((*m2.shape, 4))
        rgba[..., :3] = color[:3]
        rgba[..., 3]  = m2 * color[3]
        ax.imshow(rgba)

    for poly in polys:
        pts = np.array(poly).reshape(-1, 2)
        ax.plot(pts[:, 0], pts[:, 1], '-r', lw=1.4)

    for (x, y, w, h), lab in zip(boxes, labels):
        ax.add_patch(Rectangle((x, y), w, h,
                               edgecolor='lime', facecolor='none', lw=1.3))
        ax.text(x, y - 2, lab, fontsize=7, color='yellow',
                bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'))

    ax.axis('off')
    fig.savefig(save_p, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ========== 主流程 ==========
def main():
    img_dir  = Path(IMAGE_DIR)
    in_json  = Path(INPUT_JSON)
    out_json = in_json.with_name(in_json.stem + "_with_SAM_segm.json")
    vis_dir  = out_json.parent / (in_json.stem + "_visualisations")   # <<< 改：动态命名
    if VISUALIZE:
        vis_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads(in_json.read_text())
    imgid2anns = {}
    for ann in coco["annotations"]:
        imgid2anns.setdefault(ann["image_id"], []).append(ann)

    predictor = load_predictor(SAM_CFG, SAM_CKPT, DEVICE)

    for img_info in tqdm(coco["images"], desc="SAM-2 segm"):
        anns = imgid2anns.get(img_info["id"], [])
        if anns and all("segmentation" in a for a in anns):
            continue

        if not anns:
            continue

        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        tmp_dir = img_path.parent / "_sam2_tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_img = tmp_dir / "0000000.jpg"
        shutil.copy(img_path, tmp_img)

        state = predictor.init_state(str(tmp_dir))
        predictor.reset_state(state)

        id2ann = {}
        num_added = 0
        for local_id, ann in enumerate(anns):
            box_xyxy = np.array(coco_bbox_to_xyxy(ann["bbox"]), dtype=np.float32)
            id2ann[local_id] = ann

            if hasattr(predictor, "add_new_box"):
                predictor.add_new_box(state, frame_idx=0,
                                       obj_id=local_id, box=box_xyxy)
                num_added += 1
            elif hasattr(predictor, "add_new_points_or_box"):
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=local_id,
                    box=box_xyxy
                )
                num_added += 1
            else:
                raise AttributeError(
                    "SAM-2 predictor 缺少 add_new_box / add_new_points_or_box 接口"
                )

        if num_added == 0:
            try:
                tmp_img.unlink(); tmp_dir.rmdir()
            except Exception:
                pass
            continue

        raw_masks = [] 
        for _, obj_ids, logits in predictor.propagate_in_video(state):
            for idx, oid in enumerate(obj_ids):
                if oid not in id2ann:
                    continue
                mask = (logits[idx] > 0.5).cpu().numpy().astype(np.uint8)   # <<< 改：阈值 0.5
                raw_masks.append(mask)
                ann  = id2ann[oid]
                ann["segmentation"] = mask_to_polygon(mask, fallback_box=ann["bbox"])
                ann["area"] = int(mask.sum())

        try:
            tmp_img.unlink(); tmp_dir.rmdir()
        except Exception:
            pass

        if VISUALIZE and anns:
            img_bgr = cv2.imread(str(img_path))
            polys  = [poly for a in anns if "segmentation" in a for poly in a["segmentation"]]  # <<< 改：支持多 polygon
            boxes  = [a["bbox"] for a in anns]
            labels = [next(c["name"] for c in coco["categories"] if c["id"] == a["category_id"])  # <<< 改：直接映射
                      for a in anns]
            save_p = vis_dir / f'{Path(img_info["file_name"]).stem}_segm.png'
            show_vis(img_bgr, raw_masks, polys, boxes, labels, save_p)

    out_json.write_text(json.dumps(coco))
    print(f"✅ saved to {out_json}")
    if VISUALIZE:
        print(f"✅ visualisations -> {vis_dir}")

if __name__ == "__main__":
    main()
