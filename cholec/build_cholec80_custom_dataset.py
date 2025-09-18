#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构建 no-time-to-train 可用的自定义数据集（CHolec80）
- 从 /mnt/disk0/haoding/cholec80/annotated_data/videoXX/ws_0 读取:
  - images/*.jpg
  - prompts.json (来自你的点提示标注工具: objects => points/labels/obj_id)
- 用 SAM-2 (Hydra + build_sam2_video_predictor) 将点提示 -> mask -> bbox
- 生成 COCO 风格:
  /mnt/disk0/haoding/no-time-to-train/data/
    ├── annotations/
    │   ├── custom_references.json
    │   ├── custom_targets.json
    │   └── references_visualisations/*.jpg
    └── images/*.jpg   (重命名为 videoXX_0000000.jpg 防止跨视频重名)
"""

import os
import re  # <<< keep
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import torch
import sys
import cv2  # <<< ADDED
from pathlib import Path
import os, tempfile, shutil

TMP_ROOT = Path("/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/_sam2_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


GENERATE_REFERENCES = True   # 是否生成 custom_references.json（refs）
GENERATE_TARGETS    = True    # 是否生成 custom_targets.json（targets）


# ====== 你的数据与输出配置 ======
# videoXX/ws_0/{images,prompts.json}
DATA_ROOT = "/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data"
VIDEOS = "21-30"  # "all" 或 VIDEOS = "3-6"

# 中间产物与导出（图像拷贝/annotations/可视化）全部写到 no-time-to-train 名下
OUTPUT_FOLDER = "/projects/surgical-video-digital-twin/pretrain_params/cwz/no-time-to-train/data"

# ====== 仓库根目录（用于 Hydra 进入仓库再用相对 config_path） ======
REPO_ROOT = "/home/wcheng31/no-time-to-train"

# ====== SAM-2======
# 配置文件仍在代码仓库里（随 REPO_ROOT 迁移）
SAM2_CFG_NAME = "/home/wcheng31/no-time-to-train/cholec/configs/sam2/sam2_hiera_l.yaml"
# 预训练权重统一从 pretrain_params 读取（如 sam2）
SAM2_CKPT = "/projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 类别映射======
CATEGORIES = [
    (1, "Gallbladder"),
    (2, "Left Grasper"),
    (3, "TOP Grasper"),
    (4, "Right Grasper"),
    (5, "Bipolar"),
    (6, "Hook"),
    (7, "Scissors"),
    (8, "Clipper"),
    (9, "Irrigator"),
    (10, "SpecimenBag"),
]

# ========= 导入 & 构建 SAM-2（Hydra 版） =========
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from sam2.build_sam import build_sam2_video_predictor


def setup_hydra():
    """准备 Hydra 的全局状态（进入仓库根目录后，用相对 config_path）。"""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    os.chdir(REPO_ROOT)
    initialize(config_path="configs/sam2", version_base="1.2")


def build_sam2_predictor(cfg_name: str, ckpt_path: str, device: str = "cuda"):
    """构建可用的视频预测器（单帧当作伪视频来跑）"""
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SAM2 checkpoint 不存在: {ckpt_path}")
    cfg_name = Path(cfg_name).name
    setup_hydra()
    predictor = build_sam2_video_predictor(cfg_name, ckpt_path, device=device)
    return predictor


def list_videos(root: Path, spec: str) -> List[str]:
    if spec == "all":
        vids = []
        for p in sorted(root.glob("video*/ws_0/prompts.json")):
            vids.append(p.parent.parent.name.replace("video", ""))  # "video01" -> "01"
        return vids
    elif "-" in spec:
        try:
            start, end = map(int, spec.strip().split("-"))
            vids = [f"{i:02d}" for i in range(start, end + 1)]
            return vids
        except Exception:
            raise ValueError(f"VIDEOS 设置错误，应为 'all' 或 'start-end'，你设置的是：{spec}")
    else:
        raise ValueError(f"VIDEOS 设置错误，应为 'all' 或 'start-end'，你设置的是：{spec}")


def safe_filename(video_id: str, frame_name: str) -> str:
    """避免跨视频重名：video01_0000000.jpg"""
    return f"video{video_id}_{frame_name}"


def make_image_id(video_id: str, frame_stem: str) -> int:
    """
    生成稳定唯一的 COCO 图像 ID：
    用 2位视频号 + 7位帧号 组成整数，例如 video '01' & frame '0000000' -> 010000000 (int)
    """
    v = int(video_id)
    f = int(frame_stem)
    return int(f"{v:02d}{f:07d}")


def mask_to_bbox(mask: np.ndarray) -> Tuple[List[float], float]:
    """二值 mask -> COCO bbox [x,y,w,h] 与 area"""
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)
        else:
            mask = np.squeeze(mask, axis=-1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0], 0.0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
    area = float(mask.sum())
    return bbox, area


# <<< ADDED: 二值掩码 -> polygon
def mask_to_polygon(mask: np.ndarray):
    """二值掩码 -> COCO polygon（可能多个外轮廓）"""
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0) if mask.shape[0] == 1 else np.squeeze(mask, axis=-1)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segm = []
    for c in contours:
        c = c.flatten().tolist()
        if len(c) >= 6:  # 至少三点
            segm.append(c)
    return segm
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def ensure_dirs(out_root: Path):
    images_dir = out_root / "images"
    ann_dir = out_root / "annotations"
    vis_dir = ann_dir / "references_visualisations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, ann_dir, vis_dir


def draw_and_save_vis(image_path: Path, anns: List[Dict], save_path: Path):
    im = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for ann in anns:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ========= 核心：用你的 SAM-2 接口把 points/labels -> mask =========
def points_to_best_mask(predictor, img_path: Path, objects: List[Dict]) -> Dict[int, np.ndarray]:
    """
    用 Hydra 版 SAM-2 视频预测器跑单帧：
    - 把单张图复制到一个临时“伪视频”目录，命名 0000000.jpg
    - 对每个 object 调 add_new_points[_or_box]
    - propagate_in_video() 后拿每个 obj_id 的 mask（取 logits>0） 
    返回：{obj_id: mask(H,W)}
    """
    import inspect
    masks_dict: Dict[int, np.ndarray] = {}

    # 1) 在可写目录里创建临时“伪视频”目录（避免只读 datasets 报 PermissionError）
    try:
        tmp_ctx = tempfile.TemporaryDirectory(dir=str(TMP_ROOT))
    except Exception:
        tmp_ctx = tempfile.TemporaryDirectory()  # 回退到 /tmp

    with tmp_ctx as tdir:
        tdir_path = Path(tdir)
        tmp_img = tdir_path / "0000000.jpg"
        shutil.copy(img_path, tmp_img)

        # 2) 兼容不同版本的 SAM-2 init_state 签名
        #    有的版本需要 video_path，有的需要 img_paths，有的两者都要
        try:
            sig = inspect.signature(predictor.init_state)
            params = sig.parameters
            if "video_path" in params and "img_paths" in params and params["img_paths"].default is inspect._empty:
                # 需要两个都提供
                state = predictor.init_state(video_path=str(tdir_path), img_paths=[str(tmp_img)])
            elif "video_path" in params:
                state = predictor.init_state(video_path=str(tdir_path))
            elif "img_paths" in params:
                state = predictor.init_state(img_paths=[str(tmp_img)])
            else:
                # 最后兜底（老版本可能只接收一个位置参数）
                state = predictor.init_state(str(tdir_path))
        except TypeError:
            # 遇到“missing img_paths”这类 TypeError，再次尝试传两个参数
            state = predictor.init_state(video_path=str(tdir_path), img_paths=[str(tmp_img)])

        # 有的实现提供 reset_state，有的不需要；这里尽量兼容
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)

        # 3) 添点（points）或 box（依据 API 提供）
        for obj in objects:
            pts = np.array(obj["points"], dtype=np.float32)
            labs = np.array(obj["labels"], dtype=np.int32)

            if hasattr(predictor, "add_new_points_or_box"):
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(obj["obj_id"]),
                    points=pts,
                    labels=labs
                )
            elif hasattr(predictor, "add_new_points"):
                predictor.add_new_points(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=int(obj["obj_id"]),
                    points=pts,
                    labels=labs
                )
            else:
                raise AttributeError(
                    "Predictor has neither 'add_new_points_or_box' nor 'add_new_points'."
                )

        # 4) 推理并取出二值掩码
        for _, obj_ids, logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(obj_ids):
                m = (logits[i] > 0.5).detach().cpu().numpy().astype(np.uint8)
                masks_dict[int(oid)] = m

    return masks_dict




def main():
    data_root = Path(DATA_ROOT)
    out_root = Path(OUTPUT_FOLDER)
    images_dir, ann_dir, vis_dir = ensure_dirs(out_root)

    old_refs_path = ann_dir / "custom_references.json"
    old_tars_path = ann_dir / "custom_targets.json"
    old_refs = {"images": [], "annotations": [], "categories": []}
    old_tars = {"images": [], "annotations": [], "categories": []}
    ref_next_ann_id = 1
    tar_next_ann_id = 1
    if old_refs_path.exists():
        old_refs = json.loads(old_refs_path.read_text())
        ref_next_ann_id = max([a["id"] for a in old_refs.get("annotations", [])] + [0]) + 1
    if old_tars_path.exists():
        old_tars = json.loads(old_tars_path.read_text())
        tar_next_ann_id = max([a["id"] for a in old_tars.get("annotations", [])] + [0]) + 1

    predictor = build_sam2_predictor(SAM2_CFG_NAME, SAM2_CKPT, DEVICE)
    categories = [{"id": cid, "name": name} for cid, name in CATEGORIES]
    cat_id_to_name = dict(CATEGORIES)

    ref_images, ref_annotations = [], []
    tar_images = []
    tar_annotations = []
    used_category_ids = set()

    videos = list_videos(data_root, VIDEOS)

    vis_base_count = {cid: 0 for cid, _ in CATEGORIES}
    vis_count_per_class = {cid: 0 for cid, _ in CATEGORIES}

    for vid in videos:
        vdir = data_root / f"video{vid}" / "ws_0"
        prompts_path = vdir / "prompts.json"
        frames_dir = vdir / "images"
        if not prompts_path.exists() or not frames_dir.exists():
            continue

        keyframes = json.loads(prompts_path.read_text())

        if GENERATE_REFERENCES:
            for kf in tqdm(keyframes, desc=f"video{vid} refs"):
                frame_file = kf["frame_file"]
                frame_stem = Path(frame_file).stem
                src_img_path = frames_dir / frame_file
                dst_name = safe_filename(vid, frame_file)
                dst_img_path = images_dir / dst_name
                if not dst_img_path.exists():
                    shutil.copyfile(src_img_path, dst_img_path)
                image_id = make_image_id(vid, frame_stem)

                pil = Image.open(src_img_path).convert("RGB")
                w, h = pil.size

                ref_images.append({
                    "id": image_id,
                    "file_name": dst_name,
                    "height": h,
                    "width": w
                })

                objects = kf.get("objects", [])
                if not objects:
                    continue

                masks_dict = points_to_best_mask(predictor, src_img_path, objects)
                vis_anns_this_image = []

                for obj in objects:
                    oid = int(obj["obj_id"])
                    cat_id = oid
                    mask = masks_dict.get(oid, None)
                    if mask is None or mask.sum() == 0:
                        continue
                    bbox, area = mask_to_bbox(mask)
                    ref_annotations.append({
                        "id": ref_next_ann_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": float(area),
                        "iscrowd": 0,
                        "segmentation": mask_to_polygon(mask)  # <<< ADDED: refs 也写 polygon
                    })
                    ref_next_ann_id += 1
                    vis_anns_this_image.append({"bbox": bbox})
                    used_category_ids.add(cat_id)

                if vis_anns_this_image:
                    first_cat = int(objects[0]["obj_id"])
                    vis_count_per_class[first_cat] += 1
                    class_name = cat_id_to_name[first_cat]
                    idx = vis_base_count[first_cat] + vis_count_per_class[first_cat]
                    vis_name = f"{class_name}_{idx}.jpg"
                    draw_and_save_vis(dst_img_path, vis_anns_this_image, vis_dir / vis_name)

    if GENERATE_TARGETS:
        for vid in videos:
            vdir = data_root / f"video{vid}" / "ws_0"
            frames_dir = vdir / "images"
            prompts_path = vdir / "prompts.json"
            if not frames_dir.exists():
                continue

            frame2objs = {}
            if prompts_path.exists():
                _keyframes = json.loads(prompts_path.read_text())
                frame2objs = {kf["frame_file"]: kf.get("objects", []) for kf in _keyframes}

            for f in sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem)):
                frame_stem = f.stem
                dst_name = safe_filename(vid, f.name)
                dst_img_path = images_dir / dst_name
                if not dst_img_path.exists():
                    shutil.copyfile(f, dst_img_path)
                pil = Image.open(f).convert("RGB")
                w, h = pil.size
                image_id = make_image_id(vid, frame_stem)

                tar_images.append({
                    "id": image_id,
                    "file_name": dst_name,
                    "height": h,
                    "width": w
                })

                objects = frame2objs.get(f.name, [])
                if objects:
                    masks_dict = points_to_best_mask(predictor, f, objects)
                    for obj in objects:
                        cid = int(obj["obj_id"])
                        m = masks_dict.get(cid, None)
                        if m is None or m.sum() == 0:
                            continue
                        bbox, area = mask_to_bbox(m)
                        tar_annotations.append({
                            "id": tar_next_ann_id,
                            "image_id": image_id,
                            "category_id": cid,
                            "bbox": bbox,
                            "area": float(area),
                            "iscrowd": 0,
                            "segmentation": mask_to_polygon(m)
                        })
                        tar_next_ann_id += 1
                        used_category_ids.add(cid)

    new_ref_json = {
        "images": ref_images,
        "annotations": ref_annotations,
        "categories": [{"id": cid, "name": name} for cid, name in CATEGORIES]  # <<< CHANGED: 固定写满 10 类
    }
    new_tar_json = {
        "images": tar_images,
        "annotations": tar_annotations,
        "categories": [{"id": cid, "name": name} for cid, name in CATEGORIES]
    }

    merged_refs = {
        "images": old_refs.get("images", []) + new_ref_json["images"],
        "annotations": old_refs.get("annotations", []) + new_ref_json["annotations"],
        "categories": new_ref_json["categories"]
    }
    merged_tars = {
        "images": old_tars.get("images", []) + new_tar_json["images"],
        "annotations": old_tars.get("annotations", []) + new_tar_json["annotations"],
        "categories": new_tar_json["categories"]
    }

    (ann_dir / "custom_references.json").write_text(json.dumps(merged_refs))
    (ann_dir / "custom_targets.json").write_text(json.dumps(merged_tars))

    print("✅ Saved annotations to", ann_dir / "custom_references.json")
    print("✅ Saved target metadata to", ann_dir / "custom_targets.json")
    print("✅ Visualisations in", vis_dir)
    print("✅ Images in", images_dir)


if __name__ == "__main__":
    main()
