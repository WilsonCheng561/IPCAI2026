#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Depth-Anything v2 & DINO point prompts (strictly keep both sampling algorithms unchanged)
2025-08-12  by WZC

新增（最小增量）：
- --prompts_json：仅处理该 JSON 中出现的 frame_file 帧
- 其余算法/参数保持与之前一致

输入：
  --img_dir    原图目录（只用于灰度、可视化尺寸等）
  --depth_dir  Depth-Anything v2 的 *_depth_viz.jpg / *_depth16.png 所在目录
  --dino_dir   DINO *_dino_attn.jpg 所在目录
输出：
  --out_dir    合并后的 *_objects.json（obj_id = depth 后接 dino，并顺延累加）

合并规则：
- 每张图先用 depth 算法取点；再用 dino 算法取点；
- dino 的 obj_id 在 depth 的数量基础上累加偏移；
- labels 均为 1（正样本），不做类别区分。
"""

import os, re, json, argparse
import numpy as np
import cv2
from skimage import morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# ======================= 工具：从 prompts.json 取帧名 =======================
def _extract_frame_files_from_text(txt: str):
    # 兜底：正则提取 "frame_file": "xxx.jpg"
    return list({m.group(1) for m in re.finditer(r'"frame_file"\s*:\s*"([^"]+)"', txt)})

def load_frame_list_from_prompts(prompts_json: str):
    """
    返回 set[str]：prompts.json 里出现过的 frame_file
    兼容：list[dict] / {frames:[...]} / 任意文本里含 "frame_file":"xxx"
    """
    with open(prompts_json, "r") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {d["frame_file"] for d in data if isinstance(d, dict) and "frame_file" in d}
        if isinstance(data, dict):
            if "frames" in data and isinstance(data["frames"], list):
                return {d["frame_file"] for d in data["frames"] if isinstance(d, dict) and "frame_file" in d}
            # 常见直接平铺
            if "frame_file" in data:
                return {data["frame_file"]}
            # 兜底 regex
            return set(_extract_frame_files_from_text(raw))
        # 兜底 regex
        return set(_extract_frame_files_from_text(raw))
    except Exception:
        return set(_extract_frame_files_from_text(raw))

# ======================= Depth-Anything v2 取点算法（保持不变） =======================
def inner_circle_mask(h, w, border_frac=0.06):
    yy, xx = np.mgrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    r  = 0.5 * min(h, w)
    r_in = r * (1.0 - border_frac)
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= (r_in ** 2)).astype(np.uint8)

def attn_from_viz(viz_bgr, use_grad=True, grad_alpha=0.35):
    red  = viz_bgr[:, :, 2].astype(np.float32) / 255.0
    blue = viz_bgr[:, :, 0].astype(np.float32) / 255.0
    attn = np.clip(red - blue, 0.0, 1.0)
    if use_grad:
        gray = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        attn = np.clip((1 - grad_alpha) * attn + grad_alpha * grad, 0.0, 1.0)
    return attn

def load_depth_gray_norm(base, viz_dir, fallback_viz_bgr):
    p16_png = os.path.join(viz_dir, f"{base}_depth16.png")
    arr = None
    if os.path.exists(p16_png):
        arr = cv2.imread(p16_png, cv2.IMREAD_UNCHANGED)
    if arr is not None and arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    if arr is not None:
        arr = arr.astype(np.float32)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    gray = cv2.cvtColor(fallback_viz_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray

def adaptive_thresholding_depth(attn):
    valid = attn[attn > 0]
    thr = np.percentile(valid, 60.0) if valid.size else 0.0
    return (attn >= thr).astype(np.uint8)

def morphological_enhancement_depth(mask):
    filled = morphology.remove_small_holes(mask.astype(bool), area_threshold=48)
    ker    = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, ker, 1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  ker, 1)
    return opened.astype(np.uint8)

def split_regions_watershed(attn, mask, min_peak_dist=18, peak_pct=75):
    peaks = peak_local_max(attn, min_distance=min_peak_dist,
                           threshold_abs=np.percentile(attn, peak_pct),
                           labels=mask, exclude_border=False)
    if peaks.size == 0:
        return mask
    markers = np.zeros(attn.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i
    labels = watershed(-attn, markers=markers, mask=mask)
    return (labels > 0).astype(np.uint8)

def connected_regions_depth(mask, attn, min_area=80):
    items = []
    labels = measure.label(mask)
    for r in measure.regionprops(labels):
        if r.area < min_area: continue
        minr,minc,maxr,maxc = r.bbox
        score = float(attn[r.coords[:, 0], r.coords[:, 1]].mean())
        aspect = 1.0
        if r.minor_axis_length > 1e-6:
            aspect = float(r.major_axis_length / r.minor_axis_length)
        ecc = getattr(r, "eccentricity", 0.0)
        items.append({"box":[int(minc),int(minr),int(maxc),int(maxr)],
                      "score":score,"aspect":aspect,"ecc":float(ecc)})
    items.sort(key=lambda x: (x["aspect"] < 2.0, -x["score"]))
    return items

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter = max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    if inter <= 0: return 0.0
    area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
    return inter/max(1.0,(area_a+area_b-inter))

def nms_regions(items, thr=0.3, topk=9):
    kept=[]
    for it in items:
        if all(iou(it["box"], r["box"]) < thr for r in kept):
            kept.append(it)
        if len(kept) == topk: break
    return kept

def sample_points_multi(attn_norm, gray_norm, inner_mask, box,
                        pos_max=3, min_dist=8,
                        canny_low=30, canny_high=120, shi_quality=0.01):
    x0,y0,x1,y1 = box
    roi_attn = attn_norm[y0:y1, x0:x1]
    roi_gray = gray_norm[y0:y1, x0:x1]
    roi_msk  = inner_mask[y0:y1, x0:x1]
    H,W = roi_attn.shape

    gray_u8 = (roi_gray * 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, canny_low, canny_high)
    cand = []
    ys, xs = np.nonzero(edges)
    if ys.size: cand.append(np.stack([xs, ys], 1))

    corners = cv2.goodFeaturesToTrack(gray_u8, 150, shi_quality, 5)
    if corners is not None:
        cs = corners.squeeze(1).astype(int)
        cand.append(cs[:, [0,1]])

    thr_g = np.percentile(roi_gray[roi_msk>0], 75.0) if roi_msk.sum() else np.percentile(roi_gray, 75.0)
    bright = (roi_gray >= thr_g).astype(np.uint8) * roi_msk
    by, bx = np.nonzero(bright)
    if by.size: cand.append(np.stack([bx, by], 1))

    thr_a = np.percentile(roi_attn[roi_msk>0], 70.0) if roi_msk.sum() else np.percentile(roi_attn, 70.0)
    blob = ((roi_attn >= thr_a) | (roi_gray >= thr_g)).astype(np.uint8) * roi_msk
    skel = morphology.skeletonize(blob.astype(bool)).astype(np.uint8) if blob.any() else np.zeros_like(blob)
    sy, sx = np.nonzero(skel)
    if sy.size: cand.append(np.stack([sx, sy], 1))

    if len(cand):
        coords = np.unique(np.concatenate(cand,0), axis=0)
        coords = coords[roi_msk[coords[:,1], coords[:,0]] > 0]
    else:
        coords = np.empty((0,2), dtype=int)

    pos=[]
    if coords.size>0:
        a = roi_attn[coords[:,1], coords[:,0]]
        g = roi_gray[coords[:,1], coords[:,0]]
        k = skel[coords[:,1], coords[:,0]].astype(np.float32)
        score = 0.6*a + 0.4*g + 0.2*k
        order = np.argsort(score)[::-1]
        for idx in order:
            if len(pos)==pos_max: break
            cx,cy = int(coords[idx,0]), int(coords[idx,1])
            gx,gy = x0+cx, y0+cy
            if all(abs(gx-px)+abs(gy-py) >= min_dist for px,py in pos):
                pos.append((gx,gy))
    if len(pos) < pos_max:
        comp = 0.6*roi_attn + 0.4*roi_gray + 0.2*skel.astype(np.float32)
        flat = np.argsort(comp.flatten())[::-1]
        for ind in flat:
            if len(pos)==pos_max: break
            yy, xx = divmod(ind, W)
            if roi_msk[yy,xx]==0: continue
            gx,gy = x0+xx, y0+yy
            if all(abs(gx-px)+abs(gy-py) >= min_dist for px,py in pos):
                pos.append((gx,gy))
    return pos

# ======================= DINO 取点算法（保持不变） =======================
def adaptive_thresholding_dino(attn):
    thr = attn.mean() + 0.2 * attn.std()
    return (attn >= thr).astype(np.uint8)

def morphological_enhancement_dino(mask):
    filled = morphology.remove_small_holes(mask.astype(bool), area_threshold=64)
    ker    = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, ker, 1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  ker, 1)
    return opened.astype(np.uint8)

def connected_regions_dino(mask, attn, min_area=150):
    items=[]
    for r in measure.regionprops(measure.label(mask)):
        if r.area < min_area: continue
        minr,minc,maxr,maxc = r.bbox
        score = float(attn[r.coords[:,0], r.coords[:,1]].mean())
        items.append({'box':[minc,minr,maxc,maxr],'score':score})
    items.sort(key=lambda x:x['score'], reverse=True)
    return items

def nms_regions_dino(items, thr=0.3, topk=9):
    kept=[]
    for it in items:
        if all(iou(it['box'], r['box']) < thr for r in kept):
            kept.append(it)
        if len(kept)==topk: break
    return kept

def sample_points_edge(attn_norm, gray, box,
                       pos_max=3, min_dist=10,
                       canny=(50,150), quality=0.01):
    x0,y0,x1,y1 = box
    roi_attn = attn_norm[y0:y1, x0:x1]
    roi_gray = gray[y0:y1, x0:x1]
    edges = cv2.Canny(roi_gray, *canny)
    ys, xs = np.nonzero(edges)
    corners = cv2.goodFeaturesToTrack(roi_gray, 100, quality, 5)
    if corners is not None:
        cs = corners.squeeze(1).astype(int)
        ys  = np.concatenate([ys,  cs[:,1]])
        xs  = np.concatenate([xs,  cs[:,0]])
    coords = np.unique(np.stack([xs,ys],1), axis=0) if xs.size else np.empty((0,2), int)
    pos=[]
    if coords.size>0:
        vals = roi_attn[coords[:,1], coords[:,0]]
        order = np.argsort(vals)[::-1]
        for idx in order:
            if len(pos)==pos_max: break
            cx,cy = coords[idx]; gx,gy = x0+cx, y0+cy
            if all(abs(gx-px)+abs(gy-py) >= min_dist for px,py in pos):
                pos.append((gx,gy))
    if len(pos)<pos_max:
        flat_idx = np.argsort(roi_attn.flatten())[::-1]
        h,w = roi_attn.shape
        for idx in flat_idx:
            if len(pos)==pos_max: break
            cy,cx = divmod(idx,w)
            gx,gy = x0+cx, y0+cy
            if all(abs(gx-px)+abs(gy-py) >= min_dist for px,py in pos):
                pos.append((gx,gy))
    return pos

# ======================= 合并流程 =======================
def process_one_image_depth(img_bgr, viz_bgr, base,
                            max_regions=9, pos_max=3,
                            border_frac=0.06, iou_thr=0.3):
    H, W = viz_bgr.shape[:2]
    inner = inner_circle_mask(H, W, border_frac=border_frac)

    # ✅ 修复：删除对未定义 viz_bgr_path 的调用，只保留 depth_dir_global 版本
    gray_depth = load_depth_gray_norm(base, depth_dir_global, viz_bgr)

    attn = attn_from_viz(viz_bgr, use_grad=True, grad_alpha=0.35) * inner
    gray_depth = gray_depth * inner

    mask = morphological_enhancement_depth(adaptive_thresholding_depth(attn)) * inner
    mask = split_regions_watershed(attn, mask, min_peak_dist=18, peak_pct=75)
    regions = connected_regions_depth(mask, attn, min_area=80)
    regions = nms_regions(regions, thr=iou_thr, topk=max_regions)

    objects=[]
    for ridx, reg in enumerate(regions, 1):
        pts = sample_points_multi(attn, gray_depth, inner, reg["box"],
                                  pos_max=pos_max, min_dist=8)
        objects.append({"obj_id": int(ridx),
                        "points": [[int(x), int(y)] for (x,y) in pts],
                        "labels": [1]*len(pts)})
    return objects


def process_one_image_dino(img_bgr, attn_bgr, max_regions=9, pos_max=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    red  = attn_bgr[:,:,2].astype(np.float32)/255.
    blue = attn_bgr[:,:,0].astype(np.float32)/255.
    attn_norm = np.clip(red-blue, 0, 1)
    mask = morphological_enhancement_dino(adaptive_thresholding_dino(attn_norm))
    regions = nms_regions_dino(connected_regions_dino(mask, attn_norm),
                               thr=0.3, topk=max_regions)
    objects=[]
    for ridx, reg in enumerate(regions, 1):
        pts = sample_points_edge(attn_norm, gray, reg['box'], pos_max=pos_max)
        objects.append({"obj_id": int(ridx),
                        "points": [[int(x), int(y)] for (x,y) in pts],
                        "labels": [1]*len(pts)})
    return objects

# ======================= 主入口 =======================
def main():
    ap = argparse.ArgumentParser("Merge Depth-Anything-v2 & DINO point prompts")
    ap.add_argument('--img_dir',   default='/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data/video21/ws_0/images')
    ap.add_argument('--depth_dir', default='/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/figures_attn_depthv2')
    ap.add_argument('--dino_dir',  default='/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/figures_attn_dino1')
    ap.add_argument('--out_dir',   default='/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/points')
    ap.add_argument('--max_regions', type=int, default=9)
    ap.add_argument('--pos_max',     type=int, default=3)
    ap.add_argument('--border_frac', type=float, default=0.06)
    ap.add_argument('--iou_thr',     type=float, default=0.30)
    # 新增：仅处理 prompts.json 中的帧
    ap.add_argument('--prompts_json', type=str, default=None,
                    help='若提供，则只处理此文件中出现的 frame_file 帧')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 选择要处理的帧
    if args.prompts_json and os.path.exists(args.prompts_json):
        target_frames = sorted(load_frame_list_from_prompts(args.prompts_json))
    else:
        target_frames = sorted([f for f in os.listdir(args.img_dir)
                                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])

    # 为了 Depth 灰度读取，保存 depth_dir 到全局（最小改动）
    global depth_dir_global
    depth_dir_global = args.depth_dir

    for fn in target_frames:
        img_path = os.path.join(args.img_dir, fn)
        if not os.path.exists(img_path):
            print(f"[skip] image not found: {img_path}")
            continue
        base = os.path.splitext(fn)[0]

        depth_viz_path_jpg = os.path.join(args.depth_dir, f"{base}_depth_viz.jpg")
        depth_viz_path_png = os.path.join(args.depth_dir, f"{base}_depth_viz.png")
        if os.path.exists(depth_viz_path_jpg):
            viz_path = depth_viz_path_jpg
        elif os.path.exists(depth_viz_path_png):
            viz_path = depth_viz_path_png
        else:
            viz_path = None

        dino_path = os.path.join(args.dino_dir, f"{base}_dino_attn.jpg")
        if not os.path.exists(dino_path):
            dino_path = None

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[skip] load fail: {img_path}")
            continue

        depth_objs = []
        if viz_path:
            viz_bgr = cv2.imread(viz_path)
            if viz_bgr is not None:
                # 注意：process_one_image_depth 内部使用的 gray_depth 从 depth_dir_global 读取 *_depth16.png
                depth_objs = process_one_image_depth(img_bgr, viz_bgr, base,
                                                     max_regions=args.max_regions,
                                                     pos_max=args.pos_max,
                                                     border_frac=args.border_frac,
                                                     iou_thr=args.iou_thr)

        dino_objs = []
        if dino_path:
            attn_bgr = cv2.imread(dino_path)
            if attn_bgr is not None:
                dino_objs = process_one_image_dino(img_bgr, attn_bgr,
                                                   max_regions=args.max_regions,
                                                   pos_max=args.pos_max)

        # 合并（obj_id 顺延）
        merged = []
        merged.extend(depth_objs)
        offset = len(depth_objs)
        for o in dino_objs:
            merged.append({
                "obj_id": int(o["obj_id"] + offset),
                "points": o["points"],
                "labels": o["labels"]
            })

        out_json = {
            "frame_id": 0,                # 不变更 frame_id（可按需要自增）
            "frame_file": fn,
            "objects": merged
        }
        with open(os.path.join(args.out_dir, f"{base}_objects.json"), "w") as fp:
            json.dump(out_json, fp, indent=2)

        print(f"[MERGE] {fn}: depth {len(depth_objs)} + dino {len(dino_objs)} → total {len(merged)}")

    print("\nAll done. JSON saved to:", args.out_dir)


if __name__ == "__main__":
    main()
