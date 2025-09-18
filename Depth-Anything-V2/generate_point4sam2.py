#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth-Anything v2 depth_viz → ≤9 regions, each with ≤3 edge/gray/morph positive points
2025-08-08  by WZC

输入：
  --img_dir  原始 RGB 图（用于灰度边缘/角点与可视化）
  --viz_dir  Depth-Anything v2 生成的 *_depth_viz.jpg 以及（可选）*_depth16.png
输出：
  --out_dir  每帧一个 *_objects.json（SAM2点提示），以及可选的点可视化

JSON 结构：
{
  "frame_id": N,
  "frame_file": "xxxx.jpg",
  "objects": [
    {"obj_id":1, "points":[[x,y],...], "labels":[1,1,...]},
    ...
  ]
}
"""

import os, argparse, json, cv2, numpy as np
from skimage import morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# -------------------- 工具函数 --------------------
def inner_circle_mask(h, w, border_frac=0.06):
    """生成内窥镜内圈掩膜（排除四周一圈）。border_frac 越大，排除越多。"""
    yy, xx = np.mgrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    r  = 0.5 * min(h, w)
    r_in = r * (1.0 - border_frac)
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r_in ** 2)
    return mask.astype(np.uint8)

def attn_from_viz(viz_bgr, use_grad=True, grad_alpha=0.35):
    """把 depth_viz(图)转成 0-1 的注意力图：
       1) 红-蓝差（红越高）  2) 可选融合深度梯度（Sobel）"""
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
    """
    读取 *_depth16.png 并归一化到 0-1；若不存在，则用 depth_viz 的灰度作替代。
    """
    p16_png = os.path.join(viz_dir, f"{base}_depth16.png")
    arr = None
    if os.path.exists(p16_png):
        arr = cv2.imread(p16_png, cv2.IMREAD_UNCHANGED)
    if arr is not None and arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    if arr is not None:
        arr = arr.astype(np.float32)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # fallback：把 viz 转灰度
    gray = cv2.cvtColor(fallback_viz_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray

# -------------------- 阈值 & 形态学 ----------------------------
def adaptive_thresholding(attn):
    # 更低阈值：对非零区域取 60 分位；若全 0，回退 0.0
    valid = attn[attn > 0]
    thr = np.percentile(valid, 60.0) if valid.size else 0.0
    return (attn >= thr).astype(np.uint8)

def morphological_enhancement(mask):
    filled = morphology.remove_small_holes(mask.astype(bool), area_threshold=48)
    ker    = np.ones((5, 5), np.uint8)            # 更小核，减少粘连
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, ker, 1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  ker, 1)
    return opened.astype(np.uint8)

# -------------------- 过大区域的分水岭切割 ----------------------
def split_regions_watershed(attn, mask, min_peak_dist=18, peak_pct=75):
    """
    用分水岭把过度连在一起的高响应区域切开：
    - 以 attn 的局部峰值(≥peak_pct 分位)做标记
    - 在 -attn 上做 watershed，限制在 mask 内
    """
    peaks = peak_local_max(attn, min_distance=min_peak_dist,
                           threshold_abs=np.percentile(attn, peak_pct),
                           labels=mask, exclude_border=False)
    if peaks.size == 0:
        return mask
    markers = np.zeros(attn.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = i
    labels = watershed(-attn, markers=markers, mask=mask)
    new_mask = (labels > 0).astype(np.uint8)
    return new_mask

# -------------------- 连通域 & NMS -----------------------------
def connected_regions(mask, attn, min_area=80):  # 放宽面积
    items = []
    labels = measure.label(mask)
    for r in measure.regionprops(labels):
        if r.area < min_area:
            continue
        minr, minc, maxr, maxc = r.bbox
        score = float(attn[r.coords[:, 0], r.coords[:, 1]].mean())
        aspect = 1.0
        if r.minor_axis_length > 1e-6:
            aspect = float(r.major_axis_length / r.minor_axis_length)
        ecc = getattr(r, "eccentricity", 0.0)
        items.append({
            "box": [int(minc), int(minr), int(maxc), int(maxr)],
            "score": score,
            "aspect": aspect,
            "ecc": float(ecc)
        })
    items.sort(key=lambda x: (x["aspect"] < 2.0, -x["score"]))  # 细长优先
    return items

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1); area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1.0, (area_a + area_b - inter))

def nms_regions(items, thr=0.3, topk=9):
    kept = []
    for it in items:
        if all(iou(it["box"], ref["box"]) < thr for ref in kept):
            kept.append(it)
        if len(kept) == topk:
            break
    return kept

# -------------------- 区域内：多线索采点（边缘/角点 + 灰度 + 骨架） ----
def sample_points_multi(attn_norm, gray_norm, inner_mask, box,
                        pos_max=3, min_dist=8,
                        canny_low=30, canny_high=120, shi_quality=0.01):
    x0, y0, x1, y1 = box
    roi_attn = attn_norm[y0:y1, x0:x1]
    roi_gray = gray_norm[y0:y1, x0:x1]
    roi_msk  = inner_mask[y0:y1, x0:x1]
    H, W = roi_attn.shape

    # --- 候选 1：Canny 边缘 ---
    gray_u8 = (roi_gray * 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, canny_low, canny_high)
    cand = []
    ys, xs = np.nonzero(edges)
    if ys.size:
        cand.append(np.stack([xs, ys], 1))

    # --- 候选 2：Shi-Tomasi 角点 ---
    corners = cv2.goodFeaturesToTrack(gray_u8, 150, shi_quality, 5)
    if corners is not None:
        cs = corners.squeeze(1).astype(int)
        cand.append(cs[:, [0, 1]])  # x,y

    # --- 候选 3：灰度亮点（高分位）---
    if roi_msk.sum() > 0:
        thr_g = np.percentile(roi_gray[roi_msk > 0], 75.0)
    else:
        thr_g = np.percentile(roi_gray, 75.0)
    bright = (roi_gray >= thr_g).astype(np.uint8) * roi_msk
    by, bx = np.nonzero(bright)
    if by.size:
        cand.append(np.stack([bx, by], 1))

    # --- 候选 4：骨架中心线（形态学，偏细长）---
    # 以 (attn 高 or gray 高) 作为工具候选，再 skeletonize
    blob = ((roi_attn >= np.percentile(roi_attn[roi_msk > 0], 70.0) if roi_msk.sum() else np.percentile(roi_attn, 70.0)) |
            (roi_gray >= thr_g)).astype(np.uint8) * roi_msk
    if blob.any():
        skel = morphology.skeletonize(blob.astype(bool)).astype(np.uint8)
        sy, sx = np.nonzero(skel)
        if sy.size:
            cand.append(np.stack([sx, sy], 1))
    else:
        skel = np.zeros_like(blob, dtype=np.uint8)

    # 汇总候选并限于内圈
    if len(cand):
        coords = np.concatenate(cand, 0)
        coords = np.unique(coords, axis=0)
        coords = coords[roi_msk[coords[:, 1], coords[:, 0]] > 0]
    else:
        coords = np.empty((0, 2), dtype=int)

    pos = []

    # --- 按复合分数排序：0.6*attn + 0.4*gray + 0.2*skel_bonus ---
    if coords.size > 0:
        a = roi_attn[coords[:, 1], coords[:, 0]]
        g = roi_gray[coords[:, 1], coords[:, 0]]
        k = skel[coords[:, 1], coords[:, 0]].astype(np.float32)
        score = 0.6 * a + 0.4 * g + 0.2 * k
        order = np.argsort(score)[::-1]
        for idx in order:
            if len(pos) == pos_max: break
            cx, cy = int(coords[idx, 0]), int(coords[idx, 1])
            gx, gy = x0 + cx, y0 + cy
            if all(abs(gx - px) + abs(gy - py) >= min_dist for px, py in pos):
                pos.append((gx, gy))

    # --- 回退：直接在复合图上取峰值 ---
    if len(pos) < pos_max:
        comp = 0.6 * roi_attn + 0.4 * roi_gray + 0.2 * skel.astype(np.float32)
        flat = np.argsort(comp.flatten())[::-1]
        for ind in flat:
            if len(pos) == pos_max: break
            yy, xx = divmod(ind, W)
            if roi_msk[yy, xx] == 0: continue
            gx, gy = x0 + xx, y0 + yy
            if all(abs(gx - px) + abs(gy - py) >= min_dist for px, py in pos):
                pos.append((gx, gy))
    return pos

# -------------------- 主批处理 --------------------
def process_images(img_dir, viz_dir, out_dir,
                   max_regions=9, pos_max=3,
                   border_frac=0.06, iou_thr=0.3):
    os.makedirs(out_dir, exist_ok=True)
    frames = [f for f in os.listdir(img_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

    frame_id = 0
    for fn in sorted(frames):
        base = os.path.splitext(fn)[0]
        img_path = os.path.join(img_dir, fn)
        viz_path = os.path.join(viz_dir, f"{base}_depth_viz.jpg")
        if not os.path.exists(viz_path):
            viz_path = os.path.join(viz_dir, f"{base}_depth_viz.png")
            if not os.path.exists(viz_path):
                print("skip(no viz):", fn)
                continue

        rgb  = cv2.imread(img_path)
        viz  = cv2.imread(viz_path)
        if rgb is None or viz is None:
            print("skip(load fail):", fn)
            continue

        H, W = viz.shape[:2]
        inner = inner_circle_mask(H, W, border_frac=border_frac)
        gray_depth = load_depth_gray_norm(base, viz_dir, viz)  # 0-1

        # 1) viz → attn，并把外圈设为 0
        attn = attn_from_viz(viz, use_grad=True, grad_alpha=0.35)
        attn = attn * inner
        gray_depth = gray_depth * inner

        # 2) 连通域 + 分水岭细分
        mask = morphological_enhancement(adaptive_thresholding(attn)) * inner
        mask = split_regions_watershed(attn, mask, min_peak_dist=18, peak_pct=75)

        regions = connected_regions(mask, attn, min_area=80)
        regions = nms_regions(regions, thr=iou_thr, topk=max_regions)

        # 3) 区域内取点（多线索）
        objects = []
        for ridx, reg in enumerate(regions, 1):
            pts = sample_points_multi(attn, gray_depth, inner, reg["box"],
                                      pos_max=pos_max, min_dist=8)
            objects.append({
                "obj_id": int(ridx),
                "points": [[int(x), int(y)] for (x, y) in pts],
                "labels": [1] * len(pts)
            })

        # 4) 写 JSON
        frame_id += 1
        out_json = {
            "frame_id": int(frame_id),
            "frame_file": fn,
            "objects": objects
        }
        with open(os.path.join(out_dir, f"{base}_objects.json"), "w") as fp:
            json.dump(out_json, fp, indent=2)

        # 5) 可视化（可选）
        vis = rgb.copy()
        cmap = [(255, 0, 0), (0, 255, 0), (0, 255, 255),
                (255, 255, 0), (255, 0, 255), (0, 128, 255),
                (128, 0, 255), (255, 128, 0), (0, 255, 128)]
        for o in objects:
            col = cmap[(o["obj_id"] - 1) % len(cmap)]
            for (x, y) in o["points"]:
                cv2.circle(vis, (x, y), 5, col, -1)
        cv2.imwrite(os.path.join(out_dir, f"{base}_objects.jpg"), vis)

        print(f"[✓] {fn}: {len(objects)} regions, "
              f"{sum(len(o['points']) for o in objects)} points")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser("Depth-viz → region-aware points for SAM2")
    ap.add_argument('--img_dir',   default='/home/haoding/Wenzheng/dinov2/figures')
    ap.add_argument('--viz_dir',   default='/home/haoding/Wenzheng/dinov2/figures_attn_depthv2')
    ap.add_argument('--out_dir',   default='/home/haoding/Wenzheng/dinov2/figures_depthv2_points')
    ap.add_argument('--max_regions', type=int, default=9)
    ap.add_argument('--pos_max',     type=int, default=3)
    ap.add_argument('--border_frac', type=float, default=0.06, help='内圈半径的外缩比例，排除外环')
    ap.add_argument('--iou_thr',     type=float, default=0.30)
    args = ap.parse_args()

    process_images(args.img_dir, args.viz_dir, args.out_dir,
                   max_regions=args.max_regions, pos_max=args.pos_max,
                   border_frac=args.border_frac, iou_thr=args.iou_thr)

if __name__ == "__main__":
    main()
