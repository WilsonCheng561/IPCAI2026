#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO attention → ≤9 connected regions, each with ≤3 edge-aware positive points
2025-08-07  by WZC
"""

import os, argparse, json, cv2, numpy as np
from skimage import morphology, measure

# -------------------- 阈值 & 形态学 ----------------------------
def adaptive_thresholding(attn):
    thr = attn.mean() + 0.2 * attn.std()
    return (attn >= thr).astype(np.uint8)

def morphological_enhancement(mask):
    filled = morphology.remove_small_holes(mask.astype(bool), area_threshold=64)
    ker    = np.ones((7,7), np.uint8)
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, ker, 1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  ker, 1)
    return opened.astype(np.uint8)

# -------------------- 连通域 & NMS -----------------------------
def connected_regions(mask, attn, min_area=150):
    items=[]
    for r in measure.regionprops(measure.label(mask)):
        if r.area < min_area: continue
        minr,minc,maxr,maxc = r.bbox
        score = float(attn[r.coords[:,0], r.coords[:,1]].mean())
        items.append({'box':[minc,minr,maxc,maxr],'score':score})
    items.sort(key=lambda x:x['score'], reverse=True)
    return items

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    inter = max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    if inter==0: return 0.
    area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
    return inter/(area_a+area_b-inter)

def nms_regions(items, thr=0.3, topk=9):
    kept=[]
    for it in items:
        if all(iou(it['box'],r['box'])<thr for r in kept):
            kept.append(it)
        if len(kept)==topk: break
    return kept

# -------------------- 边缘/角点采样 -----------------------------
def sample_points_edge(attn_norm, gray, box,
                       pos_max=3, min_dist=10,
                       canny=(50,150), quality=0.01):
    x0,y0,x1,y1 = box
    roi_attn = attn_norm[y0:y1, x0:x1]
    roi_gray = gray[y0:y1, x0:x1]

    # ① Canny 边缘
    edges = cv2.Canny(roi_gray, *canny)
    ys, xs = np.nonzero(edges)

    # ② Shi-Tomasi 角点
    corners = cv2.goodFeaturesToTrack(roi_gray, 100, quality, 5)
    if corners is not None:
        cs = corners.squeeze(1).astype(int)
        ys  = np.concatenate([ys,  cs[:,1]])
        xs  = np.concatenate([xs,  cs[:,0]])

    # 候选去重
    coords = np.unique(np.stack([xs,ys],1), axis=0)
    if coords.size==0: coords = np.empty((0,2),int)

    # ③ 按 attention 值排序
    vals = roi_attn[coords[:,1], coords[:,0]]
    order = np.argsort(vals)[::-1]
    pos=[]
    for idx in order:
        if len(pos)==pos_max: break
        cx,cy = coords[idx]
        gx,gy = x0+cx, y0+cy
        if all(abs(gx-px)+abs(gy-py) >= min_dist for px,py in pos):
            pos.append((gx,gy))

    # ④ 若不足，回退到最高 attn
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

# -------------------- 主批处理 --------------------------------
def process_images(img_dir, attn_dir, out_dir,
                   max_regions, pos_max):
    os.makedirs(out_dir, exist_ok=True)
    imgs=[f for f in os.listdir(img_dir)
          if f.lower().endswith((".jpg",".png",".bmp",".webp"))]

    frame_id=0
    for fn in imgs:
        base = os.path.splitext(fn)[0]
        img_path  = os.path.join(img_dir, fn)
        attn_path = os.path.join(attn_dir,f"{base}_dino_attn.jpg")
        if not os.path.exists(attn_path): continue

        img  = cv2.imread(img_path)
        attn = cv2.imread(attn_path)
        if img is None or attn is None: continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        red  = attn[:,:,2].astype(np.float32)/255.
        blue = attn[:,:,0].astype(np.float32)/255.
        attn_norm = np.clip(red-blue,0,1)

        mask    = morphological_enhancement(adaptive_thresholding(attn_norm))
        regions = nms_regions(connected_regions(mask,attn_norm),
                              thr=0.3, topk=max_regions)

        objects=[]
        for ridx,reg in enumerate(regions,1):
            pts = sample_points_edge(attn_norm, gray, reg['box'],
                                     pos_max=pos_max)
            objects.append({"obj_id":int(ridx),
                            "points":[[int(x),int(y)] for x,y in pts],
                            "labels":[1]*len(pts)})

        frame_id+=1
        with open(os.path.join(out_dir,f"{base}_objects.json"),"w") as fp:
            json.dump({"frame_id":int(frame_id),
                       "frame_file":fn,
                       "objects":objects}, fp, indent=2)

        print(f"[✓] {fn}: {len(objects)} regions, "
              f"{sum(len(o['points']) for o in objects)} points")

# -------------------- CLI -------------------------------------
def main():
    ap=argparse.ArgumentParser("edge-aware point sampler per bbox")
    ap.add_argument('--original_dir', default='/home/haoding/Wenzheng/dinov2/figures')
    ap.add_argument('--attn_dir',     default='/home/haoding/Wenzheng/dinov2/figures_attn_dino1')
    ap.add_argument('--output_dir',   default='/home/haoding/Wenzheng/dinov2/points')
    ap.add_argument('--max_regions',  type=int, default=9)
    ap.add_argument('--pos_max',      type=int, default=3)
    args=ap.parse_args()

    process_images(args.original_dir, args.attn_dir, args.output_dir,
                   args.max_regions, args.pos_max)

if __name__=="__main__":
    main()
