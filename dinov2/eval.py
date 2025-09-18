#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SAM-2 masks with IoU threshold (class-agnostic):
- GT masks: run SAM-2 using ws_0/prompts.json (points/labels)
- Pred masks: run SAM-2 using ws_0/points/*_objects.json (points/labels)
- Compute P/R via IoU >= threshold using greedy bipartite matching (class-agnostic)

Usage example:
python /home/wcheng31/dinov2/eval.py \
  --root /projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data \
  --v_start 21 --v_end 30 \
  --ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
  --cfg sam2_hiera_l.yaml \
  --iou_thr 0.6 \
  --tmp-root /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/tmp_eval \
  --out-root /projects/surgical-video-digital-twin/pretrain_params/cwz/ours/eval
"""

import os, sys, json, argparse, shutil, tempfile
import numpy as np
import cv2, torch
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize

# --- Make sure SAM2 repo is importable ---
SAM2_REPO_ROOT = "/home/wcheng31/no-time-to-train"
if SAM2_REPO_ROOT not in sys.path:
    sys.path.insert(0, SAM2_REPO_ROOT)

try:
    # prefer the backup builder if present
    from sam2.backup.build_sam import build_sam2_video_predictor
except Exception:
    from sam2.build_sam import build_sam2_video_predictor


# ---------------- Hydra / device ----------------
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

# ---------------- IO helpers ----------------
def load_gt_prompts(prompts_path):
    with open(prompts_path, "r") as f:
        data = json.load(f)
    # 支持 {"frames":[...]} 或 直接列表 / 单帧字典
    frames = {}
    if isinstance(data, dict) and "frames" in data:
        it = data["frames"]
    elif isinstance(data, list):
        it = data
    else:
        it = [data]
    for frm in it:
        fn = frm.get("frame_file") or frm.get("image") or frm.get("img") or ""
        frames[fn] = frm
    return frames

def load_pred_points(points_dir):
    """读取我们生成的 *_objects.json，返回 frame_file -> [{'obj_id', 'points', 'labels'}, ...]"""
    out = {}
    for fn in os.listdir(points_dir):
        if not fn.endswith("_objects.json"):
            continue
        with open(os.path.join(points_dir, fn), "r") as f:
            d = json.load(f)
        frame_file = d.get("frame_file", None)
        if not frame_file:
            frame_file = fn.replace("_objects.json", "") + ".jpg"
        out[frame_file] = d.get("objects", [])
    return out

# ---------------- mask utils ----------------
def to_bin2d(mask) -> np.ndarray:
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

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    A = to_bin2d(a).astype(bool)
    B = to_bin2d(b).astype(bool)
    inter = np.logical_and(A, B).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(A, B).sum()
    return float(inter) / float(max(union, 1))

def greedy_match_iou(pred_masks, gt_masks, iou_thr=0.5):
    """
    pred_masks, gt_masks: list of HxW binary masks
    返回 (TP, FP, FN)
    """
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return 0, 0, 0
    if len(pred_masks) == 0:
        return 0, 0, len(gt_masks)
    if len(gt_masks) == 0:
        return 0, len(pred_masks), 0

    P, G = len(pred_masks), len(gt_masks)
    iou_mat = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            iou_mat[i, j] = mask_iou(pred_masks[i], gt_masks[j])

    # 贪心：按 IoU 从大到小
    pairs = []
    used_p = np.zeros(P, dtype=bool)
    used_g = np.zeros(G, dtype=bool)
    flat_idx = np.argsort(iou_mat, axis=None)[::-1]
    for f in flat_idx:
        p = f // G
        g = f %  G
        if used_p[p] or used_g[g]:
            continue
        if iou_mat[p, g] >= iou_thr:
            used_p[p] = True
            used_g[g] = True
            pairs.append((p, g))

    TP = len(pairs)
    FP = int((~used_p).sum())
    FN = int((~used_g).sum())
    return TP, FP, FN

# ---------------- SAM-2 runner ----------------
def run_sam2_masks_for_objects(predictor, img_path, objects, tmp_dir):
    """
    用一组 objects 点提示生成一批 mask（单帧），返回 list[np.uint8(H,W)]。
    objects: [{"obj_id":int, "points":[[x,y],...], "labels":[0/1,...]}, ...]
    兼容不同版本 SAM2 的 init_state：
      - init_state(video_path=...)
      - init_state(img_paths=[...])
      - init_state(video_path=..., img_paths=[...])  ← 某些版本要求两者同时
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # 准备“单帧视频”资源
    dst_name = "0000000.jpg"
    dst_path = os.path.join(tmp_dir, dst_name)
    shutil.copy(img_path, dst_path)

    # 兼容三种签名（按需降级）
    state = None
    try:
        state = predictor.init_state(video_path=tmp_dir, img_paths=[dst_path])
    except TypeError:
        try:
            state = predictor.init_state(video_path=tmp_dir)
        except TypeError:
            state = predictor.init_state(img_paths=[dst_path])

    predictor.reset_state(state)

    # 加点
    valid_oids = []
    for obj in objects:
        pts = np.array(obj.get("points", []), np.float32)
        labs = np.array(obj.get("labels", []), np.int32)
        if pts.size == 0:
            continue
        if hasattr(predictor, "add_new_points_or_box"):
            predictor.add_new_points_or_box(state, 0, obj["obj_id"], pts, labs)
        elif hasattr(predictor, "add_new_points"):
            predictor.add_new_points(state, 0, obj["obj_id"], pts, labs)
        else:
            raise AttributeError("Predictor lacks add_new_points* methods")
        valid_oids.append(obj["obj_id"])

    # 推理并取 mask（不同版本返回项名不同，这里统一成 score>0 后二值化）
    masks = []
    for out in predictor.propagate_in_video(state):
        # 期望 out 是 (frame_idx, obj_ids, tensor)
        if not isinstance(out, (tuple, list)) or len(out) < 3:
            continue
        _, obj_ids, scores = out  # scores: (B,1,H,W) 或 (B,H,W)

        if isinstance(scores, torch.Tensor):
            for i, oid in enumerate(obj_ids):
                if oid in valid_oids:
                    s = scores[i]
                    if s.ndim == 4 and s.shape[1] == 1:  # (1,H,W)
                        s = s[0]
                    m = (s > 0).detach().cpu().numpy().astype(np.uint8)
                    masks.append(m)

    # 只删临时帧文件（tmp_dir 在外层按视频清理）
    try:
        os.remove(dst_path)
    except Exception:
        pass

    return masks


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser("Evaluate IoU-based P/R (class-agnostic) using SAM-2 on GT/pred points")
    ap.add_argument("--root", default="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data")
    ap.add_argument("--v_start", type=int, default=21)
    ap.add_argument("--v_end",   type=int, default=30)
    ap.add_argument("--ckpt", required=True, help="SAM-2 checkpoint")
    ap.add_argument("--cfg",  required=True, help="SAM-2 config yaml (under configs/sam2)")
    ap.add_argument("--pts_dirname", default="points", help="pred points dir name inside ws_0")
    ap.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for a match")
    # 临时目录（放可写位置），不可写时回落 /tmp
    ap.add_argument("--tmp-root",
                    default="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/tmp_eval",
                    help="a writable directory to hold per-video temp folders")
    # 评估输出根目录（避免写只读数据集）
    ap.add_argument("--out-root", type=str, default=None,
                    help="Where to save eval results; mirrors videoXX/ws_0/ under this root. "
                         "Default: same as --root (may be read-only).")
    args = ap.parse_args()

    # SAM-2
    setup_hydra()
    device = setup_device()
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)

    total = np.zeros(3, dtype=np.int64)  # TP, FP, FN
    per_video = {}

    # tmp-root 可写性检测
    tmp_root = args.tmp_root
    try:
        os.makedirs(tmp_root, exist_ok=True)
        p = os.path.join(tmp_root, ".write_test")
        with open(p, "w") as f:
            f.write("ok")
        os.remove(p)
    except Exception:
        print(f"[WARN] cannot write to tmp-root: {tmp_root} -> fallback to /tmp")
        tmp_root = tempfile.gettempdir()

    # out-root：默认与 root 相同；建议传入可写路径
    out_root = args.out_root if args.out_root is not None else args.root
    try:
        os.makedirs(out_root, exist_ok=True)
    except Exception as e:
        print(f"[WARN] cannot create out-root {out_root}: {e}")
        print("[WARN] fallback to current working directory for outputs.")
        out_root = os.getcwd()

    for vid in range(args.v_start, args.v_end + 1):
        base = os.path.join(args.root, f"video{vid}", "ws_0")
        img_dir = os.path.join(base, "images")
        gt_path = os.path.join(base, "prompts.json")
        pred_pts_dir = os.path.join(base, args.pts_dirname)

        if not (os.path.exists(gt_path) and os.path.isdir(img_dir) and os.path.isdir(pred_pts_dir)):
            print(f"skip video{vid}: missing images/prompts/points")
            continue

        gt_frames = load_gt_prompts(gt_path)
        pred_frames = load_pred_points(pred_pts_dir)

        # 临时帧目录 → 可写 tmp-root
        tmp_dir = os.path.join(tmp_root, f"sam2_eval_video{vid}")
        os.makedirs(tmp_dir, exist_ok=True)

        tp = fp = fn_cnt = 0
        frame_names = sorted(set(gt_frames.keys()) & set(pred_frames.keys()))
        if not frame_names:
            frame_names = sorted(gt_frames.keys())

        for fn in frame_names:
            img_path = os.path.join(img_dir, fn)
            if not os.path.exists(img_path):
                continue

            g_obj = gt_frames.get(fn, {}).get("objects", [])
            p_obj = pred_frames.get(fn, [])

            gt_masks = run_sam2_masks_for_objects(predictor, img_path, g_obj, tmp_dir)
            pr_masks = run_sam2_masks_for_objects(predictor, img_path, p_obj, tmp_dir)

            tpi, fpi, fni = greedy_match_iou(pr_masks, gt_masks, iou_thr=args.iou_thr)
            tp += tpi
            fp += fpi
            fn_cnt += fni

        # 清理当前视频的临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

        total += np.array([tp, fp, fn_cnt], dtype=np.int64)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn_cnt, 1)
        per_video[vid] = {"TP": int(tp), "FP": int(fp), "FN": int(fn_cnt),
                          "precision": round(prec, 4), "recall": round(rec, 4)}

        # 写入每视频结果到 out-root 的镜像目录
        out_base = os.path.join(out_root, f"video{vid}", "ws_0")
        os.makedirs(out_base, exist_ok=True)
        with open(os.path.join(out_base, "eval_iou.json"), "w") as f:
            json.dump(per_video[vid], f, indent=2)

        print(f"[video{vid}] TP={tp} FP={fp} FN={fn_cnt} | P={prec:.4f} R={rec:.4f}")

    P = total[0] / max(total[0] + total[1], 1)
    R = total[0] / max(total[0] + total[2], 1)
    overall = {"TP": int(total[0]), "FP": int(total[1]), "FN": int(total[2]),
               "precision": round(P, 4), "recall": round(R, 4)}
    print("\n=== Overall ===")
    print(overall)

    # 写总体结果到 out-root
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "eval_iou_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)


if __name__ == "__main__":
    main()
