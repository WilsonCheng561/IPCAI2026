#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SAM2+Classifier with the user's requested "matched-only mAP":

- PRED points:  /.../ours/cholec80_videoXX/points/0004325_objects.json
  For EACH point:
      * ext_mask (BASE ckpt) -> mask for matching & visualization
      * ext_cls  (RESUME finetuned) -> classification score only
- GT points:    /.../annotated_data/videoXX/ws_0/prompts.json
  For EACH GT object (positive points):
      * ext_mask (BASE ckpt) -> GT mask for matching & visualization

- Matching: per FRAME, per CLASS, greedy by score, IoU>=thr => a match.
  NEW mAP definition ("matched-only"): AP_c = (#matched pairs) / (#GT of class)
  i.e., IGNORE unmatched predictions (no precision penalty). mAP is mean over
  foreground classes.

- Diagnostics kept (with your required prints):
  * class-agnostic Recall (GT hit by ANY prediction)
  * conditional class-accuracy on matched pairs
  * IoU percentiles on correctly classified matches

- Visualization:
  Randomly sample up to --viz-num images that have >=1 matched pair.
  For each image: render
    Left  = ALL GT masks (from BASE ckpt) with class labels
    Right = ALL matched PRED masks (from BASE ckpt) with predicted class+score
  Save under <out-dir>/vis/*.jpg

Important: All masks (GT & Pred) are from BASE SAM2 ckpt ONLY (ext_mask).
Classifier head uses RESUME (ext_cls), but its masks are NOT used.
"""

import os, json, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# ========= import your training module bits =========
import sys
CLS_FILE = "/home/wcheng31/sam2_classify/train_sam2_classify_finetune_distill.py"
if CLS_FILE not in sys.modules:
    sys.path.append(str(Path(CLS_FILE).parent))
from train_sam2_classify_finetune_distill import (
    FineTuneSam2Wrapper, CosineClassifier, MLPHead, MLPBNHead
)

# --------- label mapping (fixed 5 classes) ----------
LABEL_MAP_5CLS = {
    "tool_to_id": {
        "background": 0, "clipper": 1, "grasper": 2, "hook": 3, "scissors": 4
    }
}
ID2NAME = {v:k for k,v in LABEL_MAP_5CLS["tool_to_id"].items()}
CLS_IDS_NO_BG = [1,2,3,4]

def map_objid_to_5cls(obj_id: int) -> str:
    if obj_id in (2,3,4): return "grasper"
    if obj_id == 6:       return "hook"
    if obj_id == 7:       return "scissors"
    if obj_id == 8:       return "clipper"
    return "background"

# --------------------- IO helpers ---------------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, p: Path):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# --------- parse auto-points JSON (flexible) ----------
def parse_auto_points(p: Path) -> List[List[float]]:
    """
    Accepts several shapes:
      - {"objects":[{"points":[[x,y],...]}...]}   (preferred)
      - {"points":[[x,y],...]}
      - [[x,y], ...]
      - [{"x":..,"y":..}, ...]
    Returns list of [x,y] (float). (labels assumed positive)
    """
    try:
        obj = load_json(p)
    except Exception:
        return []
    pts = []
    if isinstance(obj, dict) and "objects" in obj and isinstance(obj["objects"], list):
        for o in obj["objects"]:
            arr = o.get("points", []) or []
            for q in arr:
                if isinstance(q, (list,tuple)) and len(q)>=2:
                    pts.append([float(q[0]), float(q[1])])
                elif isinstance(q, dict) and "x" in q and "y" in q:
                    pts.append([float(q["x"]), float(q["y"])])
        return pts
    if isinstance(obj, dict) and "points" in obj:
        arr = obj["points"] or []
        for q in arr:
            if isinstance(q, (list,tuple)) and len(q)>=2:
                pts.append([float(q[0]), float(q[1])])
            elif isinstance(q, dict) and "x" in q and "y" in q:
                pts.append([float(q["x"]), float(q["y"])])
        return pts
    if isinstance(obj, list):
        for q in obj:
            if isinstance(q, (list,tuple)) and len(q)>=2:
                pts.append([float(q[0]), float(q[1])])
            elif isinstance(q, dict) and "x" in q and "y" in q:
                pts.append([float(q["x"]), float(q["y"])])
        return pts
    return []

# ---------- parse GT prompts.json to per-object positive points ----------
def load_gt_objects(prompts_json: Path) -> Dict[str, List[Dict]]:
    """
    Returns: dict frame_file(str) -> list of GT objects:
      each: {"class_id": int, "points": [[x,y],...]}
    Only keep objects with at least one positive point (labels==1).
    background class is EXCLUDED from GT (we don't evaluate it).
    """
    data = load_json(prompts_json)
    frames = data["frames"] if (isinstance(data, dict) and "frames" in data) else (data if isinstance(data, list) else [])
    out: Dict[str, List[Dict]] = {}
    for frm in frames:
        frame_file = frm.get("frame_file", None)
        if not frame_file: continue
        objs = []
        for obj in (frm.get("objects", []) or []):
            oid = int(obj.get("obj_id", -1))
            cls = map_objid_to_5cls(oid)
            if cls == "background":  # do not evaluate bg
                continue
            pts = obj.get("points", []) or []
            labs = obj.get("labels", []) or []
            pos = []
            if labs:
                n = min(len(pts), len(labs))
                for i in range(n):
                    if int(labs[i]) == 1:
                        x,y = pts[i][:2]; pos.append([float(x), float(y)])
            else:
                for p in pts:
                    x,y=p[:2]; pos.append([float(x), float(y)])
            if len(pos)==0: continue
            objs.append({
                "class_id": LABEL_MAP_5CLS["tool_to_id"][cls],
                "points": pos
            })
        if objs:
            out[frame_file] = objs
    return out

# -------------------- dataset over frames --------------------
def parse_videos(s: str) -> List[str]:
    s = (s or "").strip()
    if not s: return []
    if "-" in s and "," not in s:
        a,b = s.split("-",1); a=a.strip(); b=b.strip()
        if a.startswith("video") and b.startswith("video"):
            ai=int(a.replace("video","")); bi=int(b.replace("video",""))
            return [f"video{v:02d}" for v in range(ai, bi+1)]
    return [p.strip() for p in s.split(",") if p.strip()]

class FrameSet(Dataset):
    """
    Drive by AUTO points dir structure:
      <auto_root>/cholec80_videoXX/points/0004325_objects.json
    Only keep frames that also exist in GT prompts.json (videoXX/ws_0/prompts.json).
    """
    def __init__(self,
                 annotated_root: Path,   # /.../annotated_data
                 auto_root: Path,        # /.../ours
                 videos: List[str],      # ['video41',...]
                 ):
        super().__init__()
        self.items: List[Dict] = []
        for vid in videos:
            auto_dir = auto_root / f"cholec80_{vid}" / "points"
            gt_prom = annotated_root / vid / "ws_0" / "prompts.json"
            img_dir = annotated_root / vid / "ws_0" / "images"
            if not auto_dir.is_dir():
                print(f"[WARN] no auto points for {vid}: {auto_dir}")
                continue
            if not gt_prom.exists():
                print(f"[WARN] no GT prompts for {vid}: {gt_prom}")
                continue
            if not img_dir.is_dir():
                print(f"[WARN] no images for {vid}: {img_dir}")
                continue

            gt_map = load_gt_objects(gt_prom)  # frame_file -> list of gt objs
            for jf in sorted(auto_dir.glob("*_objects.json")):
                stem = jf.name.replace("_objects.json","")
                frame_file = f"{stem}.jpg"
                if frame_file not in gt_map:
                    # keep aligned with GT to evaluate
                    continue
                img_path = img_dir / frame_file
                if not img_path.exists():
                    continue
                auto_pts = parse_auto_points(jf)
                self.items.append({
                    "video": vid,
                    "frame_file": frame_file,
                    "image_path": str(img_path),
                    "auto_points": auto_pts,
                    "gt_objs": gt_map[frame_file],
                })

    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = cv2.imread(it["image_path"], cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(it["image_path"])
        return {"image": img, "meta": it}

def collate_frames(batch):
    return {"images": [b["image"] for b in batch], "metas": [b["meta"] for b in batch]}

# --------------------- utilities: IoU & matching ---------------------
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni   = np.logical_or(a, b).sum()
    return float(inter) / float(uni) if uni > 0 else 0.0

def greedy_match_per_class(preds: List[Dict], gts: List[Dict], class_id: int, iou_thr: float):
    """
    preds: list of {image_key, class_id, score, mask, image_path}
    gts  : list of {image_key, class_id, mask, image_path}
    Return matched pairs list:
      [{"image_key", "image_path", "class_id", "score", "iou", "pred_mask", "gt_mask"}]
    Greedy by score, per frame, and per class.
    """
    from collections import defaultdict
    P_by_img = defaultdict(list)
    G_by_img = defaultdict(list)
    for p in preds:
        if int(p["class_id"]) == class_id:
            P_by_img[p["image_key"]].append(p)
    for g in gts:
        if int(g["class_id"]) == class_id:
            G_by_img[g["image_key"]].append(dict(g, matched=False))
    pairs = []
    for key, plist in P_by_img.items():
        plist = sorted(plist, key=lambda x: -float(x["score"]))
        glist = G_by_img.get(key, [])
        for pr in plist:
            best_j, best_iou = -1, 0.0
            for j, gg in enumerate(glist):
                if gg["matched"]: continue
                iou = mask_iou(pr["mask"], gg["mask"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thr:
                glist[best_j]["matched"] = True
                pairs.append({
                    "image_key": key,
                    "image_path": pr.get("image_path", None),
                    "class_id": class_id,
                    "score": float(pr["score"]),
                    "iou": float(best_iou),
                    "pred_mask": pr["mask"],
                    "gt_mask": glist[best_j]["mask"]
                })
    return pairs

# --------------- AP (matched-only) = recall over matched ----------------
def compute_ap_matched_only(preds: List[Dict], gts: List[Dict], class_ids: List[int], thr: float):
    """
    For each class: AP_c = (#matched pairs) / (#GT of class)
    i.e., IGNORE unmatched predictions (no precision penalty).
    Return:
      ap_per_class: {cid: AP_c}
      matched_pairs_all: dict(cid -> list of pairs dict as returned by greedy_match_per_class)
    """
    ap = {}
    pairs_all = {}
    n_gt = {cid: 0 for cid in class_ids}
    for g in gts:
        cid = int(g["class_id"])
        if cid in n_gt:
            n_gt[cid] += 1
    for cid in class_ids:
        pairs = greedy_match_per_class(preds, gts, cid, thr)
        pairs_all[cid] = pairs
        npos = n_gt.get(cid, 0)
        ap[cid] = float(len(pairs)) / float(max(1, npos))
    return ap, pairs_all

# ====== DIAGNOSTICS ======
def class_agnostic_recall(preds, gts, iou_thr):
    # GT hit by any prediction, class-agnostic
    from collections import defaultdict
    pred_map = defaultdict(list)
    gt_map   = defaultdict(list)
    for p in preds: pred_map[p["image_key"]].append(p["mask"])
    for g in gts:   gt_map[g["image_key"]].append(g["mask"])
    hit, tot = 0, 0
    for k in gt_map:
        pm = pred_map.get(k, [])
        for gm in gt_map[k]:
            tot += 1
            ok = False
            for m in pm:
                iou = mask_iou(m, gm)
                if iou >= iou_thr:
                    ok = True; break
            if ok: hit += 1
    return (hit / max(1, tot)), hit, tot

def match_pairs_class_agnostic(preds, gts, iou_thr):
    # class-agnostic greedy matching by score, returns (imgkey, pred_cls, gt_cls, iou)
    from collections import defaultdict
    pred_by_img = defaultdict(list)
    gt_by_img   = defaultdict(list)
    for p in preds: pred_by_img[p["image_key"]].append(p)
    for g in gts:   gt_by_img[g["image_key"]].append(dict(g, matched=False))
    pairs = []
    for key, plist in pred_by_img.items():
        plist = sorted(plist, key=lambda x: -float(x["score"]))
        glist = gt_by_img.get(key, [])
        for pr in plist:
            best_i, best_iou = -1, 0.0
            for i, gg in enumerate(glist):
                if gg["matched"]: continue
                iou = mask_iou(pr["mask"], gg["mask"])
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= iou_thr:
                gl = glist[best_i]
                glist[best_i]["matched"] = True
                pairs.append((key, int(pr["class_id"]), int(gl["class_id"]), float(best_iou)))
    return pairs

def confusion_and_iou_stats(pairs, n_classes=5):
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    ious_correct = []
    for _, pc, gc, iou in pairs:
        C[gc, pc] += 1
        if pc == gc:
            ious_correct.append(iou)
    ious_correct = np.array(ious_correct, dtype=np.float32) if len(ious_correct) else np.array([], np.float32)
    pct = {}
    if len(ious_correct):
        pct = {
            "p50": float(np.percentile(ious_correct, 50)),
            "p75": float(np.percentile(ious_correct, 75)),
            "p90": float(np.percentile(ious_correct, 90)),
            "mean": float(ious_correct.mean())
        }
    return C, pct

# ---------------------------- visualization ----------------------------
def color_for_class(cid: int):
    # fixed palette for 5 classes (BGR)
    palette = {
        0: (180, 180, 180),  # background (rarely shown)
        1: (0, 165, 255),    # clipper - orange
        2: (0, 255, 0),      # grasper - green
        3: (255, 0, 0),      # hook - blue (BGR)
        4: (128, 0, 255),    # scissors - purple
    }
    return palette.get(cid, (255,255,255))

def draw_masks_panel(img_bgr: np.ndarray, items: List[Dict], use_pred: bool):
    """
    items:
      - GT:    list of {"mask", "class_id"}
      - Pred:  list of {"pred_mask","class_id","score"}  (we draw pred_mask)
    """
    vis = img_bgr.copy()
    overlay = vis.copy()
    for it in items:
        cid = int(it["class_id"])
        col = color_for_class(cid)
        m = it.get("mask", None)
        if use_pred:
            m = it.get("pred_mask", None)
        if m is None: continue
        m8 = (m.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 填充 + 粗边框
        cv2.drawContours(overlay, contours, -1, col, thickness=cv2.FILLED)
        cv2.drawContours(overlay, contours, -1, (0,0,0), thickness=3)
        cv2.drawContours(overlay, contours, -1, col, thickness=2)
        # 文本（类别/分数）+ 底条 + 描边
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            j = int(np.argmax(areas))
            x,y,w,h = cv2.boundingRect(contours[j])
            txt = ID2NAME.get(cid, f"class{cid}")
            if use_pred and ("score" in it):
                txt = f"{txt}:{it['score']:.2f}"
            bar_h = 22
            y0 = max(0, y - bar_h - 2)
            cv2.rectangle(overlay, (x, y0), (x + max(80, w), y0 + bar_h), (0,0,0), thickness=-1)
            cv2.putText(overlay, txt, (x + 4, y0 + bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(overlay, txt, (x + 4, y0 + bar_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    alpha = 0.40
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return vis

def visualize_random_pairs(pairs_by_class: Dict[int, List[Dict]],
                           gts_all: List[Dict],
                           out_dir: Path,
                           viz_num: int,
                           seed: int = 123):
    """
    Randomly pick up to viz_num images that have at least one matched pair.
    For each image: render GT (left) and all matched PRED (right).
    """
    random.seed(seed)
    # collect matched-by-image
    by_img_pairs = {}
    for cid, lst in pairs_by_class.items():
        for p in lst:
            key = p["image_key"]
            by_img_pairs.setdefault(key, []).append(p)
    if not by_img_pairs:
        return
    img_keys = list(by_img_pairs.keys())
    random.shuffle(img_keys)
    sel = img_keys[:min(viz_num, len(img_keys))]

    # index GT by image_key
    gt_by_img = {}
    for g in gts_all:
        key = g["image_key"]
        gt_by_img.setdefault(key, []).append(g)

    vis_dir = out_dir / "vis"
    ensure_dir(vis_dir)
    for key in sel:
        pairs = by_img_pairs[key]
        # 获取图像路径
        img_path = None
        for p in pairs:
            if p.get("image_path", None):
                img_path = p["image_path"]; break
        if img_path is None:
            gts = gt_by_img.get(key, [])
            for g in gts:
                if g.get("image_path", None):
                    img_path = g["image_path"]; break
        if img_path is None or not Path(img_path).exists():
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: continue

        gt_items = [{"mask": g["mask"], "class_id": int(g["class_id"])} for g in gt_by_img.get(key, [])]
        pred_items = [{"pred_mask": p["pred_mask"], "class_id": int(p["class_id"]), "score": float(p["score"])} for p in pairs]

        vis_left  = draw_masks_panel(img, gt_items, use_pred=False)
        vis_right = draw_masks_panel(img, pred_items, use_pred=True)
        vis_cat = np.concatenate([vis_left, vis_right], axis=1)

        # 标题含每类计数
        from collections import Counter
        cnt = Counter([int(p["class_id"]) for p in pairs])
        summary = " | ".join([f"{ID2NAME[c]}:{cnt.get(c,0)}" for c in [1,2,3,4]])
        title = f"GT (left) | Matched Predictions (right)  [{summary}]"
        cv2.putText(vis_cat, title,
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis_cat, title,
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        fname = key.replace("/", "_")
        cv2.imwrite(str(vis_dir / f"{fname}.jpg"), vis_cat)

# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser("SAM2+Classifier eval (matched-only mAP) + diagnostics + visualization")
    ap.add_argument("--annotated-root", type=str,
                    default="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data")
    ap.add_argument("--auto-root", type=str,
                    default="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours",
                    help="contains cholec80_videoXX/points/*.json")
    ap.add_argument("--videos", type=str, default="video41-video50")

    # model (use tiny to match your distill_maskcls_t)
    ap.add_argument("--sam2-cfg", type=str, default="sam2_hiera_t.yaml")
    ap.add_argument("--sam2-ckpt", type=str,
                    default="/projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_tiny.pt")
    ap.add_argument("--resume", type=str, required=True, help="classifier finetune ckpt (best_full_finetune.pt)")
    ap.add_argument("--head", choices=["cosine","linear","mlp","mlp_bn"], default="cosine")
    ap.add_argument("--hidden", type=str, default="1024,512")

    # thresholds & runtime
    ap.add_argument("--mask-thr", type=float, default=0.5, help="binarize prob to mask")
    ap.add_argument("--iou-list", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--mask-dilate", type=int, default=0, help="optional dilation kernel size (pixels)")
    ap.add_argument("--batch-size", type=int, default=2, help="batch = #frames")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-input-edge", type=int, default=1536)
    ap.add_argument("--out-dir", type=str,
                    default="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/eval_maskcls_points_vs_gt_matched")
    ap.add_argument("--dump-preds", action="store_true")
    ap.add_argument("--debug-dump", action="store_true")

    # visualization
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-num", type=int, default=20)

    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    videos = parse_videos(args.videos)

    annotated_root = Path(args.annotated_root)
    auto_root      = Path(args.auto_root)
    out_dir        = Path(args.out_dir); ensure_dir(out_dir)

    # dataset
    ds = FrameSet(annotated_root, auto_root, videos)
    if len(ds)==0:
        raise SystemExit("No overlapping frames between AUTO points and GT prompts.json.")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, collate_fn=collate_frames,
                    pin_memory=True, persistent_workers=(args.workers>0))

    # ========= TWO SEPARATE WRAPPERS =========
    # 1) ext_mask: BASE ckpt ONLY, used for ALL masks (GT & Pred) for matching/vis
    ext_mask = FineTuneSam2Wrapper(args.sam2_cfg, args.sam2_ckpt, device=device,
                                   max_input_edge=args.max_input_edge).to(device)
    ext_mask.set_trainable("none")
    ext_mask.eval()

    # 2) ext_cls: for classification ONLY; load resume's states if available
    ext_cls = FineTuneSam2Wrapper(args.sam2_cfg, args.sam2_ckpt, device=device,
                                  max_input_edge=args.max_input_edge).to(device)
    ext_cls.set_trainable("none")
    ext_cls.eval()

    # probe dim from BASE extractor（尺寸一致）
    sample = ds[0]
    with torch.no_grad():
        feat = ext_cls([sample["image"]], [np.zeros((0,3),np.float32)], [{"image_path": sample["meta"]["image_path"]}])
    in_dim = int(feat.shape[-1])

    # head
    if args.head=="cosine":
        head = CosineClassifier(in_dim, 5, scale=16.0).to(device)
    elif args.head=="linear":
        head = MLPHead(in_dim, 5, hidden=0, drop=0.0).to(device)
    elif args.head=="mlp":
        h = int(args.hidden.split(",")[0]) if args.hidden.strip() else 1024
        head = MLPHead(in_dim, 5, hidden=h, drop=0.0).to(device)
    else:
        hs = [int(x) for x in args.hidden.split(",") if x.strip()]
        if not hs: hs=[1024,512]
        head = MLPBNHead(in_dim, 5, hidden_layers=hs, drop=0.0).to(device)

    # ===== load ckpt into ext_cls + head (NOT into ext_mask) =====
    ckpt = torch.load(args.resume, map_location="cpu")
    if isinstance(ckpt, dict) and ("head_state" in ckpt):
        head.load_state_dict(ckpt["head_state"], strict=False)
        # if finetuned SAM2 exists in ckpt, load into ext_cls ONLY
        if "sam2_state" in ckpt:
            try:
                ext_cls.load_state_dict(ckpt["sam2_state"], strict=False)
            except Exception:
                pass
    else:
        head.load_state_dict(ckpt, strict=False)

    ext_cls.eval(); head.eval()

    iou_list = [float(s) for s in args.iou_list.split(",") if s.strip()]
    preds_all: List[Dict] = []
    gts_all  : List[Dict] = []

    # debug counters
    pred_class_hist = np.zeros(5, dtype=np.int64)
    nonempty_masks = 0
    total_pred_masks = 0
    out_of_bounds_points = 0
    total_auto_points = 0

    print(f"[INFO] Frames to eval: {len(ds)} (videos {videos})")
    amp_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)

    for batch in tqdm(dl, ncols=100, desc="Eval"):
        images: List[np.ndarray] = batch["images"]
        metas : List[Dict]       = batch["metas"]

        for img, meta in zip(images, metas):
            H0, W0 = img.shape[:2]
            image_key = f'{meta["video"]}/{meta["frame_file"]}'

            # ----- GT masks via ext_mask (BASE ckpt) -----
            gt_objs = meta["gt_objs"]
            if len(gt_objs):
                imgs_b = []
                metas_b= []
                pts_b  = []
                for ob in gt_objs:
                    pts_np = np.array([[float(x), float(y), 1.0] for (x,y) in ob["points"]], dtype=np.float32)
                    imgs_b.append(img)
                    metas_b.append({"image_path": meta["image_path"]})
                    pts_b.append(pts_np)
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                    _, mlog = ext_mask(imgs_b, pts_b, metas_b, return_mask_logits=True)
                mlog = mlog.detach().cpu().float().numpy()
                for i, ob in enumerate(gt_objs):
                    mprob = 1/(1+np.exp(-mlog[i,0]))
                    mup = cv2.resize(mprob, (W0,H0), interpolation=cv2.INTER_LINEAR)
                    mb = (mup >= float(args.mask_thr)).astype(np.uint8)
                    if args.mask_dilate and args.mask_dilate>0:
                        k = int(args.mask_dilate); k = k+1 if k%2==0 else k
                        mb = cv2.dilate(mb, np.ones((k,k), np.uint8), iterations=1)
                    gts_all.append({
                        "image_key": image_key,
                        "image_path": meta["image_path"],
                        "class_id": int(ob["class_id"]),
                        "mask": mb
                    })

            # ----- PRED: auto points -----
            auto_pts = meta["auto_points"] or []
            if len(auto_pts)==0:
                continue

            # detect if points are normalized [0,1], and scale if so
            if len(auto_pts):
                xs = [p[0] for p in auto_pts]
                ys = [p[1] for p in auto_pts]
                if all(0.0 <= v <= 1.0 for v in xs+ys):
                    auto_pts = [[p[0]*W0, p[1]*H0] for p in auto_pts]

            # stats: OOB
            for (x,y) in auto_pts:
                total_auto_points += 1
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W0 and 0 <= yi < H0):
                    out_of_bounds_points += 1

            CHUNK = 64
            for s in range(0, len(auto_pts), CHUNK):
                pts_chunk = auto_pts[s:s+CHUNK]
                imgs_b = [img for _ in pts_chunk]
                metas_b= [{"image_path": meta["image_path"]} for _ in pts_chunk]
                pts_b  = [np.array([[float(x),float(y),1.0]], dtype=np.float32) for (x,y) in pts_chunk]

                # (A) masks via BASE ext_mask
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                    _, mlog_m = ext_mask(imgs_b, pts_b, metas_b, return_mask_logits=True)
                mlog_np = mlog_m.detach().cpu().float().numpy()

                # (B) classification via ext_cls + head
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                    feats_cls = ext_cls(imgs_b, pts_b, metas_b, return_mask_logits=False)
                    logits = head(feats_cls)
                    prob = torch.softmax(logits.float(), dim=1)
                prob_np = prob.detach().cpu().numpy()

                # gather
                for i in range(len(pts_chunk)):
                    mprob = 1/(1+np.exp(-mlog_np[i,0]))
                    mup = cv2.resize(mprob, (W0,H0), interpolation=cv2.INTER_LINEAR)
                    mb = (mup >= float(args.mask_thr)).astype(np.uint8)
                    if args.mask_dilate and args.mask_dilate>0:
                        k = int(args.mask_dilate); k = k+1 if k%2==0 else k
                        mb = cv2.dilate(mb, np.ones((k,k), np.uint8), iterations=1)
                    total_pred_masks += 1
                    if mb.any(): nonempty_masks += 1
                    p = prob_np[i]; cid = int(p.argmax()); score=float(p[cid])
                    pred_class_hist[cid] += 1
                    preds_all.append({
                        "image_key": image_key,
                        "image_path": meta["image_path"],
                        "class_id": cid,
                        "score": score,
                        "mask": mb
                    })

    # ---------- matched-only AP/mAP ----------
    results = {}
    matched_pairs_for_viz = {}
    for thr in iou_list:
        ap_per, pairs_all = compute_ap_matched_only(preds_all, gts_all, CLS_IDS_NO_BG, thr)
        mAP_matched = float(np.mean([ap_per[c] for c in CLS_IDS_NO_BG])) if CLS_IDS_NO_BG else 0.0
        results[f"AP_matched@{thr:.2f}"] = {"mAP": mAP_matched, "per_class": {ID2NAME[c]: ap_per[c] for c in CLS_IDS_NO_BG}}
        print(f"[AP_matched] IoU={thr:.2f}: mAP={mAP_matched:.4f} | " +
              ", ".join([f"{ID2NAME[c]}={ap_per[c]:.3f}" for c in CLS_IDS_NO_BG]))
        matched_pairs_for_viz = pairs_all  # keep last thr's matches for visualization

    save_json(results, out_dir / "map_iou_results_matched_only.json")
    print("[SAVE] ->", out_dir/"map_iou_results_matched_only.json")

    if args.dump_preds:
        lite = [{
            "image_key": p["image_key"],
            "class_id": int(p["class_id"]),
            "class_name": ID2NAME.get(int(p["class_id"]), f"class{p['class_id']}"),
            "score": float(p["score"])
        } for p in preds_all]
        save_json(lite, out_dir / "preds_lite.json")
        print("[SAVE] preds lite ->", out_dir/"preds_lite.json")

    # ---------- diagnostics (keep your required prints) ----------
    diag_thr = iou_list[-1] if len(iou_list) else 0.5
    rec_val, hit, tot = class_agnostic_recall(preds_all, gts_all, diag_thr)
    pairs_diag = match_pairs_class_agnostic(preds_all, gts_all, diag_thr)
    cond_cls_acc = (sum(1 for _,pc,gc,_ in pairs_diag if pc==gc) / max(1,len(pairs_diag))) if len(pairs_diag) else 0.0
    C, pct = confusion_and_iou_stats(pairs_diag, n_classes=5)

    print(f"\nIoU@{diag_thr:.2f}:")
    print(f"  class-agnostic Recall         : {rec_val:.4f} (hit/total={[int(hit), int(tot)]})")
    print(f"  cond. class acc on matched    : {cond_cls_acc:.4f}")
    if pct:
        print(f"  IoU(correct cls) p50/p75/p90/mean: {pct.get('p50',0):.3f}/{pct.get('p75',0):.3f}/{pct.get('p90',0):.3f}/{pct.get('mean',0):.3f}")
    else:
        print("  IoU(correct cls) p50/p75/p90/mean: N/A")

    extra = {
        "pred_class_hist": {ID2NAME[i]: int(v) for i,v in enumerate(pred_class_hist)},
        "pred_total": int(pred_class_hist.sum()),
        "nonempty_mask_ratio": float(nonempty_masks / max(1, total_pred_masks)),
        "auto_points_total": int(total_auto_points),
        "auto_points_oob": int(out_of_bounds_points),
        "auto_points_oob_ratio": float(out_of_bounds_points / max(1, total_auto_points))
    }
    diag = {
        f"IoU@{diag_thr:.2f}": {
            "class_agnostic_Recall": round(rec_val, 4),
            "recall_hit_over_total": [int(hit), int(tot)],
            "cond_class_acc_on_matched": round(cond_cls_acc, 4),
            "confusion_on_matched": C.tolist(),
            "iou_stats_on_correct_class": pct
        },
        "extra_stats": extra
    }
    save_json(diag, out_dir / "diagnostics.json")
    print("[SAVE] diagnostics ->", out_dir/"diagnostics.json")

    # ---------- visualization (masks from BASE ext_mask only) ----------
    if args.viz:
        visualize_random_pairs(matched_pairs_for_viz, gts_all, out_dir, viz_num=args.viz_num, seed=123)
        print("[SAVE] visualizations ->", out_dir/"vis")

if __name__ == "__main__":
    main()
