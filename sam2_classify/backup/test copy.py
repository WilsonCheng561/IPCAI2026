#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate SAM2+Classifier with:
  - PRED points:  /.../ours/cholec80_videoXX/points/0004325_objects.json
      -> for EACH point: run SAM2 to get (mask) and run classifier head to get class+score
  - GT points:    /.../annotated_data/videoXX/ws_0/prompts.json
      -> for EACH GT object (positive points): run SAM2 to get GT mask
  - Matching: per frame, per class, greedy by score, IoU>=thr => TP; else FP; COCO-style AP/mAP.

Also prints diagnostics to pinpoint bottlenecks:
  - Class-agnostic AP upper bound (only mask/location matters)
  - Class-agnostic Recall (GT covered by any prediction)
  - Conditional class-accuracy on matched pairs
  - Confusion matrix (on matched)
  - IoU stats on correctly classified matches
  - Prediction class histogram, non-empty-mask rate, OOB points ratio, etc.

Defaults use SAM2 tiny, matching your 'distill_maskcls_t' checkpoint.
"""

import os, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

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

# --------------- AP by IoU (mask↔mask), per class ---------------
def compute_ap_iou(preds: List[Dict], gts: List[Dict], class_ids: List[int], thr: float) -> Dict[int,float]:
    """
    preds: list of {image_key, class_id, score, mask: np.uint8(H,W)}
    gts  : list of {image_key, class_id, mask: np.uint8(H,W)}
    Return AP per class at IoU threshold.
    """
    ap = {}
    # group gts by (image_key, class_id)
    gt_map: Dict[Tuple[str,int], List[Dict]] = {}
    for g in gts:
        k=(g["image_key"], g["class_id"])
        gt_map.setdefault(k, []).append({**g, "matched": False})
    for cid in class_ids:
        P = [p for p in preds if p["class_id"]==cid]
        if not P: ap[cid]=0.0; continue
        P.sort(key=lambda x: -float(x["score"]))
        tp=np.zeros(len(P), np.float32); fp=np.zeros(len(P), np.float32)
        npos = sum(len(v) for (k,c),v in gt_map.items() if c==cid)
        for i,pr in enumerate(P):
            key = pr["image_key"]; pm = pr["mask"]
            pool = gt_map.get((key,cid), [])
            best_j=-1; best_iou=0.0
            for j,g in enumerate(pool):
                if g["matched"]: continue
                gm = g["mask"]
                inter = np.logical_and(pm, gm).sum()
                union = np.logical_or(pm, gm).sum()
                iou = (inter/union) if union>0 else 0.0
                if iou>best_iou:
                    best_iou=iou; best_j=j
            if best_iou >= thr and best_j>=0:
                tp[i]=1.0; pool[best_j]["matched"]=True
            else:
                fp[i]=1.0
        tp_c=np.cumsum(tp); fp_c=np.cumsum(fp)
        if npos==0: ap[cid]=0.0; continue
        recall = tp_c / npos
        precision = tp_c / np.maximum(1, tp_c+fp_c)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre= np.concatenate(([0.0], precision,[0.0]))
        for k in range(mpre.size-1,0,-1):
            mpre[k-1]=max(mpre[k-1], mpre[k])
        idx = np.where(mrec[1:]!=mrec[:-1])[0]
        ap[cid] = float(np.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]))
    return ap

# ====== DIAGNOSTICS ======
def class_agnostic_ap(preds, gts, iou_thr):
    P = [dict(p, class_id=1) for p in preds]
    G = [dict(g, class_id=1) for g in gts]
    ap1 = compute_ap_iou(P, G, class_ids=[1], thr=iou_thr)[1]
    return ap1

def class_agnostic_recall(preds, gts, iou_thr):
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
                inter = np.logical_and(m, gm).sum()
                uni   = np.logical_or(m, gm).sum()
                iou   = (inter/uni) if uni>0 else 0.0
                if iou >= iou_thr:
                    ok = True; break
            if ok: hit += 1
    return (hit / max(1, tot)), hit, tot

def match_pairs_class_agnostic(preds, gts, iou_thr):
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
                inter = np.logical_and(pr["mask"], gg["mask"]).sum()
                uni   = np.logical_or(pr["mask"], gg["mask"]).sum()
                iou   = (inter/uni) if uni>0 else 0.0
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

# ---------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser("SAM2+Classifier eval: AUTO points vs GT points -> masks IoU AP (+diagnostics)")
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
    ap.add_argument("--iou-list", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--mask-dilate", type=int, default=0, help="optional dilation kernel size (pixels)")
    ap.add_argument("--batch-size", type=int, default=2, help="batch = #frames (we do point-batching per frame)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-input-edge", type=int, default=1536)
    ap.add_argument("--out-dir", type=str,
                    default="/projects/surgical-video-digital-twin/pretrain_params/cwz/ours/eval_maskcls_points_vs_gt")
    ap.add_argument("--dump-preds", action="store_true", help="dump light meta of predictions")
    ap.add_argument("--debug-dump", action="store_true", help="dump extra debug stats")

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

    # model wrapper
    ext = FineTuneSam2Wrapper(args.sam2_cfg, args.sam2_ckpt, device=device,
                              max_input_edge=args.max_input_edge).to(device)
    ext.set_trainable("none")

    # probe dim
    sample = ds[0]
    with torch.no_grad():
        feat = ext([sample["image"]], [np.zeros((0,3),np.float32)], [{"image_path": sample["meta"]["image_path"]}])
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

    # load ckpt & warn mismatch
    ckpt = torch.load(args.resume, map_location="cpu")
    if isinstance(ckpt, dict) and ("head_state" in ckpt):
        head.load_state_dict(ckpt["head_state"], strict=False)
        cargs = ckpt.get("args", {})
        ck_cfg  = cargs.get("sam2_cfg", None)
        ck_ckpt = cargs.get("sam2_ckpt", None)
        if (ck_cfg and Path(ck_cfg).name != Path(args.sam2_cfg).name) or \
           (ck_ckpt and Path(ck_ckpt).name != Path(args.sam2_ckpt).name):
            print("\n\033[91m[WARN] resume ckpt was trained with "
                  f"cfg={ck_cfg} ckpt={ck_ckpt}, now you use cfg={args.sam2_cfg} ckpt={args.sam2_ckpt}.\n"
                  "Make them consistent (tiny↔tiny / large↔large)!\033[0m\n")
        try:
            ext.load_state_dict(ckpt["sam2_state"], strict=False)
        except Exception:
            pass
    else:
        head.load_state_dict(ckpt, strict=False)

    ext.eval(); head.eval()

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
    for batch in tqdm(dl, ncols=100, desc="Eval"):
        images: List[np.ndarray] = batch["images"]
        metas : List[Dict]       = batch["metas"]

        for img, meta in zip(images, metas):
            H0, W0 = img.shape[:2]
            image_key = f'{meta["video"]}/{meta["frame_file"]}'

            # ----- build GT masks (each object -> mask with ALL positive points) -----
            gt_objs = meta["gt_objs"]
            if len(gt_objs):
                imgs_b = []
                metas_b= []
                pts_b  = []
                for ob in gt_objs:
                    # points are pixel coordinates
                    pts_np = np.array([[float(x), float(y), 1.0] for (x,y) in ob["points"]], dtype=np.float32)
                    imgs_b.append(img)
                    metas_b.append({"image_path": meta["image_path"]})
                    pts_b.append(pts_np)
                with torch.inference_mode(), torch.cuda.amp.autocast(
                    enabled=True, dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
                ):
                    _, mlog = ext(imgs_b, pts_b, metas_b, return_mask_logits=True)
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
                        "class_id": int(ob["class_id"]),
                        "mask": mb
                    })

            # ----- PRED: auto points -> class+score & mask (one point each) -----
            auto_pts = meta["auto_points"] or []
            if len(auto_pts)==0:
                continue

            # detect if points are normalized [0,1], and scale if so
            # rule: if all 0<=x<=1 and 0<=y<=1, treat as normalized
            if len(auto_pts):
                xs = [p[0] for p in auto_pts]
                ys = [p[1] for p in auto_pts]
                if all(0.0 <= v <= 1.0 for v in xs+ys):
                    auto_pts = [[p[0]*W0, p[1]*H0] for p in auto_pts]

            # stats: out-of-bounds
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
                with torch.inference_mode(), torch.cuda.amp.autocast(
                    enabled=True, dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)
                ):
                    feats, mlog = ext(imgs_b, pts_b, metas_b, return_mask_logits=True)
                    logits = head(feats)
                    prob = torch.softmax(logits.float(), dim=1)
                prob_np = prob.detach().cpu().numpy()
                mlog_np = mlog.detach().cpu().float().numpy()
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
                        "class_id": cid,
                        "score": score,
                        "mask": mb
                    })

    # ---------- compute AP for each IoU thresh, then COCO mAP ----------
    results = {}
    for thr in iou_list:
        ap_per = compute_ap_iou(preds_all, gts_all, CLS_IDS_NO_BG, thr)
        mAP = float(np.mean([ap_per[c] for c in CLS_IDS_NO_BG])) if CLS_IDS_NO_BG else 0.0
        results[f"AP@{thr:.2f}"] = {"mAP": mAP, "per_class": {ID2NAME[c]: ap_per[c] for c in CLS_IDS_NO_BG}}
        print(f"[AP] IoU={thr:.2f}: mAP={mAP:.4f} | " +
              ", ".join([f"{ID2NAME[c]}={ap_per[c]:.3f}" for c in CLS_IDS_NO_BG]))
    if len(iou_list)>=10 and abs(iou_list[0]-0.5)<1e-6 and abs(iou_list[-1]-0.95)<1e-6:
        mAP_coco = float(np.mean([results[f"AP@{t:.2f}"]["mAP"] for t in iou_list]))
        results["mAP_50_95"] = mAP_coco
        print(f"[mAP] COCO 0.50:0.95 = {mAP_coco:.4f}")

    save_json(results, out_dir / "map_iou_results.json")
    print("[SAVE] ->", out_dir/"map_iou_results.json")

    if args.dump_preds:
        lite = [{
            "image_key": p["image_key"],
            "class_id": int(p["class_id"]),
            "class_name": ID2NAME.get(int(p["class_id"]), f"class{p['class_id']}"),
            "score": float(p["score"])
        } for p in preds_all]
        save_json(lite, out_dir / "preds_lite.json")
        print("[SAVE] preds lite ->", out_dir/"preds_lite.json")

    # ---------- diagnostics ----------
    DIAG_THRS = [0.50, 0.75, 0.80]
    diag = {}
    for thr in DIAG_THRS:
        ca_ap   = class_agnostic_ap(preds_all, gts_all, thr)
        rec, hit, tot = class_agnostic_recall(preds_all, gts_all, thr)
        pairs = match_pairs_class_agnostic(preds_all, gts_all, thr)
        cls_acc = (sum(1 for _,pc,gc,_ in pairs if pc==gc) / max(1,len(pairs))) if len(pairs) else 0.0
        C, pct = confusion_and_iou_stats(pairs, n_classes=5)
        diag[f"IoU@{thr:.2f}"] = {
            "class_agnostic_AP_upperbound": round(ca_ap, 4),
            "class_agnostic_Recall": round(rec, 4),
            "recall_hit_over_total": [int(hit), int(tot)],
            "cond_class_acc_on_matched": round(cls_acc, 4),
            "confusion_on_matched": C.tolist(),
            "iou_stats_on_correct_class": pct
        }

    # extra debug stats
    extra = {
        "pred_class_hist": {ID2NAME[i]: int(v) for i,v in enumerate(pred_class_hist)},
        "pred_total": int(pred_class_hist.sum()),
        "nonempty_mask_ratio": float(nonempty_masks / max(1, total_pred_masks)),
        "auto_points_total": int(total_auto_points),
        "auto_points_oob": int(out_of_bounds_points),
        "auto_points_oob_ratio": float(out_of_bounds_points / max(1, total_auto_points))
    }
    diag["extra_stats"] = extra

    save_json(diag, out_dir / "diagnostics.json")
    print("\n===== DIAGNOSTICS =====")
    for k, v in diag.items():
        if k == "extra_stats":
            print("extra_stats:", v)
            continue
        print(f"{k}:")
        print("  class-agnostic AP (upper bound):", v["class_agnostic_AP_upperbound"])
        print("  class-agnostic Recall         :", v["class_agnostic_Recall"], f"(hit/total={v['recall_hit_over_total']})")
        print("  cond. class acc on matched    :", v["cond_class_acc_on_matched"])
        if v["iou_stats_on_correct_class"]:
            s = v["iou_stats_on_correct_class"]
            print(f"  IoU(correct cls) p50/p75/p90/mean: {s.get('p50',0):.3f}/{s.get('p75',0):.3f}/{s.get('p90',0):.3f}/{s.get('mean',0):.3f}")
        else:
            print("  IoU(correct cls) stats: N/A")
    print("[SAVE] diagnostics ->", out_dir/"diagnostics.json")

if __name__ == "__main__":
    main()
