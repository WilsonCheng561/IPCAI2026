#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test SAM2(+optional finetune) + classifier head on either:
  - prepared CSV (train pipeline outputs), or
  - OOD set built from cholec80_raw/annotated_data prompts.json (videos selectable)

Prints overall & per-class acc, and saves visualizations (mask overlay + top-k).
"""

import os, json, argparse, random, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

# ---------- Paths ----------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify")
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"

# ---------- Import training-side components ----------
import sys
sys.path.append(str(SMALLFILE_ROOT))
from train_sam2_classify import (
    Sam2OfficialWrapper,
    FramePointDataset,
    collate_varlen,
    load_label_map,
    ensure_dir,
    MLPHead,
)

try:
    from train_sam2_classify import MLPBNHead, CosineClassifier
except Exception:
    MLPBNHead, CosineClassifier = None, None


# ========================= OOD (raw) building =========================

TOOL_TO_ID_OOD = {"background":0, "clipper":1, "grasper":2, "hook":3, "scissors":4}
ID_TO_TOOL_OOD = {v:k for k,v in TOOL_TO_ID_OOD.items()}

def _map_obj_id_to_cls(obj_id: int) -> str:
    # 2/3/4 -> grasper; 6->hook; 7->scissors; 8->clipper; others -> background
    if obj_id in (2,3,4): return "grasper"
    if obj_id == 6:       return "hook"
    if obj_id == 7:       return "scissors"
    if obj_id == 8:       return "clipper"
    return "background"

def _save_json(obj, p: Path):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_ood_manifest(raw_root: Path, videos: List[str], out_csv: Path) -> pd.DataFrame:
    rows = []
    for vid in videos:
        vid_dir = raw_root / vid / "ws_0"
        img_dir = vid_dir / "images"
        prm = vid_dir / "prompts.json"
        if not prm.exists():
            print(f"[WARN] missing prompts: {prm}")
            continue
        data = _load_json(prm)
        for rec in data:
            frame_file = rec.get("frame_file", None)
            if not frame_file:
                fid = rec.get("frame_id", None)
                frame_file = f"{int(fid):07d}.jpg" if fid is not None else None
            if not frame_file:
                continue
            img_path = img_dir / frame_file
            if not img_path.exists():
                print(f"[SKIP] image not found: {img_path}")
                continue

            buckets: Dict[str, List[List[float]]] = {}
            for obj in rec.get("objects", []):
                oid = int(obj.get("obj_id", -1))
                cls = _map_obj_id_to_cls(oid)
                pts = obj.get("points", [])
                labs = obj.get("labels", [])
                if len(labs) != len(pts):
                    L = min(len(pts), len(labs))
                    pts, labs = pts[:L], labs[:L]
                for (xy, lab) in zip(pts, labs):
                    if not isinstance(xy, (list,tuple)) or len(xy) < 2:
                        continue
                    x, y = float(xy[0]), float(xy[1])
                    label = 1.0 if int(lab) > 0 else 0.0
                    buckets.setdefault(cls, []).append([x, y, label])

            for cls, pts in buckets.items():
                if len(pts) == 0: continue
                rows.append({
                    "image_path": str(img_path),
                    "tool": cls,
                    "points_json": json.dumps(pts),
                    "frame_id": rec.get("frame_id", -1),
                    "clip_name": vid,
                })

    df = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    print(f"[BUILD][OOD] saved -> {out_csv}  (#rows={len(df)})")
    return df

class OODDataset(Dataset):
    def __init__(self, manifest_csv: Path, label_map_json: Path, resize: Optional[int] = None):
        super().__init__()
        self.df = pd.read_csv(manifest_csv)
        with open(label_map_json, "r", encoding="utf-8") as f:
            lm = json.load(f)
        self.tool2id = lm["tool_to_id"]
        self.resize = resize

        def has_points(s: str) -> bool:
            try:
                arr = json.loads(s) if isinstance(s, str) and s.strip() else []
                return len(arr) > 0
            except Exception:
                return False
        self.df = self.df[self.df["points_json"].apply(has_points)].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def _load_img(self, p: str):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(p)
        if self.resize and self.resize > 0:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self._load_img(row["image_path"])
        H, W = img.shape[:2]
        pts = json.loads(row["points_json"])
        pts_out = []
        for p in pts:
            x = float(np.clip(p[0], 0, W-1)); y = float(np.clip(p[1], 0, H-1))
            label = 1.0 if float(p[2]) > 0 else 0.0
            pts_out.append([x, y, label])
        tool = str(row["tool"])
        tool_id = int(self.tool2id[tool])
        return {
            "image": img,
            "points": np.array(pts_out, dtype=np.float32),
            "tool_id": tool_id,
            "meta": {"image_path": row["image_path"], "tool": tool, "clip_name": row.get("clip_name","")}
        }

# ========================= Head helpers =========================

def _parse_hidden_list(s: str) -> List[int]:
    out = []
    try:
        for p in str(s).split(","):
            p = p.strip()
            if p:
                out.append(int(p))
    except Exception:
        pass
    return out

def _build_head_from_ckpt_args(in_dim: int, n_classes: int, args_dict: Dict, override_head: Optional[str] = None):
    head_type = override_head or args_dict.get("head", "mlp")
    drop      = float(args_dict.get("drop", 0.0))
    scale     = float(args_dict.get("scale", 16.0))
    hidden    = args_dict.get("hidden", "0")

    if head_type == "linear":
        return MLPHead(in_dim, n_classes, hidden=0, drop=drop)
    if head_type == "mlp":
        h = 0
        if isinstance(hidden, str):
            lst = _parse_hidden_list(hidden); h = (lst[0] if lst else 0)
        elif isinstance(hidden, (list, tuple)) and len(hidden) > 0:
            h = int(hidden[0])
        return MLPHead(in_dim, n_classes, hidden=h, drop=drop)
    if head_type == "mlp_bn" and MLPBNHead is not None:
        lst = _parse_hidden_list(hidden) or [1024, 512]
        return MLPBNHead(in_dim, n_classes, hidden_layers=lst, drop=drop)
    if head_type == "cosine" and CosineClassifier is not None:
        return CosineClassifier(in_dim, n_classes, scale=scale)
    return MLPHead(in_dim, n_classes, hidden=0, drop=drop)

# ========================= Core eval =========================

@torch.no_grad()
def _mask_and_feat_for_one(extractor: Sam2OfficialWrapper,
                           img_bgr: np.ndarray,
                           pts_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    img_t, (H0,W0), (H_in,W_in), sy, sx = extractor._preprocess_manual(img_bgr)
    img_feat, img_pe, high_res = extractor._get_image_embed(img_t)

    if pts_np is None or len(pts_np) == 0:
        mask = torch.ones((1,1,img_feat.shape[-2], img_feat.shape[-1]), device=img_feat.device)
    else:
        coords = extractor._map_points_scale_xy(pts_np, sy, sx).to(img_feat.device)
        labels = torch.from_numpy(np.asarray(pts_np, np.float32)[:, 2]).unsqueeze(0).to(img_feat.device)
        sp, dp = extractor._encode_prompts(coords, labels)
        mask_logits = extractor._decode_mask(img_feat, img_pe, sp, dp, high_res)
        mask = torch.sigmoid(mask_logits)
        if mask.shape[-2:] != img_feat.shape[-2:]:
            mask = F.interpolate(mask, size=img_feat.shape[-2:], mode="bilinear", align_corners=False)

    feat = (img_feat * mask).flatten(2).sum(dim=-1) / (mask.flatten(2).sum(dim=-1) + 1e-6)
    feat = feat.squeeze(0)

    img_rgb = (img_t[0].permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)
    vis_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask2d = mask.squeeze().detach().cpu().numpy().astype(np.float32)
    return vis_bgr, mask2d, feat

@torch.no_grad()
def evaluate_batchwise(extractor: Sam2OfficialWrapper, head: nn.Module, loader: DataLoader, device: str):
    head.eval()
    ce = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss, total_n, total_correct = 0.0, 0, 0
    n_classes = None

    pbar = tqdm(loader, total=len(loader), ncols=100, desc="[test] eval", leave=True)
    for batch in pbar:
        imgs, pts = batch["images"], batch["points"]
        y = batch["targets"].to(device)

        feats = extractor(imgs, pts)
        logits = head(feats)
        loss = ce(logits, y)

        pred = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        total_correct += (pred == y).sum().item()

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        if n_classes is None:
            n_classes = int(logits.shape[1])

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(total_correct/max(1,total_n)):.3f}")

    y_true = np.concatenate(y_true) if y_true else np.zeros((0,), np.int64)
    y_pred = np.concatenate(y_pred) if y_pred else np.zeros((0,), np.int64)

    overall_acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    per_class_acc, per_class_cnt = {}, {}
    if len(y_true):
        for c in np.unique(y_true):
            idx = (y_true == c)
            per_class_cnt[int(c)] = int(idx.sum())
            per_class_acc[int(c)] = float((y_pred[idx] == c).mean()) if idx.any() else 0.0

    avg_loss = total_loss / max(1, total_n)
    return avg_loss, overall_acc, per_class_acc, per_class_cnt, y_true, y_pred

@torch.no_grad()
def visualize_samples(extractor: Sam2OfficialWrapper,
                      head: nn.Module,
                      df: pd.DataFrame,
                      label_map: Dict,
                      out_dir: Path,
                      num_vis: int = 50,
                      conf_thr: float = 0.5,
                      topk: int = 3,
                      device: str = "cuda"):
    ensure_dir(out_dir)
    id2tool = {int(v): str(k) for k, v in label_map["tool_to_id"].items()}
    count = min(num_vis, len(df))
    pbar = tqdm(range(count), ncols=100, desc="[vis] writing", leave=True)
    for i in pbar:
        row = df.iloc[i]
        img_path = row["image_path"]; pts_json = row["points_json"]
        tool_name = str(row["tool"])
        tool_id = label_map["tool_to_id"].get(tool_name, None)
        if tool_id is None: continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: continue

        try:
            pts = json.loads(pts_json) if isinstance(pts_json, str) and pts_json.strip() else []
        except Exception:
            pts = []
        pts_np = np.asarray(pts, np.float32) if len(pts) else np.zeros((0,3), np.float32)

        vis_bgr, mask2d, feat = _mask_and_feat_for_one(extractor, img, pts_np)

        H, W = vis_bgr.shape[:2]
        if mask2d.shape[:2] != (H, W):
            mask_vis = cv2.resize(mask2d, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            mask_vis = mask2d

        logits = head(feat.unsqueeze(0).to(device))
        prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        top_idx = prob.argsort()[::-1][:topk]
        top_pairs = [(int(j), float(prob[j])) for j in top_idx]

        m = (mask_vis >= conf_thr).astype(np.uint8)
        color = np.zeros_like(vis_bgr); color[..., 2] = (m * 255)
        overlay = cv2.addWeighted(vis_bgr, 1.0, color, 0.35, 0.0)

        y0 = 28
        cv2.putText(overlay, f"GT: {tool_name}", (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        y = y0 + 28
        for rank, (cls_id, p) in enumerate(top_pairs, 1):
            cls_name = id2tool.get(cls_id, str(cls_id))
            cv2.putText(overlay, f"Top{rank}: {cls_name}  {p:.3f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            y += 24

        cv2.imwrite(str(out_dir / f"vis_{i:04d}.jpg"), overlay)
    print(f"[VIS] saved {count} images to: {out_dir.resolve()}")
    try:
        with open(out_dir / "index.txt", "w", encoding="utf-8") as f:
            for i in range(count):
                f.write(str((out_dir / f"vis_{i:04d}.jpg").resolve()) + "\n")
    except Exception as e:
        print(f"[WARN] write index.txt failed: {e}")

# ========================= Main =========================

def _choose_split_csv() -> Path:
    test_csv = SMALLFILE_ROOT / "test_manifest.csv"
    val_csv  = SMALLFILE_ROOT / "val_manifest.csv"
    mf_csv   = SMALLFILE_ROOT / "manifest.csv"
    if test_csv.exists(): return test_csv
    if val_csv.exists():  return val_csv
    return mf_csv

def main():
    ap = argparse.ArgumentParser()
    # which dataset to eval
    ap.add_argument("--eval-set", choices=["prepared", "ood_raw"], default="prepared",
                    help="'prepared' uses existing CSV; 'ood_raw' builds OOD from prompts.json")
    ap.add_argument("--videos", type=str,
                    default="video01,video02,video03,video04,video05,video06,video07,video08,video09,video10",
                    help="Only for --eval-set ood_raw; comma-separated video ids.")
    ap.add_argument("--raw-root", type=str,
                    default="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data",
                    help="Root of annotated_data for OOD.")
    ap.add_argument("--ood-out-csv", type=str, default=str(SMALLFILE_ROOT / "ood_manifest.csv"))
    ap.add_argument("--ood-label-map", type=str, default=str(SMALLFILE_ROOT / "label_map_ood.json"))

    # model & ckpt
    ap.add_argument("--resume-mode", choices=["head", "full"], default="head",
                    help="'head' loads head-only ckpt (SAM2 frozen). 'full' loads finetuned {sam2+head}.")
    ap.add_argument("--ckpt", type=str, default=str(CKPT_ROOT / "best_head.pt"),
                    help="Path to checkpoint. head-mode: best_head.pt; full-mode: best_full_finetune.pt")
    ap.add_argument("--head-override", type=str, default=None,
                    help="Optional override ('linear','mlp','mlp_bn','cosine') if ckpt args missing.")

    # sam2 backbones
    ap.add_argument("--sam2-cfg", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_l.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_large.pt"))

    # general
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num-vis", type=int, default=50)
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- build/select dataset ---------
    if args.eval_set == "prepared":
        label_map_path = SMALLFILE_ROOT / "label_map.json"
        if not label_map_path.exists():
            raise FileNotFoundError(f"[prepared] missing label_map: {label_map_path}")
        label_map = load_label_map(label_map_path)
        split_csv = _choose_split_csv()
        if not split_csv.exists():
            raise FileNotFoundError(f"[prepared] cannot find CSV: {split_csv}")
        df = pd.read_csv(split_csv)
        print(f"[INFO] Evaluating (prepared): {split_csv}  (#rows={len(df)})")
        ds = FramePointDataset(split_csv, label_map_path, resize=None)
    else:
        # OOD raw
        ood_label_map_path = Path(args.ood_label_map)
        if not ood_label_map_path.exists():
            _save_json({"tool_to_id": TOOL_TO_ID_OOD}, ood_label_map_path)
            print(f"[SAVE] OOD label_map -> {ood_label_map_path}")
        else:
            lm_tmp = _load_json(ood_label_map_path)
            assert lm_tmp.get("tool_to_id", {}) == TOOL_TO_ID_OOD, "OOD label_map mismatch"
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
        ood_csv = Path(args.ood_out_csv)
        if not ood_csv.exists():
            _build_ood_manifest(Path(args.raw_root), videos, ood_csv)
        else:
            print(f"[INFO] Using existing OOD manifest: {ood_csv}")
        df = pd.read_csv(ood_csv)
        print(f"[INFO] Evaluating (OOD raw): {ood_csv}  (#rows={len(df)})")
        label_map_path = ood_label_map_path
        label_map = _load_json(label_map_path)
        ds = OODDataset(ood_csv, label_map_path, resize=None)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_varlen, pin_memory=True)

    # --------- build model ---------
    extractor = Sam2OfficialWrapper(args.sam2_cfg, args.sam2_ckpt, device=device)

    # probe feature dim
    probe = next(iter(dl))
    with torch.no_grad():
        feat_probe = extractor(probe["images"][:1], probe["points"][:1])
    in_dim = int(feat_probe.shape[-1])
    n_classes = len(label_map["tool_to_id"])

    # load ckpt
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # build head
    ck_args = ckpt.get("args", {})
    in_dim_ck = int(ckpt.get("in_dim", in_dim))
    n_classes_ck = int(ckpt.get("n_classes", n_classes))
    if n_classes_ck != n_classes:
        print(f"[WARN] n_classes mismatch: ckpt={n_classes_ck}, current={n_classes}. Using current={n_classes}.")
        n_classes_ck = n_classes
    head = _build_head_from_ckpt_args(in_dim_ck, n_classes_ck, ck_args, override_head=args.head_override).to(device)

    if args.resume_mode == "full":
        # Expect ckpt contains 'sam2_state' and 'head_state' (finetuned)
        sam2_state = ckpt.get("sam2_state", None)
        head_state = ckpt.get("head_state", None) or ckpt  # fallback
        if sam2_state is not None:
            missing, unexpected = extractor.model.load_state_dict(sam2_state, strict=False)
            if missing or unexpected:
                print(f"[LOAD][SAM2] missing={len(missing)} unexpected={len(unexpected)}")
            print(f"[RESUME] Loaded finetuned SAM2 from: {ckpt_path}")
        else:
            print(f"[WARN] No 'sam2_state' in {ckpt_path}; proceeding with base SAM2 weights.")
        head.load_state_dict(head_state, strict=False)
        print(f"[RESUME] Loaded head state (full mode) from: {ckpt_path}")
    else:
        # head-only
        state = ckpt.get("head_state", ckpt)
        head.load_state_dict(state, strict=False)
        print(f"[RESUME] Loaded head state (head-only mode) from: {ckpt_path}")

    # --------- evaluate ---------
    test_loss, overall_acc, per_class_acc, per_class_cnt, y_true, y_pred = evaluate_batchwise(
        extractor, head, dl, device
    )

    id2tool = {int(v): str(k) for k, v in label_map["tool_to_id"].items()}
    print(f"\n=== Overall ===\nLoss: {test_loss:.4f}  Acc: {overall_acc:.4f}  (#samples={len(y_true)})")
    print("\n=== Per-class Acc ===")
    for cid in sorted(per_class_acc.keys()):
        cname = id2tool.get(cid, str(cid))
        cnt   = per_class_cnt.get(cid, 0)
        print(f"{cid:3d} {cname:>20s}: acc={per_class_acc[cid]:.4f}  (n={cnt})")

    # --------- visualize ---------
    vis_dir = Path(args.out_dir) / ("vis_test_ood" if args.eval_set=="ood_raw" else "vis_test")
    visualize_samples(extractor, head, df, label_map, vis_dir,
                      num_vis=args.num_vis, conf_thr=args.conf_thr, topk=args.topk, device=device)

if __name__ == "__main__":
    main()

# 评测你现成的 CSV（SAM2 冻结，仅加载 head）
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --eval-set prepared \
#   --resume-mode head \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_head.pt \
#   --sam2-cfg /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --batch-size 128 --num-vis 50


# 评测 OOD（前 10 个视频，可改），使用 finetuned 的 SAM2+MLP 联合权重
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --eval-set ood_raw \
#   --videos "video01,video02,video03,video04,video05,video06,video07,video08,video09,video10" \
#   --resume-mode full \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_full_finetune.pt \
#   --sam2-cfg /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --batch-size 128 --num-vis 50


# 评测 OOD，但只加载老的 head（SAM2 用原始权重）
# python /home/wcheng31/sam2_classify/test_sam2_classify.py \
#   --eval-set ood_raw \
#   --resume-mode head \
#   --ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/best_head.pt
