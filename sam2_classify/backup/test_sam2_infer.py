#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 + CSV 点提示 -> 掩码推理 & 可视化（无分类头，使用官方 video_predictor，坐标对齐正确）
"""

import os, json, argparse, shutil, random
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"

# 官方 predictor（与你给的“正确 infer”一致）
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize
from sam2.backup.build_sam import build_sam2_video_predictor


# --------------------- Utils ---------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p: Path):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_label_map(p: Path) -> Dict[str, Dict[str, int]]:
    return load_json(p)

TOOL_TO_ID_OOD = {"background":0, "clipper":1, "grasper":2, "hook":3, "scissors":4}

def _map_obj_id_to_cls(obj_id: int) -> str:
    if obj_id in (2,3,4): return "grasper"
    if obj_id == 6:       return "hook"
    if obj_id == 7:       return "scissors"
    if obj_id == 8:       return "clipper"
    return "background"

def _build_ood_manifest(raw_root: Path, videos: List[str], out_csv: Path) -> pd.DataFrame:
    rows = []
    for vid in videos:
        vid_dir = raw_root / vid / "ws_0"
        img_dir = vid_dir / "images"
        prm = vid_dir / "prompts.json"
        if not prm.exists():
            print(f"[WARN] missing prompts: {prm}")
            continue
        data = load_json(prm)
        for rec in data:
            frame_file = rec.get("frame_file")
            if not frame_file:
                fid = rec.get("frame_id", None)
                frame_file = f"{int(fid):07d}.jpg" if fid is not None else None
            if not frame_file:
                continue
            img_path = img_dir / frame_file
            if not img_path.exists():
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

def _get_sam_module_from_predictor(predictor):
    # 尝试多种常见名称，返回真正的 SAM2 nn.Module
    for name in ["model", "sam_model", "sam2", "sam"]:
        m = getattr(predictor, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    # 实在不行就直接返回 predictor 本体（load_state_dict(strict=False) 也能跳过不匹配）
    return predictor

def load_finetuned_into_predictor(predictor, ft_path: str):
    ckpt = torch.load(ft_path, map_location="cpu")
    state = ckpt.get("sam2_state", ckpt)  # 兼容：有的是 dict，有的是直接 state_dict
    # 去掉可能的 "model." 前缀，方便和 predictor 内部命名对齐
    new_state = {}
    for k, v in state.items():
        new_state[k[6:]] = v if k.startswith("model.") else v  # 简单去前缀；不去也没关系，strict=False 会跳过
    sam = _get_sam_module_from_predictor(predictor)
    missing, unexpected = sam.load_state_dict(new_state, strict=False)
    print(f"[LOAD-FT] loaded finetuned SAM2 from: {ft_path}")
    if missing:   print(f"[LOAD-FT] missing keys: {len(missing)} (OK if仅分类头/不相关模块)")
    if unexpected:print(f"[LOAD-FT] unexpected keys: {len(unexpected)} (OK if来源不同包名)")


class FramePointDataset(Dataset):
    def __init__(self, manifest_csv: Path, label_map_json: Path):
        super().__init__()
        self.df = pd.read_csv(manifest_csv)
        lm = load_label_map(label_map_json)
        self.tool2id = lm["tool_to_id"]

        def has_points(s: str) -> bool:
            try:
                arr = json.loads(s) if isinstance(s, str) and s.strip() else []
                return len(arr) > 0
            except Exception:
                return False
        self.df = self.df[self.df["points_json"].apply(has_points)].reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        H, W = img.shape[:2]
        pts = json.loads(row["points_json"])
        pts_out = []
        for p in pts:
            x = float(np.clip(p[0], 0, W-1)); y = float(np.clip(p[1], 0, H-1))
            label = 1.0 if float(p[2]) > 0 else 0.0
            pts_out.append([x, y, label])
        tool = str(row["tool"])
        tool_id = int(self.tool2id[tool])
        return {"image": img, "image_path": img_path, "points": np.array(pts_out, dtype=np.float32),
                "tool": tool, "tool_id": tool_id}

def collate_varlen(batch):
    return batch


# ---------------- 背景策略 ----------------
def _prep_points_for_bg(points: np.ndarray, tool_id: int, bg_mask_mode: str, bg_mix_p: float) -> np.ndarray:
    if tool_id != 0:
        return points
    mode = bg_mask_mode
    if mode == "mix":
        mode = "global" if (random.random() < float(bg_mix_p)) else "pos"
    if mode == "global":
        return np.zeros((0,3), np.float32)
    pts = np.asarray(points, np.float32).copy()
    if pts.size > 0: pts[:, 2] = 1.0
    return pts


# --------------- predictor（单帧） ---------------
def _init_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize(config_path="configs/sam2", version_base="1.2")

def _build_predictor(cfg: str, ckpt: str, device: torch.device):
    _init_hydra()
    return build_sam2_video_predictor(cfg, ckpt, device=device)

@torch.no_grad()
def infer_mask_one_image_with_predictor(
    predictor,
    img_bgr: np.ndarray,
    pts_np: np.ndarray,
    tool_id: int,
    bg_mask_mode: str,
    bg_mix_p: float,
    tmp_dir: Path,
) -> np.ndarray:
    """
    返回二值 mask (H,W) uint8；若没有点（例如 background+global/mix 采样为空），
    直接返回全 0 掩码，避免 predictor 抛出 'No input points...'。
    """
    H, W = img_bgr.shape[:2]

    # 背景策略（可能把点清空）
    pts_np = _prep_points_for_bg(pts_np, tool_id, bg_mask_mode, bg_mix_p)
    if pts_np is None or pts_np.size == 0:
        # 没有点：直接返回全 0（你也可以改成 np.ones((H,W),np.uint8) 表示“全图”）
        return np.zeros((H, W), dtype=np.uint8)

    # --- 有点才调用 predictor ---
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_img = tmp_dir / "0000000.jpg"
    cv2.imwrite(str(tmp_img), img_bgr)

    state = predictor.init_state(video_path=str(tmp_dir))
    predictor.reset_state(state)

    pts_xy = pts_np[:, :2].astype(np.float32)
    labs   = (pts_np[:, 2] > 0).astype(np.int64)
    oid = 1
    if hasattr(predictor, "add_new_points"):
        predictor.add_new_points(state, 0, oid, pts_xy, labs)
    else:
        predictor.add_new_points_or_box(state, 0, oid, pts_xy, labs)

    mask = np.zeros((H, W), np.uint8)
    for _, obj_ids, logits in predictor.propagate_in_video(state):
        for i, got_oid in enumerate(obj_ids):
            if got_oid == oid:
                m = (logits[i] > 0).detach().cpu().numpy()
                m = np.squeeze(m)
                if m.ndim != 2:
                    m = m.reshape(m.shape[-2], m.shape[-1])
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                mask = np.maximum(mask, (m > 0).astype(np.uint8))

    try:
        tmp_img.unlink(missing_ok=True)
    except Exception:
        pass

    return mask


# --------------- 可视化 ---------------
def _prepare_vis_df(df: pd.DataFrame, num_vis: int, stratified: bool) -> pd.DataFrame:
    if "image_path" in df.columns:
        df = df.drop_duplicates(subset=["image_path"]).copy()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if not stratified or "tool" not in df.columns:
        return df.head(min(num_vis, len(df)))
    k = df["tool"].nunique()
    per_cls = max(1, num_vis // k)
    parts = [g.head(min(per_cls, len(g))) for _, g in df.groupby("tool")]
    df_bal = pd.concat(parts).sample(frac=1.0, random_state=123).reset_index(drop=True)
    return df_bal.head(min(num_vis, len(df_bal)))

@torch.no_grad()
def visualize_masks(
    predictor,
    df: pd.DataFrame,
    label_map: Dict,
    out_dir: Path,
    num_vis: int = 50,
    conf_thr: float = 0.5,
    stratified_vis: bool = True,
    bg_mask_mode: str = "mix",
    bg_mix_p: float = 0.5,
):
    ensure_dir(out_dir)
    df_vis = _prepare_vis_df(df, num_vis=num_vis, stratified=stratified_vis)

    tmp_dir = out_dir / "_tmp_single_frame"
    ensure_dir(tmp_dir)

    pbar = tqdm(range(len(df_vis)), ncols=100, desc="[vis] writing", leave=True)
    for i in pbar:
        row = df_vis.iloc[i]
        img_path = row["image_path"]; tool_name = str(row["tool"])
        tool_id  = int(label_map["tool_to_id"][tool_name])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: continue

        try:
            pts = json.loads(row["points_json"]) if isinstance(row["points_json"], str) and row["points_json"].strip() else []
        except Exception:
            pts = []
        pts_np = np.asarray(pts, np.float32) if len(pts) else np.zeros((0, 3), np.float32)

        # 掩码（已保证为 (H,W)）
        mask = infer_mask_one_image_with_predictor(
            predictor, img, pts_np, tool_id,
            bg_mask_mode=bg_mask_mode, bg_mix_p=bg_mix_p,
            tmp_dir=tmp_dir
        )

        # === 关键修复：保证布尔掩码是 2D，并用于像素索引 ===
        m_bin = (mask.astype(bool))
        right_img = img.copy()
        right_img[m_bin] = 0  # (H,W) 布尔索引 -> (N,3) OK

        for (x, y, lab) in pts_np:
            c = (0, 255, 0) if lab > 0 else (0, 0, 255)
            cv2.circle(right_img, (int(round(x)), int(round(y))), 5, c, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(right_img, (int(round(x)), int(round(y))), 6, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        side_by_side = np.concatenate([img, right_img], axis=1)
        cv2.putText(side_by_side, f"GT: {tool_name}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"vis_{i:04d}.jpg"), side_by_side)

    try: shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception: pass

    print(f"[VIS] saved {len(df_vis)} images to: {out_dir.resolve()}")
    try:
        with open(out_dir / "index.txt", "w", encoding="utf-8") as f:
            for i in range(len(df_vis)):
                f.write(str((out_dir / f"vis_{i:04d}.jpg").resolve()) + "\n")
    except Exception as e:
        print(f"[WARN] write index.txt failed: {e}")


# ---------------- Main ----------------
def _choose_split_csv() -> Path:
    test_csv = SMALLFILE_ROOT / "test_manifest.csv"
    val_csv  = SMALLFILE_ROOT / "val_manifest.csv"
    mf_csv   = SMALLFILE_ROOT / "manifest.csv"
    if test_csv.exists(): return test_csv
    if val_csv.exists():  return val_csv
    return mf_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", choices=["prepared", "ood_raw"], default="prepared")
    ap.add_argument("--videos", type=str,
                    default="video01,video02,video03,video04,video05,video06,video07,video08,video09,video10")
    ap.add_argument("--raw-root", type=str,
                    default="/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data")
    ap.add_argument("--ood-out-csv", type=str, default=str(SMALLFILE_ROOT / "ood_manifest.csv"))
    ap.add_argument("--ood-label-map", type=str, default=str(SMALLFILE_ROOT / "label_map_ood.json"))

    ap.add_argument("--sam2-cfg", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_l.yaml"))
    ap.add_argument("--sam2-ckpt", type=str, default=str(PRETRAIN_ROOT / "sam2_hiera_large.pt"))

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--num-vis", type=int, default=50)
    ap.add_argument("--conf-thr", type=float, default=0.5)
    ap.add_argument("--out-dir", type=str, default=str(CKPT_ROOT))
    ap.add_argument("--stratified-vis", action="store_true")
    ap.add_argument("--vis-points-only", action="store_true")

    ap.add_argument("--bg-mask-mode", choices=["pos","global","mix"], default="mix")
    ap.add_argument("--bg-mix-p", type=float, default=0.5)

    ap.add_argument("--finetuned-ckpt", type=str, default="",
                help="训练脚本导出的 best_full_finetune.pt；若提供则把其中 sam2_state 加载到 predictor")


    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 选择/构建数据
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
        ds = FramePointDataset(split_csv, label_map_path)
    else:
        ood_label_map_path = Path(args.ood_label_map)
        if not ood_label_map_path.exists():
            save_json({"tool_to_id": TOOL_TO_ID_OOD}, ood_label_map_path)
            print(f"[SAVE] OOD label_map -> {ood_label_map_path}")
        else:
            lm_tmp = load_json(ood_label_map_path)
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
        label_map = load_json(label_map_path)
        ds = FramePointDataset(ood_csv, label_map_path)

    _ = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_varlen)  # 仅检查可读性

    # points-only
    if args.vis_points_only:
        out_dir = Path(args.out_dir) / ("vis_points_only_ood" if args.eval_set=="ood_raw" else "vis_points_only")
        ensure_dir(out_dir)
        df_vis = _prepare_vis_df(pd.read_csv(_choose_split_csv()) if args.eval_set=="prepared" else df,
                                 num_vis=args.num_vis, stratified=args.stratified_vis)
        for i, row in tqdm(list(df_vis.iterrows()), ncols=100, desc="[vis] points-only"):
            img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            if img is None: continue
            pts = json.loads(row["points_json"])
            right = img.copy()
            H, W = right.shape[:2]
            for p in pts:
                x = int(np.clip(round(float(p[0])), 0, W-1))
                y = int(np.clip(round(float(p[1])), 0, H-1))
                c = (0,255,0) if float(p[2])>0 else (0,0,255)
                cv2.circle(right, (x,y), 5, c, -1, lineType=cv2.LINE_AA)
                cv2.circle(right, (x,y), 6, (0,0,0), 1, cv2.LINE_AA)
            out = np.concatenate([img, right], axis=1)
            cv2.imwrite(str(out_dir / f"vis_pts_{i:04d}.jpg"), out)
        return

    predictor = _build_predictor(str(args.sam2_cfg), str(args.sam2_ckpt), device=device)
    if args.finetuned-ckpt:
        load_finetuned_into_predictor(predictor, args.finetuned_ckpt)
    print("SAM-2 predictor ready.\n")

    vis_dir = Path(args.out_dir) / ("vis_test_ood" if args.eval_set=="ood_raw" else "vis_test")
    vis_df  = pd.read_csv(_choose_split_csv()) if args.eval_set=="prepared" else df
    visualize_masks(
        predictor,
        vis_df,
        load_label_map(label_map_path),
        vis_dir,
        num_vis=args.num_vis,
        conf_thr=args.conf_thr,
        stratified_vis=args.stratified_vis,
        bg_mask_mode=args.bg_mask_mode,
        bg_mix_p=args.bg_mix_p,
    )

if __name__ == "__main__":
    main()

# python /home/wcheng31/sam2_classify/test_sam2_infer.py   --eval-set prepared   --sam2-cfg sam2_hiera_l.yaml   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt   --batch-size 256 --workers 4   --bg-mask-mode mix --bg-mix-p 0.5   --stratified-vis   --num-vis 50   --out-dir /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier

# python /home/wcheng31/sam2_classify/test_sam2_infer.py \
#   --eval-set prepared \
#   --sam2-cfg sam2_hiera_t.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_tiny.pt \
#   --finetuned-ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/distill_maskcls_t/best_full_finetune.pt \
#   --num-vis 100 --stratified-vis \
#   --bg-mask-mode mix --bg-mix-p 0.5 \
#   --out-dir /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/vis_finetuned_tiny
