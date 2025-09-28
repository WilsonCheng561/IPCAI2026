#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV pipeline:
  1) Resplit by VIDEO (non-overlap) -> 8:1:1
  2) Compact background per image: keep 1 contact-merged + 1 random-1pt
  3) Per-class downsample by image groups:
       default factors: background=10, hook=5, clipper=1, grasper=1, scissors=1

Inputs  (read):
  /home/wcheng31/sam2_classify/config/{manifest.csv,train_manifest.csv,val_manifest.csv,test_manifest.csv}

Outputs (write; suffix controlled by --suffix, default "10"):
  /home/wcheng31/sam2_classify/config/{manifest,train_manifest,val_manifest,test_manifest}_<suffix>.csv
"""

import argparse, json, re, hashlib, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import cv2

CFG_DIR_DEFAULT = Path("/home/wcheng31/sam2_classify/config")
BACKGROUND_NAME = "background"

# -------------------- helpers: parse / sort keys --------------------
FRAME_NUM_RE = re.compile(r"frame[_\-]?(\d+)\.(?:jpg|png|jpeg)$", re.IGNORECASE)
VID_PATTERNS = [
    re.compile(r"(cholec80_\d+)", re.IGNORECASE),
    re.compile(r"(case_\d+_video_part_\d+_segment_\d+)", re.IGNORECASE),
]

def _extract_frame_num(p: str) -> int:
    if not isinstance(p, str): return 10**12
    m = FRAME_NUM_RE.search(p)
    if not m: return 10**12
    try: return int(m.group(1))
    except Exception: return 10**12

def _sort_key_for_group(df_group: pd.DataFrame) -> Tuple:
    r0 = df_group.iloc[0]
    task = str(r0.get("task",""))
    clip = str(r0.get("clip_name",""))
    if "frame_abs_index" in df_group.columns:
        fa = int(df_group["frame_abs_index"].min())
    else:
        fa = min(_extract_frame_num(x) for x in df_group["image_path"].tolist())
    return (task, clip, fa, str(r0.get("image_path","")))

def _extract_video_id(row: dict) -> str:
    task = str(row.get("task","")).lower()
    for pat in VID_PATTERNS:
        m = pat.search(task)
        if m: return m.group(1)
    clip = str(row.get("clip_name","")).lower()
    for pat in VID_PATTERNS:
        m = pat.search(clip)
        if m: return m.group(1)
    ip = str(row.get("image_path","")).lower()
    parts = [p for p in ip.split("/") if p]
    if "images" in parts:
        try:
            idx = parts.index("images")
            if idx + 1 < len(parts): return parts[idx+1]
        except Exception:
            pass
    return task if task else "unknown_video"

# -------------------- Step 1: resplit by VIDEO --------------------
def resplit_by_video(df_all: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, seed=42):
    df = df_all.copy()
    df["video_id"] = [ _extract_video_id(r.to_dict()) for _, r in df.iterrows() ]
    vids = sorted(df["video_id"].unique().tolist())
    rng = random.Random(seed); rng.shuffle(vids)
    n = len(vids); n_tr = min(int(round(n*train_ratio)), n); n_va = min(int(round(n*val_ratio)), max(0,n-n_tr))
    v_tr = set(vids[:n_tr]); v_va = set(vids[n_tr:n_tr+n_va]); v_te = set(vids[n_tr+n_va:])
    df_tr = df[df["video_id"].isin(v_tr)].drop(columns=["video_id"]).reset_index(drop=True)
    df_va = df[df["video_id"].isin(v_va)].drop(columns=["video_id"]).reset_index(drop=True)
    df_te = df[df["video_id"].isin(v_te)].drop(columns=["video_id"]).reset_index(drop=True)
    return df_tr, df_va, df_te

# -------------------- Step 2: compact BG per image --------------------
def _parse_points_json(s: str) -> List[List[float]]:
    try:
        if isinstance(s, str) and s.strip():
            arr = json.loads(s)
            if isinstance(arr, list): return arr
    except Exception: pass
    return []

def _pick_contact_bg_row(rows_bg: List[dict]) -> Optional[dict]:
    if not rows_bg: return None
    return max(rows_bg, key=lambda r: int(r.get("num_points_pos", 0)))

def _find_one_pt_bg(rows_bg: List[dict], exclude_id: Optional[int]) -> Optional[dict]:
    for r in rows_bg:
        if exclude_id is not None and id(r) == exclude_id: continue
        try:
            if int(r.get("num_points_pos", 0)) == 1 and len(_parse_points_json(r.get("points_json","[]"))) == 1:
                return r
        except Exception:
            continue
    return None

def _synth_one_pt_bg(ref_meta: dict, img_path: str, seed: int) -> Optional[dict]:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None
    H,W = img.shape[:2]
    sig = hashlib.md5((img_path+f"#{seed}").encode("utf-8")).hexdigest()
    rnd = random.Random(int(sig[:12],16))
    x = rnd.uniform(0, max(1,W-1)); y = rnd.uniform(0, max(1,H-1))
    r = dict(ref_meta)
    r["tool"] = BACKGROUND_NAME
    r["num_points_pos"] = 1; r["num_points_neg"] = 0
    r["points_json"] = json.dumps([[float(x), float(y), 1.0]], ensure_ascii=False)
    return r

def compact_bg_for_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    cols = list(df.columns)
    groups: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        r = row.to_dict()
        groups.setdefault(str(r.get("image_path","")), []).append(r)

    new_rows: List[dict] = []
    for imgp, rows in groups.items():
        tools = [r for r in rows if str(r.get("tool")) != BACKGROUND_NAME]
        bgs   = [r for r in rows if str(r.get("tool")) == BACKGROUND_NAME]
        out = list(tools)
        if bgs:
            contact = _pick_contact_bg_row(bgs)
            if contact is not None: out.append(contact)
            one = _find_one_pt_bg(bgs, exclude_id=id(contact) if contact else None)
            if one is not None:
                out.append(one)
            else:
                ref = contact if contact is not None else (tools[0] if tools else bgs[0])
                synth = _synth_one_pt_bg(ref, imgp, seed)
                if synth is not None: out.append(synth)
        # keep column order
        new_rows.extend([{c: rr.get(c,"") for c in cols} for rr in out])

    return pd.DataFrame(new_rows, columns=cols)

# -------------------- Step 3: per-class downsample (image-level per class) --------------------
def downsample_per_class(df: pd.DataFrame, factors: Dict[str,int]) -> pd.DataFrame:
    """
    对每个类别 c：
      - 取子集 df_c = df[df.tool==c]
      - 按 image_path 分组 -> 按时序排序 -> 每 factor[c] 张图取一张
      - 只保留该类别在这些图上的行
    最终把所有类别的子集 concat（同一张图其他类别的行不会被“误删”）。
    """
    cols = list(df.columns)
    outs = []
    for cls, fac in factors.items():
        sub = df[df["tool"] == cls]
        if len(sub) == 0:
            continue
        groups = {k:g for k,g in sub.groupby("image_path", sort=False)}
        ordered = sorted(groups.keys(), key=lambda k: _sort_key_for_group(groups[k]))
        fac = max(1, int(fac))
        keep_keys = set(ordered[::fac])
        kept = [groups[k] for k in ordered if k in keep_keys]
        if kept:
            outs.append(pd.concat(kept, axis=0))
    if not outs:
        return df.copy()
    out = pd.concat(outs, axis=0).reset_index(drop=True)
    # 保持列顺序
    return out[cols]

# -------------------- IO & pipeline --------------------
def _load_all(cfg_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    man = cfg_dir / "manifest.csv"
    if man.exists():
        df_all = pd.read_csv(man)
    else:
        parts = []
        for name in ["train_manifest.csv","val_manifest.csv","test_manifest.csv"]:
            p = cfg_dir / name
            if p.exists(): parts.append(pd.read_csv(p))
        if not parts:
            raise SystemExit(f"No manifest.csv or train/val/test csv in {cfg_dir}")
        df_all = pd.concat(parts, ignore_index=True)
    # 同时尝试读取原始 train/val/test（若有的话给信息但不用信任它们）
    df_tr = pd.read_csv(cfg_dir/"train_manifest.csv") if (cfg_dir/"train_manifest.csv").exists() else None
    df_va = pd.read_csv(cfg_dir/"val_manifest.csv")   if (cfg_dir/"val_manifest.csv").exists()   else None
    df_te = pd.read_csv(cfg_dir/"test_manifest.csv")  if (cfg_dir/"test_manifest.csv").exists()  else None
    return df_all, df_tr, df_va, df_te

def main():
    ap = argparse.ArgumentParser("CSV pipeline: video-split + compact BG + per-class downsample")
    ap.add_argument("--config-dir", type=str, default=str(CFG_DIR_DEFAULT),
                    help="Dir that contains manifest/train/val/test CSVs (read from here, write to here).")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    # per-class factors (image-level)
    ap.add_argument("--bg",        type=int, default=10, help="downsample factor for background")
    ap.add_argument("--hook",      type=int, default=5,  help="downsample factor for hook")
    ap.add_argument("--clipper",   type=int, default=1,  help="downsample factor for clipper")
    ap.add_argument("--grasper",   type=int, default=1,  help="downsample factor for grasper")
    ap.add_argument("--scissors",  type=int, default=1,  help="downsample factor for scissors")
    # filename suffix
    ap.add_argument("--suffix",    type=str, default="10", help="suffix for output csv names, e.g., _10.csv")
    args = ap.parse_args()

    cfg = Path(args.config_dir)
    df_all, _, _, _ = _load_all(cfg)

    # Step 1: video-based split
    df_tr, df_va, df_te = resplit_by_video(df_all, args.train_ratio, args.val_ratio, args.seed)

    # Step 2: compact BG
    df_all_c = compact_bg_for_df(df_all, seed=args.seed)
    df_tr_c  = compact_bg_for_df(df_tr,  seed=args.seed)
    df_va_c  = compact_bg_for_df(df_va,  seed=args.seed)
    df_te_c  = compact_bg_for_df(df_te,  seed=args.seed)

    # Step 3: per-class downsample (image-level per class)
    factors = {
        BACKGROUND_NAME: max(1, int(args.bg)),
        "hook":          max(1, int(args.hook)),
        "clipper":       max(1, int(args.clipper)),
        "grasper":       max(1, int(args.grasper)),
        "scissors":      max(1, int(args.scissors)),
    }
    df_all_ds = downsample_per_class(df_all_c, factors)
    df_tr_ds  = downsample_per_class(df_tr_c,  factors)
    df_va_ds  = downsample_per_class(df_va_c,  factors)
    df_te_ds  = downsample_per_class(df_te_c,  factors)

    # Write with suffix
    def _w(df: pd.DataFrame, name: str):
        out = cfg / f"{name}_{args.suffix}.csv"
        df.to_csv(out, index=False)
        print(f"[WRITE] {out} (rows={len(df)})")
    _w(df_all_ds, "manifest")
    _w(df_tr_ds,  "train_manifest")
    _w(df_va_ds,  "val_manifest")
    _w(df_te_ds,  "test_manifest")

    print("[DONE] Pipeline finished.")

if __name__ == "__main__":
    main()


# 现在的配置：BG/10，Hook/5，其余不降采样；输出 *_10.csv
# python /home/wcheng31/sam2_classify/prepare_csv.py \
#   --config-dir /home/wcheng31/sam2_classify/config \
#   --bg 10 --hook 5 --clipper 1 --grasper 1 --scissors 1 \
#   --suffix 10
