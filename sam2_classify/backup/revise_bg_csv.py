#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact background rows in CSVs: keep at most
  - 1 'contact-merged' background row (choose the background row with the most points),
  - 1 'random-point' background row (prefer an existing 1-point BG row, otherwise synthesize one).
All tool rows are kept unchanged.

Inputs (default --in-config):
  /projects/surgical-video-digital-twin/datasets/sam2_classifier/config/{manifest.csv,train_manifest.csv,val_manifest.csv,test_manifest.csv}

Outputs (default --out-config):
  /home/wcheng31/sam2_classify/config/{manifest.csv,train_manifest.csv,val_manifest.csv,test_manifest.csv}

Notes:
- Pure CSV transform; no re-slicing of frames.
- Deterministic random BG with --seed (per image, hash-based offset).

Author: ChatGPT
"""
import argparse, json, os, hashlib, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import cv2

BACKGROUND_NAME = "background"

def _parse_points_json(s: str) -> List[List[float]]:
    try:
        if isinstance(s, str) and s.strip():
            arr = json.loads(s)
            if isinstance(arr, list):
                return arr
    except Exception:
        pass
    return []

def _pick_contact_bg_row(rows_bg: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """选择点数最多的 background 行，作为 contact-merged 行。若并列，取第一个。"""
    if not rows_bg:
        return None
    best = max(rows_bg, key=lambda r: int(r.get("num_points_pos", 0)))
    return best

def _find_existing_one_point_bg(rows_bg: List[Dict[str, Any]], exclude_id: Optional[int]) -> Optional[Dict[str, Any]]:
    """在 BG 行里找 1 点的行；可排除 contact 行（exclude_id 使用原行的 id()）。"""
    for r in rows_bg:
        if exclude_id is not None and id(r) == exclude_id:
            continue
        try:
            if int(r.get("num_points_pos", 0)) == 1:
                # 再稳妥检查 points_json 确实一个点
                pts = _parse_points_json(r.get("points_json", "[]"))
                if len(pts) == 1:
                    return r
        except Exception:
            continue
    return None

def _make_synth_random_bg_row(ref_meta: Dict[str, Any], img_path: str, seed: int) -> Optional[Dict[str, Any]]:
    """如果没有 1 点 BG 行，则在整图中生成一个随机点的 BG 行（1 点）。"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]

    # 基于 image_path+seed 生成可复现随机数
    sig = hashlib.md5((img_path + f"#{seed}").encode("utf-8")).hexdigest()
    rnd = random.Random(int(sig[:12], 16))
    x = rnd.uniform(0, max(1, W - 1))
    y = rnd.uniform(0, max(1, H - 1))

    r = dict(ref_meta)
    r["tool"] = BACKGROUND_NAME
    r["num_points_pos"] = 1
    r["num_points_neg"] = 0
    r["points_json"] = json.dumps([[float(x), float(y), 1.0]], ensure_ascii=False)
    return r

def _compact_for_one_image(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """
    对同一张 image_path 的所有行进行精简：
      - 非 background 行全保留；
      - background 行：保留 1 个 contact-merged（点数最多的那行），外加 1 个随机点 BG（优先复用既有1点BG，缺则合成）。
    """
    if not rows:
        return []

    # 拆分工具/BG
    tool_rows = [r for r in rows if str(r.get("tool")) != BACKGROUND_NAME]
    bg_rows   = [r for r in rows if str(r.get("tool")) == BACKGROUND_NAME]

    out = list(tool_rows)

    if bg_rows:
        # contact 汇总：选点数最多的行
        contact_row = _pick_contact_bg_row(bg_rows)
        if contact_row is not None:
            out.append(contact_row)

        # 1 点 BG：优先复用既有 1 点行（排除 contact_row 本身）
        one_point_row = _find_existing_one_point_bg(bg_rows, exclude_id=id(contact_row) if contact_row else None)
        if one_point_row is not None:
            out.append(one_point_row)
        else:
            # 没有现成的一点 BG，则合成一个
            ref = contact_row if contact_row is not None else (tool_rows[0] if tool_rows else bg_rows[0])
            img_path = str(ref.get("image_path", ""))
            synth = _make_synth_random_bg_row(ref, img_path, seed=seed)
            if synth is not None:
                out.append(synth)

    # 按列顺序返回
    return out

def _preserve_columns(df: pd.DataFrame, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """按输入 df 的列顺序构建输出 df，缺失列填空，多余字段丢弃。"""
    cols = list(df.columns)
    fixed_rows = []
    for r in rows:
        fixed = {c: r.get(c, "") for c in cols}
        fixed_rows.append(fixed)
    return pd.DataFrame(fixed_rows, columns=cols)

def process_csv(in_path: Path, out_path: Path, seed: int):
    if not in_path.exists():
        print(f"[SKIP] {in_path} not found.")
        return

    print(f"[READ] {in_path}")
    df = pd.read_csv(in_path)

    # 以 image_path 分组处理
    grouped = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        ip = str(d.get("image_path", ""))
        grouped.setdefault(ip, []).append(d)

    new_rows: List[Dict[str, Any]] = []
    for img_path, rows in grouped.items():
        new_rows.extend(_compact_for_one_image(rows, seed=seed))

    df_new = _preserve_columns(df, new_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}  (rows: {len(df_new)})")

def main():
    ap = argparse.ArgumentParser("Compact background rows in SAM2 classifier CSVs")
    ap.add_argument("--in-config", type=str,
                    default="/projects/surgical-video-digital-twin/datasets/sam2_classifier/config",
                    help="Input CSV directory (contains manifest/train/val/test CSVs)")
    ap.add_argument("--out-config", type=str,
                    default="/home/wcheng31/sam2_classify/config",
                    help="Output CSV directory to write the compacted CSVs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for synthetic BG points")
    args = ap.parse_args()

    in_dir = Path(args.in_config)
    out_dir = Path(args.out_config)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 逐个处理 4 份 CSV（存在哪个就处理哪个）
    mapping = {
        "manifest.csv":       "manifest.csv",
        "train_manifest.csv": "train_manifest.csv",
        "val_manifest.csv":   "val_manifest.csv",
        "test_manifest.csv":  "test_manifest.csv",
    }

    for in_name, out_name in mapping.items():
        process_csv(in_dir / in_name, out_dir / out_name, seed=args.seed)

    print("Done.")

if __name__ == "__main__":
    main()


# python /home/wcheng31/sam2_classify/backup/revise_bg_csv.py
