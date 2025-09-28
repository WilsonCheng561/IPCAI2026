#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downsample SAM2 classifier CSVs by **image-level**:
- Sort rows by image order (prefer frame_abs_index; fallback to parse from image_path like frame_000123.jpg).
- Keep every N-th image (default N=10), and keep **all rows** belonging to those selected images.
- Apply independently to {manifest, train_manifest, val_manifest, test_manifest}.
- Write to new files with suffix _{N}.csv (e.g., manifest_10.csv). Original CSVs are untouched.

Usage:
python /home/wcheng31/sam2_classify/downsample_by_image.py \
  --config-dir /home/wcheng31/sam2_classify/config \
  --factor 10
"""

import argparse, re, sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

CSV_NAMES = [
    "manifest.csv",
    "train_manifest.csv",
    "val_manifest.csv",
    "test_manifest.csv",
]

FRAME_NUM_RE = re.compile(r"frame[_\-]?(\d+)\.(?:jpg|png|jpeg)$", re.IGNORECASE)

def _extract_frame_num(image_path: str) -> int:
    """
    Try to parse .../frame_000123.jpg -> 123
    If not found, return a big number to push to the end (but stable).
    """
    if not isinstance(image_path, str):
        return 10**12
    m = FRAME_NUM_RE.search(image_path)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 10**12
    return 10**12

def _image_sort_key(rows_for_image: pd.DataFrame) -> Tuple:
    """
    Sorting priority:
      (task, clip_name, min(frame_abs_index), image_path)
    If frame_abs_index missing, fallback to parsing from image_path.
    """
    row0 = rows_for_image.iloc[0]
    task = str(row0.get("task", ""))
    clip = str(row0.get("clip_name", ""))
    if "frame_abs_index" in rows_for_image.columns:
        fa_min = int(rows_for_image["frame_abs_index"].min())
    else:
        # fallback: parse numeric from image_path
        fa_min = min(_extract_frame_num(p) for p in rows_for_image["image_path"].tolist())
    imgp = str(row0.get("image_path", ""))
    return (task, clip, fa_min, imgp)

def _downsample_one_csv(in_csv: Path, factor: int) -> Path:
    """
    Read a CSV, group by image_path, sort groups by temporal order,
    keep every Nth image group, and dump to *_N.csv.
    """
    if not in_csv.exists():
        print(f"[SKIP] {in_csv} not found.")
        return in_csv.with_name(in_csv.stem + f"_{factor}.csv")

    print(f"[READ] {in_csv}")
    df = pd.read_csv(in_csv)
    if len(df) == 0:
        out_csv = in_csv.with_name(in_csv.stem + f"_{factor}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[WRITE] {out_csv} (empty source)")
        return out_csv

    # group by image_path (image-level)
    groups: Dict[str, pd.DataFrame] = {k: g for k, g in df.groupby("image_path", sort=False)}
    # sort groups by (task, clip_name, frame_abs_index or parsed number, image_path)
    ordered_keys = sorted(groups.keys(),
                          key=lambda k: _image_sort_key(groups[k]))

    # take every Nth image
    keep_keys = set(ordered_keys[::factor])

    # concat rows for kept images only
    kept_rows: List[pd.DataFrame] = []
    for k in ordered_keys:
        if k in keep_keys:
            kept_rows.append(groups[k])

    df_out = pd.concat(kept_rows, axis=0).reset_index(drop=True)

    # preserve column order and write
    out_csv = in_csv.with_name(in_csv.stem + f"_{factor}.csv")
    df_out.to_csv(out_csv, index=False)

    # report
    n_img_total = len(ordered_keys)
    n_img_kept = len(keep_keys)
    print(f"[DONE] {in_csv.name}: images {n_img_kept}/{n_img_total} "
          f"({(n_img_kept/max(1,n_img_total))*100:.1f}%), rows {len(df_out)}/{len(df)}")
    print(f"[WRITE] {out_csv}")
    return out_csv

def main():
    ap = argparse.ArgumentParser("Downsample CSVs by image (keep every N-th image).")
    ap.add_argument("--config-dir", type=str, default="/home/wcheng31/sam2_classify/config",
                    help="Directory containing manifest/train/val/test CSVs.")
    ap.add_argument("--factor", type=int, default=10, help="Keep every N-th image (default: 10).")
    args = ap.parse_args()

    cfg = Path(args.config_dir)
    if not cfg.exists():
        print(f"[ERR] config dir not found: {cfg}")
        sys.exit(1)

    for name in CSV_NAMES:
        _downsample_one_csv(cfg / name, args.factor)

    print("All done.")

if __name__ == "__main__":
    main()


# python /home/wcheng31/sam2_classify/backup/downsample_10_csv.py \
#   --config-dir /home/wcheng31/sam2_classify/config \
#   --factor 10
