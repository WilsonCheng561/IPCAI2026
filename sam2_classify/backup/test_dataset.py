# import pandas as pd
# from pathlib import Path

# #这是prepared data的后一步，验证数据集划分train，val，test是否有重叠，检查toy那个数据集是否有缺失

# root = Path("/home/wcheng31/sam2_classify")
# df_tr = pd.read_csv(root/"train_manifest.csv")
# df_va = pd.read_csv(root/"val_manifest.csv")
# df_te = pd.read_csv(root/"test_manifest.csv")

# def count_overlap(a, b, keys):
#     ma = a[keys].drop_duplicates()
#     mb = b[keys].drop_duplicates()
#     m = ma.merge(mb, on=keys, how="inner")
#     return len(m), len(ma), len(mb)

# # 1) 最严格：image_path + tool
# print("overlap (image_path, tool)")
# for name, X in [("train∩test", (df_tr, df_te)), ("val∩test", (df_va, df_te)), ("train∩val", (df_tr, df_va))]:
#     n, na, nb = count_overlap(X[0], X[1], ["image_path","tool"])
#     print(f"{name}: {n} / ({na}, {nb})")

# # 2) 如果上面没重叠，再看 clip 级别（同 clip 高相似也会漏）
# print("\noverlap by clip_name")
# for name, X in [("train∩test", (df_tr, df_te)), ("val∩test", (df_va, df_te)), ("train∩val", (df_tr, df_va))]:
#     n, na, nb = count_overlap(X[0], X[1], ["clip_name","tool"])
#     print(f"{name}: {n} / ({na}, {nb})")

# # 3) 看同一帧 id（如果保留了）
# if "frame_abs_index" in df_tr.columns and "frame_abs_index" in df_te.columns:
#     print("\noverlap by (clip_name, frame_abs_index, tool)")
#     for name, X in [("train∩test", (df_tr, df_te)), ("val∩test", (df_va, df_te)), ("train∩val", (df_tr, df_va))]:
#         n, na, nb = count_overlap(X[0], X[1], ["clip_name","frame_abs_index","tool"])
#         print(f"{name}: {n} / ({na}, {nb})")


# # import os, pandas as pd
# # root="/home/wcheng31/sam2_classify"
# # for name in ["train_manifest.csv","val_manifest.csv","test_manifest.csv"]:
# #     p=f"{root}/{name}"
# #     if not os.path.exists(p): 
# #         print(name, "not found"); 
# #         continue
# #     df=pd.read_csv(p)
# #     bad = ~df["image_path"].apply(os.path.exists)
# #     print(f"{name}: missing {bad.sum()} / {len(df)}")

# import os, pandas as pd
# root="/home/wcheng31/sam2_classify"
# for name in ["train_manifest.csv","val_manifest.csv","test_manifest.csv"]:
#     p=f"{root}/{name}"
#     if not os.path.exists(p): 
#         continue
#     df=pd.read_csv(p)
#     keep = df["image_path"].apply(os.path.exists)
#     removed = (~keep).sum()
#     if removed:
#         df = df[keep].reset_index(drop=True)
#         df.to_csv(p, index=False)
#         print(f"cleaned {name}: removed {removed} missing rows, kept {len(df)}")
#     else:
#         print(f"{name}: no missing rows")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check that train / val / test splits are disjoint by VIDEO, and summarize sources.
- Reads:  /home/wcheng31/sam2_classify/config/{train,val,test}_manifest_<suffix>.csv
- Uses:   /home/wcheng31/sam2_classify/config/label_map.json
- Console:
    • rows / images / clips / videos per split
    • ALL tasks (不截断，逐行打印)
    • per-class counts per split
    • OVERLAP report (should be empty)
- Exports CSV reports to: /home/wcheng31/sam2_classify/config/reports/
"""

import argparse, json, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

CFG_DIR_DEFAULT = Path("/home/wcheng31/sam2_classify/config")

# ---- patterns to extract a stable "video_id" (aligned with your CSV pipeline) ----
VID_PATTERNS = [
    re.compile(r"(cholec80_\d+)", re.IGNORECASE),
    re.compile(r"(case_\d+_video_part_\d+_segment_\d+)", re.IGNORECASE),
]

def extract_video_id(row: dict) -> str:
    task = str(row.get("task","")).lower()
    for pat in VID_PATTERNS:
        m = pat.search(task)
        if m: return m.group(1)
    clip = str(row.get("clip_name","")).lower()
    for pat in VID_PATTERNS:
        m = pat.search(clip)
        if m: return m.group(1)
    # fallback: images/<task>/...
    ip = str(row.get("image_path","")).lower()
    parts = [p for p in ip.split("/") if p]
    if "images" in parts:
        try:
            idx = parts.index("images")
            if idx + 1 < len(parts):
                return parts[idx+1]
        except Exception:
            pass
    return (task or clip or "unknown_video")

def load_label_map(label_map_path: Path) -> Dict[str, Dict[str, int]]:
    with open(label_map_path, "r", encoding="utf-8") as f:
        return json.load(f)

def per_class_counts(df: pd.DataFrame, tool2id: Dict[str,int]) -> Dict[str,int]:
    if df is None or len(df)==0: return {}
    m = {str(k): int(v) for k,v in tool2id.items()}
    ids = [m[t] for t in df["tool"].astype(str).tolist() if t in m]
    if not ids: return {}
    arr = np.bincount(np.array(ids), minlength=max(m.values())+1)
    id2name = {v:k for k,v in m.items()}
    return {id2name[i]: int(arr[i]) for i in range(len(arr)) if i in id2name}

def summarize_split(df: pd.DataFrame, name: str):
    if df is None or len(df)==0:
        print(f"[{name}] EMPTY")
        return {"rows":0,"images":0,"clips":0,"videos":0,"by_task":{}}
    imgs = df["image_path"].nunique()
    clips = df["clip_name"].astype(str).nunique()
    vids = df["video_id"].astype(str).nunique()
    by_task = df.groupby("task").size().sort_values(ascending=False).to_dict()
    print(f"[{name}] rows={len(df):,} | images={imgs:,} | clips={clips:,} | videos={vids:,}")
    # print ALL tasks, aligned
    if by_task:
        print("  tasks (count desc):")
        width = max(10, min(60, max(len(str(t)) for t in by_task.keys())))
        for i, (t, n) in enumerate(sorted(by_task.items(), key=lambda kv: kv[1], reverse=True), 1):
            t_str = str(t)
            if len(t_str) > width:
                t_str = t_str[:width-1] + "…"
            print(f"    {i:>3d}. {t_str:<{width}s}  {n:>7d}")
    return {"rows":len(df),"images":imgs,"clips":clips,"videos":vids,"by_task":by_task}

def write_reports(cfg_dir: Path, df: pd.DataFrame, split: str):
    rep_dir = cfg_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    # per-video summary
    g = df.groupby(["video_id","task"], dropna=False)
    rows = []
    for (vid, task), sub in g:
        rows.append({
            "video_id": vid,
            "task": task,
            "rows": int(len(sub)),
            "images": int(sub["image_path"].nunique()),
            "clips": int(sub["clip_name"].astype(str).nunique()),
            "classes_present": ",".join(sorted(sub["tool"].astype(str).unique().tolist()))
        })
    out = pd.DataFrame(rows).sort_values(["task","video_id"]).reset_index(drop=True)
    out.to_csv(rep_dir / f"{split}_videos.csv", index=False)
    print(f"  -> wrote {rep_dir / f'{split}_videos.csv'} ({len(out)} rows)")

def main():
    ap = argparse.ArgumentParser("Check train/val/test video disjointness and summarize dataset sources.")
    ap.add_argument("--config-dir", type=str, default=str(CFG_DIR_DEFAULT))
    ap.add_argument("--suffix", type=str, default="10", help="matches *_manifest_<suffix>.csv")
    ap.add_argument("--allow-overlap", action="store_true", help="do not exit non-zero even if overlaps exist")
    args = ap.parse_args()

    cfg = Path(args.config_dir)
    suffix = args.suffix

    paths = {
        "train": cfg / f"train_manifest_{suffix}.csv",
        "val":   cfg / f"val_manifest_{suffix}.csv",
        "test":  cfg / f"test_manifest_{suffix}.csv",
    }
    for k,p in paths.items():
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(2)

    # load label map for class stats (optional)
    lm_path = cfg / "label_map.json"
    tool2id = {}
    if lm_path.exists():
        lm = load_label_map(lm_path)
        tool2id = lm.get("tool_to_id", {})
    else:
        print("[WARN] label_map.json not found; class stats will be skipped.")

    dfs = {}
    for split, p in paths.items():
        df = pd.read_csv(p)
        if len(df) == 0:
            dfs[split] = df
            continue
        df["video_id"] = [extract_video_id(r) for _, r in df.iterrows()]
        dfs[split] = df

    # summaries
    print("========== SPLIT SUMMARY ==========")
    _ = {k: summarize_split(v, k) for k,v in dfs.items()}

    # class stats
    if tool2id:
        print("\n========== PER-CLASS COUNTS ==========")
        for k, df in dfs.items():
            cnt = per_class_counts(df, tool2id)
            print(f"[{k}] {cnt}")

    # disjointness checks (by video_id and by clip_name for extra safety)
    print("\n========== DISJOINTNESS CHECK ==========")
    vids = {k: set(df["video_id"].astype(str).unique().tolist()) for k,df in dfs.items()}
    clps = {k: set(df["clip_name"].astype(str).unique().tolist()) for k,df in dfs.items()}

    def inter(a,b): return sorted(list(a & b))
    ov_t_v  = inter(vids["train"], vids["val"])
    ov_t_te = inter(vids["train"], vids["test"])
    ov_v_te = inter(vids["val"],   vids["test"])
    any_overlap = any([ov_t_v, ov_t_te, ov_v_te])

    print("video_id overlaps:")
    print(f"  train ∩ val : {len(ov_t_v)}"  + (f" -> {ov_t_v[:10]}..."  if ov_t_v  else ""))
    print(f"  train ∩ test: {len(ov_t_te)}" + (f" -> {ov_t_te[:10]}..." if ov_t_te else ""))
    print(f"  val   ∩ test: {len(ov_v_te)}" + (f" -> {ov_v_te[:10]}..." if ov_v_te else ""))

    ov_c_t_v  = inter(clps["train"], clps["val"])
    ov_c_t_te = inter(clps["train"], clps["test"])
    ov_c_v_te = inter(clps["val"],   clps["test"])
    any_clip_overlap = any([ov_c_t_v, ov_c_t_te, ov_c_v_te])

    print("\nclip_name overlaps (extra guard):")
    print(f"  train ∩ val : {len(ov_c_t_v)}"  + (f" -> {ov_c_t_v[:10]}..."  if ov_c_t_v  else ""))
    print(f"  train ∩ test: {len(ov_c_t_te)}" + (f" -> {ov_c_t_te[:10]}..." if ov_c_t_te else ""))
    print(f"  val   ∩ test: {len(ov_c_v_te)}" + (f" -> {ov_c_v_te[:10]}..." if ov_c_v_te else ""))

    # write per-split video reports
    print("\n========== WRITE REPORTS ==========")
    for k, df in dfs.items():
        write_reports(cfg, df, k)

    # combined "which split each video goes to"
    rep_dir = cfg / "reports"
    rows = []
    for k, df in dfs.items():
        g = df.groupby(["video_id","task"], dropna=False)
        for (vid, task), sub in g:
            rows.append({
                "split": k,
                "video_id": vid,
                "task": task,
                "rows": int(len(sub)),
                "images": int(sub["image_path"].nunique()),
                "clips": int(sub["clip_name"].astype(str).nunique())
            })
    all_map = pd.DataFrame(rows).sort_values(["video_id","split"])
    all_map.to_csv(rep_dir / "video_to_split_map.csv", index=False)
    print(f"  -> wrote {rep_dir / 'video_to_split_map.csv'} ({len(all_map)} rows)")

    if any_overlap or any_clip_overlap:
        print("\n[FAIL] Splits are NOT disjoint by video or clip. Please fix the overlaps above.")
        if not args.allow_overlap:
            sys.exit(1)
    else:
        print("\n[OK] Splits are disjoint by video and by clip.")

if __name__ == "__main__":
    main()

# python /home/wcheng31/sam2_classify/test_dataset.py \
#   --config-dir /home/wcheng31/sam2_classify/config \
#   --suffix 10
