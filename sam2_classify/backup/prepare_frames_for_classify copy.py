#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare frames strictly at annotated frameIndices (NOT fixed stride),
and control how 'contact' points are used for SAM2 prompts.

Key behaviors
增量修改: 防止train，val，test的数据泄露（在test.py测试），随机sample点+contact作为bg
-------------
1) Sample frames ONLY at frame indices present in annotation "points[].frameIndex".
2) Group points by absolute frame, then by tool. A single frame may yield multiple
   training rows (one per tool), each keeping ALL its positive points.
3) 'tool_tip' and 'tool_anchor' are treated as POSITIVE by default.
4) 'contact' handling is configurable:
     --contact-policy ignore   -> drop contact points (default)
     --contact-policy negative -> keep contact as NEGATIVE (label=0) *only if*
                                  the point's 'tool' matches the row's tool.
                                  (Contact without a 'tool' field is ignored to avoid ambiguity.)

Inputs
------
index.json: dict(uuid -> entry), where each entry has at least:
  {
    "task": "cholec80_01",test_manifest.csv
    "start": 39288,                  # absolute start frame for the clip
    "end": 39780,
    "tool": "clipper",               # top-level tool (fallback)
    "action": "clip",
    "clip_dir": ".../clips/clip_39288_39780.mp4",
    "anno_dir": ".../annotation/clip_39288_39780.json"
  }

Annotation JSON example (per-clip):
{
  "tool": "clipper",
  "affordance_range": {"start": 39288, "end": 39780},
  "action_range": {"start": 39583, "end": 39738},
  "points": [
     {"x":620.1, "y":370.1, "type":"contact",    "frameIndex":39288, "tool":"clipper", ...},
     {"x":688.1, "y":362.7, "type":"tool_anchor","frameIndex":39288, "tool":"clipper", ...},
     {"x":631.1, "y":322.8, "type":"tool_tip",   "frameIndex":39288, "tool":"clipper", ...},
     ...
  ]
}

Outputs
-------
<out_root>/
  images/<task>/<clip_name>/frame_<absidx>.jpg
  prompts/<task>/<clip_name>/<tool>/frame_<absidx>.json   # if --write-prompts (points with labels 1/0)
  manifest.csv                                            # one row per (frame_abs_index, tool)
  label_map.json

Manifest columns
----------------
uid, task, clip_name, tool, action,
start_abs, end_abs, frame_abs_index, frame_idx_in_clip,
image_path, anno_path, num_points_pos, num_points_neg, points_json

Where points_json is a JSON-serialized list of [x, y, label] with label in {1 (pos), 0 (neg)}.
"""
import argparse, json, warnings, csv, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2

# ---------------- global paths ----------------
SMALLFILE_ROOT = Path("/home/wcheng31/sam2_classify/config")   # manifest.csv, label_map.json
DATASET_ROOT   = Path("/projects/surgical-video-digital-twin/datasets/sam2_classifier")  # processed images/prompts
MP4_ROOT       = Path("/projects/surgical-video-digital-twin/datasets/toy_export")       # mp4 dataset
PRETRAIN_ROOT  = Path("/projects/surgical-video-digital-twin/pretrain_params")           # pretrained weights
CKPT_ROOT      = PRETRAIN_ROOT / "cwz" / "sam2_classifier"                              # checkpoints

# ---------------- I/O helpers ----------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to read JSON {p}: {e}")
        return None

# ---------------- core logic ----------------
POS_DEFAULT = {"tool_tip", "tool_anchor", "tip", "anchor"}  # include common synonyms
CONTACT_NAMES = {"contact"}
BACKGROUND_NAME = "background"  # 新增：把 contact 当作背景类，写到 manifest 里作为一个“工具名”

def canonical_type(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")

def in_abs_range(abs_idx: int, abs_lo: Optional[int], abs_hi: Optional[int]) -> bool:
    if abs_lo is None or abs_hi is None:
        return True
    return abs_lo <= abs_idx <= abs_hi

def gather_per_frame_per_tool_points(
    anno: Dict[str, Any],
    start_abs: int,
    range_mode: str,
    positive_types: Set[str],
    contact_policy: str
) -> Dict[int, Dict[str, Dict[str, List[List[float]]]]]:
    """
    返回：
      frames[abs_idx][tool_name] = {"pos": [[x,y,1.0], ...], "neg": []}
    其中：
      - tool_name 为具体工具名；另外我们会把 contact 点统一写入 BACKGROUND_NAME 这一类。
      - 仅使用 vis==True 的点；vis==False 直接丢弃。
    """
    frames: Dict[int, Dict[str, Dict[str, List[List[float]]]]] = {}
    if not anno:
        return frames

    abs_lo, abs_hi = None, None
    if range_mode == "affordance":
        r = anno.get("affordance_range")
        if r: abs_lo, abs_hi = int(r["start"]), int(r["end"])
    elif range_mode == "action":
        r = anno.get("action_range")
        if r: abs_lo, abs_hi = int(r["start"]), int(r["end"])

    top_tool = str(anno.get("tool", "unknown_tool"))

    for d in anno.get("points", []):
        try:
            # 只保留可见点
            if d.get("vis", True) is not True:
                continue

            abs_idx = int(d["frameIndex"])
            if not in_abs_range(abs_idx, abs_lo, abs_hi):
                continue
            t = canonical_type(d.get("type", ""))
            x, y = float(d["x"]), float(d["y"])

            if t in positive_types:
                tool_name = str(d.get("tool", top_tool))
                per_tool = frames.setdefault(abs_idx, {}).setdefault(tool_name, {"pos": [], "neg": []})
                per_tool["pos"].append([x, y, 1.0])
                continue

            # contact => 作为“背景类”正样本（不再作为各工具的 neg）
            if t in CONTACT_NAMES:
                per_bg = frames.setdefault(abs_idx, {}).setdefault(BACKGROUND_NAME, {"pos": [], "neg": []})
                per_bg["pos"].append([x, y, 1.0])
                continue

            # 其他类型忽略
        except Exception as e:
            warnings.warn(f"Bad point entry skipped: {d} ({e})")
            continue
    return frames

def save_frames_for_abs_indices(
    mp4_path: Path,
    out_dir: Path,
    abs_to_local: Dict[int, int],
    resize: Optional[int],
    jpg_q: int
) -> Dict[int, Path]:
    needed_local = set(abs_to_local.values())
    if not needed_local:
        return {}
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    saved: Dict[int, Path] = {}
    idx = 0
    ok, frame = cap.read()
    while ok and needed_local:
        if idx in needed_local:
            if resize and resize > 0:
                frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
            for abs_i, loc in abs_to_local.items():
                if loc == idx:
                    outp = out_dir / f"frame_{abs_i:06d}.jpg"
                    cv2.imwrite(str(outp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_q)])
                    saved[abs_i] = outp
            needed_local.remove(idx)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    return saved

# ====== NEW: split by clip_name (non-overlapping) ======
def _split_by_clip_nonoverlap(
    rows: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    所有同一个 clip_name 的样本进入同一个 split，避免信息泄漏。
    """
    rng = random.Random(seed)
    clips = {}
    for r in rows:
        clips.setdefault(str(r["clip_name"]), []).append(r)
    clip_names = list(clips.keys())
    rng.shuffle(clip_names)

    n = len(clip_names)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val   = min(n_val, max(0, n - n_train))

    train_clips = set(clip_names[:n_train])
    val_clips   = set(clip_names[n_train:n_train+n_val])
    test_clips  = set(clip_names[n_train+n_val:])

    train, val, test = [], [], []
    for c in train_clips: train.extend(clips[c])
    for c in val_clips:   val.extend(clips[c])
    for c in test_clips:  test.extend(clips[c])
    return train, val, test

# ====== NEW: train-only 背景增强（每个训练帧随机加 2 个背景点；如无 background 行则创建一行） ======
def _augment_background_for_train(train_rows: List[Dict[str,Any]], add_points: int = 2, seed: int = 42):
    rng = random.Random(seed)
    # 建索引：同一帧的所有行
    by_image: Dict[str, List[Dict[str,Any]]] = {}
    for r in train_rows:
        by_image.setdefault(str(r["image_path"]), []).append(r)

    new_rows: List[Dict[str,Any]] = []
    for img_path, rows in by_image.items():
        # 查找/创建 background 行
        bg_row = None
        for r in rows:
            if str(r["tool"]) == BACKGROUND_NAME:
                bg_row = r
                break
        if bg_row is None:
            # 复制元数据来新建一行 background
            ref = rows[0]
            bg_row = {
                "uid": ref["uid"],
                "task": ref["task"],
                "clip_name": ref["clip_name"],
                "tool": BACKGROUND_NAME,
                "action": ref["action"],
                "start_abs": ref["start_abs"],
                "end_abs": ref["end_abs"],
                "frame_abs_index": ref["frame_abs_index"],
                "frame_idx_in_clip": ref["frame_idx_in_clip"],
                "image_path": ref["image_path"],
                "anno_path": ref["anno_path"],
                "num_points_pos": 0,
                "num_points_neg": 0,
                "points_json": "[]",
            }
            new_rows.append(bg_row)

        # 在图片尺寸内随机加点
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # 找不到图片就跳过增强（但仍然保留已有 contact 背景点）
            continue
        H, W = img.shape[:2]
        try:
            pts = json.loads(bg_row["points_json"]) if isinstance(bg_row["points_json"], str) and bg_row["points_json"].strip() else []
        except Exception:
            pts = []
        for _ in range(add_points):
            x = rng.uniform(0, max(1, W-1))
            y = rng.uniform(0, max(1, H-1))
            pts.append([float(x), float(y), 1.0])  # 背景类的“正点”
        bg_row["points_json"] = json.dumps(pts, ensure_ascii=False)
        bg_row["num_points_pos"] = len(pts)
        bg_row["num_points_neg"] = 0

    if new_rows:
        train_rows.extend(new_rows)

# ====== END NEW ======

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=False, default=str(MP4_ROOT / "index.json"),
                    help="Path to index.json")
    ap.add_argument("--out-root", required=False, default=str(DATASET_ROOT),
                    help="Output root (processed dataset)")
    ap.add_argument("--range", choices=["clip","affordance","action"], default="clip",
                    help="Temporal range: 'clip' uses all points; 'affordance' or 'action' restricts to those ranges if present.")
    ap.add_argument("--contact-policy", choices=["ignore","negative"], default="ignore",
                    help="How to treat 'contact' points. 'ignore' drops them; 'negative' keeps as label=0 for the same tool.")
    ap.add_argument("--positive-types", type=str, default="tool_tip,tool_anchor,tip,anchor",
                    help="Comma-separated types considered positive (default includes synonyms).")
    ap.add_argument("--resize", type=int, default=None, help="Resize saved frames to a square size (e.g., 640).")
    ap.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality (1..100).")
    ap.add_argument("--write-prompts", action="store_true", help="Write per-frame per-tool prompt JSON files.")
    # ====== NEW: csv-only switch ======
    ap.add_argument("--csv-only", action="store_true",
                    help="Skip decoding/writing frames; only build CSVs (assumes frames already exist at images/<task>/<clip_name>/frame_<absidx>.jpg).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting & bg augmentation")
    # ====== END NEW ======
    args = ap.parse_args()

    index_path = Path(args.index)
    data = read_json(index_path)
    if not isinstance(data, dict):
        print("index.json must be a dict mapping uuid -> entry")
        return

    out_root = Path(args.out_root)
    safe_mkdir(out_root)

    pos_types = {canonical_type(s) for s in args.positive_types.split(",") if s.strip()}

    manifest_rows: List[Dict[str, Any]] = []
    tool_set, action_set = set(), set()
    # 确保始终包含背景类
    tool_set.add(BACKGROUND_NAME)

    for uid, e in data.items():
        task = e.get("task", "unknown_task")
        top_tool = str(e.get("tool", "unknown_tool"))
        action = str(e.get("action", ""))
        start_abs = int(e.get("start", 0))
        end_abs = int(e.get("end", 0))
        mp4 = Path(e["clip_dir"])
        anno_p = Path(e.get("anno_dir")) if e.get("anno_dir") else None

        if action:
            action_set.add(action)

        clip_name = mp4.stem
        img_dir = out_root / "images" / task / clip_name
        prm_dir = out_root / "prompts" / task / clip_name
        safe_mkdir(img_dir)
        if args.write_prompts:
            safe_mkdir(prm_dir)

        anno = read_json(anno_p) if (anno_p and anno_p.exists()) else {"tool": top_tool, "points": []}

        frames = gather_per_frame_per_tool_points(
            anno=anno,
            start_abs=start_abs,
            range_mode=args.range,
            positive_types=pos_types,
            contact_policy=args.contact_policy
        )
        if not frames:
            continue

        abs_to_local = {abs_i: (abs_i - start_abs) for abs_i in frames.keys() if (abs_i - start_abs) >= 0}

        # decide image_path source by --csv-only
        if args.csv_only:
            saved = {abs_i: (img_dir / f"frame_{abs_i:06d}.jpg") for abs_i in abs_to_local.keys()}
        else:
            saved = save_frames_for_abs_indices(mp4, img_dir, abs_to_local, args.resize, args.jpg_quality)

        for abs_i, tool_dict in frames.items():
            local_idx = abs_to_local.get(abs_i, None)
            if local_idx is None:
                continue
            img_path = saved.get(abs_i, None)
            if img_path is None:
                continue

            for tool_name, groups in tool_dict.items():
                pos_pts = groups.get("pos", [])
                neg_pts = groups.get("neg", [])
                if not pos_pts and not neg_pts:
                    continue
                tool_set.add(tool_name)
                points_combined = pos_pts + neg_pts
                if args.write_prompts:
                    tool_dir = prm_dir / tool_name
                    safe_mkdir(tool_dir)
                    with open(tool_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "image_path": str(img_path),
                            "points": points_combined,
                            "tool": tool_name,
                            "action": action,
                            "frame_abs_index": abs_i,
                            "frame_idx_in_clip": local_idx
                        }, f, ensure_ascii=False, indent=2)

                manifest_rows.append({
                    "uid": uid,
                    "task": task,
                    "clip_name": clip_name,
                    "tool": tool_name,
                    "action": action,
                    "start_abs": start_abs,
                    "end_abs": end_abs,
                    "frame_abs_index": abs_i,
                    "frame_idx_in_clip": local_idx,
                    "image_path": str(img_path),
                    "anno_path": str(anno_p) if anno_p else "",
                    "num_points_pos": len(pos_pts),
                    "num_points_neg": len(neg_pts),
                    "points_json": json.dumps(points_combined, ensure_ascii=False),
                })

    if manifest_rows:
        safe_mkdir(SMALLFILE_ROOT)
        mf = SMALLFILE_ROOT / "manifest.csv"
        with open(mf, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            for r in manifest_rows:
                writer.writerow(r)

        # 写 label_map：background 固定 id=0，其余工具从 1 开始
        other_tools = sorted([t for t in tool_set if t != BACKGROUND_NAME])
        tool_to_id = {BACKGROUND_NAME: 0}
        tool_to_id.update({k: i+1 for i, k in enumerate(other_tools)})
        lm = {
            "tool_to_id": tool_to_id,
            "action_to_id": {k: i for i, k in enumerate(sorted(action_set))},
        }
        with open(SMALLFILE_ROOT / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(lm, f, ensure_ascii=False, indent=2)

        # ====== split by clip_name (non-overlap) ======
        train_rows, val_rows, test_rows = _split_by_clip_nonoverlap(
            manifest_rows, train_ratio=0.8, val_ratio=0.1, seed=args.seed
        )

        # ====== 仅对训练集做背景增强（每帧补足/增加 2 个背景点） ======
        _augment_background_for_train(train_rows, add_points=2, seed=args.seed)

        def _dump_csv(rows, path: Path):
            with open(path, "w", newline="", encoding="utf-8") as ff:
                w = csv.DictWriter(ff, fieldnames=list(manifest_rows[0].keys()))
                w.writeheader()
                for rr in rows:
                    w.writerow(rr)

        _dump_csv(train_rows, SMALLFILE_ROOT / "train_manifest.csv")
        _dump_csv(val_rows,   SMALLFILE_ROOT / "val_manifest.csv")
        _dump_csv(test_rows,  SMALLFILE_ROOT / "test_manifest.csv")
        print(f"Split (by clip_name, non-overlap) -> train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

        print(f"Done. Wrote frames/prompts to dataset root: {out_root}")
        print(f"- manifest: {mf}")
        print(f"- label_map.json: {SMALLFILE_ROOT /'label_map.json'}")
    else:
        print("No frames were produced. Check that your annotations contain points within chosen range.")

if __name__ == "__main__":
    main()

# python /home/wcheng31/sam2_classify/prepare_frames_for_classify.py \
#   --range action \
#   --contact-policy ignore \
#   --write-prompts \
#   --csv-only
