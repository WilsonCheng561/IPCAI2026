#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-stop dataset builder for SAM2 tool-class head pretraining.

必做（默认开启，无需额外参数）：
- 自动把 index.json 中的旧前缀 "/mnt/disk1/haoding/" 替换为
  "/projects/surgical-video-digital-twin/datasets/"；若仍不存在，按
  <new_root>/<task>/{clips|annotation}/<stem>.<ext> 进行重构再校验。

构建规则：
- 仅在 annotation 的 frameIndex 上抽帧；
- 一帧同一 tool 的 tip/anchor 全部合并为 1 个 tool object；
- 背景：每帧最多 3 个 BG object = 1×contact 汇总 + ≤2×随机单点（各自单独一行，所有 split 都生成）；
- 覆盖式输出（默认清空输出根，可 --no-clean 关闭）；
- clip 级别非重叠 train/val/test；
- 可选 --write-prompts / --csv-only / --resize。

示例：
python prepare_frames_for_classify.py --verify-only
python prepare_frames_for_classify.py --write-prompts
python prepare_frames_for_classify.py --csv-only --write-prompts
"""

import argparse, json, warnings, csv, random, shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2

# ======= 固定路径（按你的环境） =======
MP4_ROOT       = Path("/projects/surgical-video-digital-twin/datasets/surg_act_09232025")
DATASET_ROOT   = Path("/projects/surgical-video-digital-twin/datasets/sam2_classifier")
SMALLFILE_ROOT = DATASET_ROOT / "config"

# ======= 常量 =======
POS_DEFAULT = {"tool_tip", "tool_anchor", "tip", "anchor"}
CONTACT_NAMES = {"contact"}
BACKGROUND_NAME = "background"

OLD_PREFIX = "/mnt/disk1/haoding/"
NEW_PREFIX = "/projects/surgical-video-digital-twin/datasets/"

# ======= 工具 =======
def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to read JSON {p}: {e}")
        return None

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def canonical_type(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")

def in_abs_range(abs_idx: int, abs_lo: Optional[int], abs_hi: Optional[int]) -> bool:
    if abs_lo is None or abs_hi is None: return True
    return abs_lo <= abs_idx <= abs_hi

# ======= A) 统一替换 + 重构路径（默认总会执行） =======
def normalize_entry_paths(uid: str, e: Dict[str, Any], new_root: Path) -> None:
    """无条件把旧前缀替换为新前缀；若替换后仍不存在，按 <new_root>/<task>/<sub>/<stem>.<ext> 重构。"""
    task = e.get("task", "")
    clip_dir = str(e.get("clip_dir", ""))
    anno_dir = str(e.get("anno_dir", ""))

    def replace_prefix(p: str) -> str:
        if p.startswith(OLD_PREFIX):
            return p.replace(OLD_PREFIX, NEW_PREFIX, 1)
        return p

    c2 = replace_prefix(clip_dir)
    a2 = replace_prefix(anno_dir)

    # 若仍不存在，重构
    def rebuild(path_old: str, is_clip: bool) -> str:
        stem = Path(path_old).stem if path_old else ""
        sub = "clips" if is_clip else "annotation"
        ext = ".mp4" if is_clip else ".json"
        if task:
            return str(new_root / task / sub / f"{stem}{ext}")
        return str(new_root / sub / f"{stem}{ext}")

    if not Path(c2).exists():
        c2 = rebuild(clip_dir, True)
    if not Path(a2).exists():
        a2 = rebuild(anno_dir, False)

    e["clip_dir"] = c2
    e["anno_dir"] = a2

# ======= C.1) 聚合：同帧/同工具/背景 =======
def gather_per_frame_per_tool_points(anno: Dict[str, Any], start_abs: int, range_mode: str, positive_types: Set[str]
) -> Dict[int, Dict[str, Dict[str, List[List[float]]]]]:
    frames: Dict[int, Dict[str, Dict[str, List[List[float]]]]] = {}
    if not anno: return frames
    abs_lo = abs_hi = None
    if range_mode == "affordance":
        r = anno.get("affordance_range");  abs_lo, abs_hi = (int(r["start"]), int(r["end"])) if r else (None, None)
    elif range_mode == "action":
        r = anno.get("action_range");      abs_lo, abs_hi = (int(r["start"]), int(r["end"])) if r else (None, None)

    top_tool = str(anno.get("tool", "unknown_tool"))
    for d in anno.get("points", []):
        if not d or d.get("vis", True) is not True: continue
        try:
            abs_idx = int(d["frameIndex"])
        except Exception:
            continue
        if not in_abs_range(abs_idx, abs_lo, abs_hi): continue
        t = canonical_type(d.get("type", ""))
        x, y = float(d.get("x", 0.0)), float(d.get("y", 0.0))
        if t in positive_types:
            tool_name = str(d.get("tool", top_tool))
            frames.setdefault(abs_idx, {}).setdefault(tool_name, {"pos": []})["pos"].append([x, y, 1.0])
        elif t in CONTACT_NAMES:
            frames.setdefault(abs_idx, {}).setdefault(BACKGROUND_NAME, {"pos": []})["pos"].append([x, y, 1.0])
    return frames

# ======= C.2) 仅保存需要的帧 =======
def save_frames_for_abs_indices(mp4_path: Path, out_dir: Path, abs_to_local: Dict[int, int],
                                resize: Optional[int], jpg_q: int) -> Dict[int, Path]:
    needed_local = set(abs_to_local.values())
    if not needed_local: return {}
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {mp4_path}")
    saved: Dict[int, Path] = {}
    idx = 0; ok, frame = cap.read()
    while ok and needed_local:
        if idx in needed_local:
            fr = frame
            if resize and resize > 0: fr = cv2.resize(fr, (resize, resize), interpolation=cv2.INTER_AREA)
            for abs_i, loc in abs_to_local.items():
                if loc == idx:
                    outp = out_dir / f"frame_{abs_i:06d}.jpg"
                    cv2.imwrite(str(outp), fr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_q)])
                    saved[abs_i] = outp
            needed_local.remove(idx)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    return saved

# ======= C.3) clip 级别拆分 =======
def split_by_clip_nonoverlap(rows: List[Dict[str, Any]], train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    buckets = {}
    for r in rows: buckets.setdefault(str(r["clip_name"]), []).append(r)
    keys = list(buckets.keys()); rng.shuffle(keys)
    n = len(keys); n_tr = min(int(round(n * train_ratio)), n); n_va = min(int(round(n * val_ratio)), max(0, n-n_tr))
    tr, va, te = set(keys[:n_tr]), set(keys[n_tr:n_tr+n_va]), set(keys[n_tr+n_va:])
    rows_tr = [r for r in rows if r["clip_name"] in tr]
    rows_va = [r for r in rows if r["clip_name"] in va]
    rows_te = [r for r in rows if r["clip_name"] in te]
    return rows_tr, rows_va, rows_te

# ======= C.4) 背景随机点（每个 object 仅 1 点；最多 2 个）=======
def make_random_bg_rows(ref_row: Dict[str, Any], img_path: Path, how_many: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed + hash(str(img_path)) % 997)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None: return []
    H, W = img.shape[:2]
    how_many = max(0, min(2, int(how_many)))
    rows = []
    for _ in range(how_many):
        x = rng.uniform(0, max(1, W-1)); y = rng.uniform(0, max(1, H-1))
        r = dict(ref_row)
        r["tool"] = BACKGROUND_NAME
        r["num_points_pos"] = 1
        r["points_json"] = json.dumps([[float(x), float(y), 1.0]], ensure_ascii=False)
        rows.append(r)
    return rows

# ======= 主流程 =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default=str(MP4_ROOT / "index.json"))
    ap.add_argument("--verify-only", action="store_true", help="只做结构校验并退出")
    ap.add_argument("--force", action="store_true", help="校验失败也继续执行（不推荐）")
    ap.add_argument("--clean", action="store_true", help="开始前清空输出目录（images/prompts/config）")
    ap.add_argument("--no-clean", dest="clean", action="store_false")
    ap.set_defaults(clean=True)
    ap.add_argument("--out-root", type=str, default=str(DATASET_ROOT))
    ap.add_argument("--range", choices=["clip","affordance","action"], default="clip")
    ap.add_argument("--positive-types", type=str, default="tool_tip,tool_anchor,tip,anchor")
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--write-prompts", action="store_true")
    ap.add_argument("--csv-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    index_path = Path(args.index)
    data = read_json(index_path)
    if not isinstance(data, dict):
        raise SystemExit("index.json must be a dict mapping uuid -> entry")

    # A) 强制归一化路径（替换旧前缀 + 重构兜底）
    for uid, e in data.items():
        normalize_entry_paths(uid, e, new_root=MP4_ROOT)

    # B) 结构校验
    missing_report = []
    for uid, e in data.items():
        clip_dir = Path(e.get("clip_dir","")); anno_dir = Path(e.get("anno_dir",""))
        if not clip_dir.exists(): missing_report.append(f"[{uid}] clip_dir missing: {clip_dir}")
        if not anno_dir.exists(): missing_report.append(f"[{uid}] anno_dir missing: {anno_dir}")
        if clip_dir.parent.name != "clips": missing_report.append(f"[{uid}] clip_dir parent not 'clips': {clip_dir}")
        if anno_dir.parent.name != "annotation": missing_report.append(f"[{uid}] anno_dir parent not 'annotation': {anno_dir}")
        if clip_dir.stem != anno_dir.stem: missing_report.append(f"[{uid}] stem mismatch: {clip_dir.name} vs {anno_dir.name}")

    if missing_report:
        print("✗ 结构校验失败：")
        for s in missing_report[:80]: print("  -", s)
        if not args.force:
            return
        print("⚠ 使用 --force 继续。")
    else:
        print("✓ 结构校验通过。")

    if args.verify_only:
        return

    # C) 构建数据
    out_root = Path(args.out_root)
    if args.clean and out_root.exists():
        print(f"清理旧输出：{out_root}")
        for sub in ["images", "prompts", "config"]:
            p = out_root / sub if sub != "config" else SMALLFILE_ROOT
            if p.exists(): shutil.rmtree(p, ignore_errors=True)
    safe_mkdir(out_root); safe_mkdir(SMALLFILE_ROOT)

    pos_types = {canonical_type(s) for s in args.positive_types.split(",") if s.strip()}
    manifest_rows: List[Dict[str, Any]] = []
    action_set = set()

    for uid, e in data.items():
        task = e.get("task", "unknown_task")
        top_tool = str(e.get("tool", "unknown_tool"))
        action = str(e.get("action", ""))
        if action:
            action_set.add(action)

        start_abs = int(e.get("start", 0)); end_abs = int(e.get("end", 0))
        mp4 = Path(e["clip_dir"]);  anno_p = Path(e.get("anno_dir")) if e.get("anno_dir") else None

        clip_name = mp4.stem
        img_dir = out_root / "images" / task / clip_name
        prm_dir = out_root / "prompts" / task / clip_name
        safe_mkdir(img_dir)
        if args.write_prompts:
            safe_mkdir(prm_dir)

        anno = read_json(anno_p) if (anno_p and anno_p.exists()) else {"tool": top_tool, "points": []}
        frames = gather_per_frame_per_tool_points(anno, start_abs, args.range, pos_types)
        if not frames:
            continue

        abs_to_local = {abs_i: (abs_i - start_abs) for abs_i in frames.keys() if (abs_i - start_abs) >= 0}

        if args.csv_only:
            saved = {abs_i: (img_dir / f"frame_{abs_i:06d}.jpg") for abs_i in abs_to_local.keys()}
        else:
            saved = save_frames_for_abs_indices(mp4, img_dir, abs_to_local, args.resize, args.jpg_quality)

        for abs_i, tool_dict in frames.items():
            local_idx = abs_to_local.get(abs_i)
            img_path = saved.get(abs_i)
            if local_idx is None or img_path is None:
                continue

            # 工具行：同帧同 tool 合并一个 object
            for tool_name, groups in tool_dict.items():
                if tool_name == BACKGROUND_NAME:
                    continue
                pos_pts = groups.get("pos", [])
                if not pos_pts:
                    continue
                row = {
                    "uid": uid, "task": task, "clip_name": clip_name, "tool": tool_name, "action": action,
                    "start_abs": start_abs, "end_abs": end_abs,
                    "frame_abs_index": abs_i, "frame_idx_in_clip": local_idx,
                    "image_path": str(img_path), "anno_path": str(anno_p) if anno_p else "",
                    "num_points_pos": len(pos_pts), "num_points_neg": 0,
                    "points_json": json.dumps(pos_pts, ensure_ascii=False),
                }
                manifest_rows.append(row)
                if args.write_prompts:
                    tp = prm_dir / tool_name; safe_mkdir(tp)
                    write_json(tp / f"{img_path.stem}.json",
                               {"image_path": str(img_path), "points": pos_pts, "tool": tool_name,
                                "action": action, "frame_abs_index": abs_i, "frame_idx_in_clip": local_idx})

            # 背景：contact 汇总（一个 object）
            bg_pts = tool_dict.get(BACKGROUND_NAME, {}).get("pos", [])
            if bg_pts:
                row_bg = {
                    "uid": uid, "task": task, "clip_name": clip_name, "tool": BACKGROUND_NAME, "action": action,
                    "start_abs": start_abs, "end_abs": end_abs,
                    "frame_abs_index": abs_i, "frame_idx_in_clip": local_idx,
                    "image_path": str(img_path), "anno_path": str(anno_p) if anno_p else "",
                    "num_points_pos": len(bg_pts), "num_points_neg": 0,
                    "points_json": json.dumps(bg_pts, ensure_ascii=False),
                }
                manifest_rows.append(row_bg)
                if args.write_prompts:
                    tp = prm_dir / BACKGROUND_NAME; safe_mkdir(tp)
                    write_json(tp / f"{img_path.stem}_contact.json",
                               {"image_path": str(img_path), "points": bg_pts, "tool": BACKGROUND_NAME,
                                "action": action, "frame_abs_index": abs_i, "frame_idx_in_clip": local_idx,
                                "kind": "contact_merged"})

            # 背景：随机 ≤2 个 object（每个 1 点）
            ref = {
                "uid": uid, "task": task, "clip_name": clip_name, "tool": BACKGROUND_NAME, "action": action,
                "start_abs": start_abs, "end_abs": end_abs,
                "frame_abs_index": abs_i, "frame_idx_in_clip": local_idx,
                "image_path": str(img_path), "anno_path": str(anno_p) if anno_p else "",
                "num_points_pos": 1, "num_points_neg": 0, "points_json": "[]",
            }
            rand_rows = make_random_bg_rows(ref, img_path, how_many=2, seed=args.seed)
            manifest_rows.extend(rand_rows)
            if args.write_prompts and rand_rows:
                tp = prm_dir / BACKGROUND_NAME; safe_mkdir(tp)
                for k, rr in enumerate(rand_rows):
                    write_json(tp / f"{img_path.stem}_rand{k+1}.json",
                               {"image_path": rr["image_path"], "points": json.loads(rr["points_json"]),
                                "tool": BACKGROUND_NAME, "action": action,
                                "frame_abs_index": abs_i, "frame_idx_in_clip": local_idx, "kind": "random_bg"})

    if not manifest_rows:
        print("No rows produced.");  return

    # label_map：background 固定 0，其余工具从 1 开始
    other_tools = sorted({r["tool"] for r in manifest_rows if r["tool"] != BACKGROUND_NAME})
    tool_to_id = {BACKGROUND_NAME: 0}
    tool_to_id.update({k: i+1 for i, k in enumerate(other_tools)})

    # 写 CSV/映射与划分
    SMALLFILE_ROOT.mkdir(parents=True, exist_ok=True)
    def _dump(rows, p: Path):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)

    write_json(SMALLFILE_ROOT / "label_map.json",
               {"tool_to_id": tool_to_id, "action_to_id": {}})

    manifest_rows_sorted = sorted(manifest_rows, key=lambda r: (r["task"], r["clip_name"], r["frame_abs_index"], r["tool"]))
    _dump(manifest_rows_sorted, SMALLFILE_ROOT / "manifest.csv")

    tr, va, te = split_by_clip_nonoverlap(manifest_rows_sorted, train_ratio=0.8, val_ratio=0.1, seed=args.seed)
    _dump(tr, SMALLFILE_ROOT / "train_manifest.csv")
    _dump(va, SMALLFILE_ROOT / "val_manifest.csv")
    _dump(te, SMALLFILE_ROOT / "test_manifest.csv")

    print(f"Split(by clip) -> train={len(tr)} val={len(va)} test={len(te)}")
    print(f"Done.\n- images/prompts root: {DATASET_ROOT}\n- manifest: {SMALLFILE_ROOT/'manifest.csv'}\n- label_map: {SMALLFILE_ROOT/'label_map.json'}")

if __name__ == "__main__":
    main()



# # 只校验结构：
# python /home/wcheng31/sam2_classify/prepare_frames_for_classify.py --verify-only


# # 校验 + 覆盖式重建（默认清空旧输出）：
# python /home/wcheng31/sam2_classify/prepare_frames_for_classify.py --write-prompts

# # 修复一下，还需要move csv
# cp -f /projects/surgical-video-digital-twin/datasets/sam2_classifier/config/*.csv \
#        /home/wcheng31/sam2_classify/config/



# # 若你已生成过帧，只想重建 CSV（不重新解码 mp4）：
# python /home/wcheng31/sam2_classify/prepare_frames_for_classify.py --csv-only --write-prompts


# # 遇到个别坏条目也想继续（不推荐）：
# python /home/wcheng31/sam2_classify/prepare_frames_for_classify.py --force --write-prompts