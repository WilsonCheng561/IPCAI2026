#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 5 类 OOD test 清单（不复制图像，直接引用原路径）：
- 输入根：/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data
  每个视频结构：videoXX/ws_0/images/*.jpg, videoXX/ws_0/prompts.json
- 默认视频列表：video01~video10（可在 VIDEO_LIST_STR 字符串里直接改）
- 输出：<out_root>/manifest.csv, <out_root>/label_map.json, <out_root>/stats.json

5 类映射：
  background：{1(Gallbladder),5(Bipolar),9(Irrigator),10(SpecimenBag), 以及未覆盖到的}
  grasper：{2,3,4}
  hook：{6}
  scissors：{7}
  clipper：{8}
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# ===== 你可以直接改这行来快速指定要处理的视频 =====
VIDEO_LIST_STR = "video01,video02,video03,video04,video05,video06,video07,video08,video09,video10"

# 源数据默认根目录
DEF_ROOT = "/projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data"
# 输出默认目录
DEF_OUT  = "/projects/surgical-video-digital-twin/ood_test_5cls"


OBJID_TO_NAME = {
    1: "Gallbladder",
    2: "Left Grasper",
    3: "TOP Grasper",
    4: "Right Grasper",
    5: "Bipolar",
    6: "Hook",
    7: "Scissors",
    8: "Clipper",
    9: "Irrigator",
    10: "SpecimenBag",
}

# 5类映射（其余全部 background）
def map_objid_to_5cls(obj_id: int) -> str:
    if obj_id in (2, 3, 4):
        return "grasper"
    if obj_id == 6:
        return "hook"
    if obj_id == 7:
        return "scissors"
    if obj_id == 8:
        return "clipper"
    return "background"

LABEL_MAP_5CLS = {
    "tool_to_id": {
        "background": 0,
        "grasper": 1,
        "hook": 2,
        "scissors": 3,
        "clipper": 4,
    }
}

def parse_video_list(s: str) -> List[str]:
    """
    支持逗号分隔，如 "video01,video02,video10"
    也支持简单的区间写法，比如 "video01-video10"
    """
    s = (s or "").strip()
    if not s:
        return []
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a = a.strip(); b = b.strip()
        if a.startswith("video") and b.startswith("video"):
            ai = int(a.replace("video", ""))
            bi = int(b.replace("video", ""))
            if ai <= bi:
                return [f"video{v:02d}" for v in range(ai, bi + 1)]
    # 默认逗号分隔
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

def load_prompts_json(p: Path) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 允许是 list 或 dict 包一层
    if isinstance(data, dict) and "frames" in data:
        return data["frames"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported prompts.json format: {p}")

def main():
    ap = argparse.ArgumentParser("Build 5-class OOD test manifest from prompts.json")
    ap.add_argument("--root", type=str, default=DEF_ROOT, help="annotated_data 根目录")
    ap.add_argument("--videos", type=str, default=VIDEO_LIST_STR, help="视频列表字符串，例如 'video01,video02' 或 'video01-video10'")
    ap.add_argument("--out-root", type=str, default=DEF_OUT, help="输出目录（保存 manifest/label_map/stats）")
    ap.add_argument("--keep-empty", action="store_true",
                    help="如果对象没有正点（labels==1），是否保留并使用所有点（极少需求）。默认跳过无正点对象。")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = parse_video_list(args.videos)
    if not videos:
        raise SystemExit("Empty video list. 请在 --videos 或脚本顶部 VIDEO_LIST_STR 指定视频。")

    rows = []
    stats = {
        "videos": videos,
        "per_class": {k: 0 for k in LABEL_MAP_5CLS["tool_to_id"].keys()},
        "objid_raw_count": {str(k): 0 for k in OBJID_TO_NAME.keys()},
        "skipped_no_positive_points": 0,
        "total_objects": 0,
        "total_samples": 0,
    }

    for vid in videos:
        base = root / vid / "ws_0"
        img_dir = base / "images"
        prom_p  = base / "prompts.json"
        if not prom_p.exists():
            print(f"[WARN] {vid} 没有 prompts.json，跳过")
            continue
        if not img_dir.is_dir():
            print(f"[WARN] {vid} 没有 images/ 目录，跳过")
            continue

        frames = load_prompts_json(prom_p)  # list of {frame_id, frame_file, objects:[{obj_id, points, labels}, ...]}

        for frm in frames:
            frame_file = frm.get("frame_file")
            if not frame_file:
                # 也许只有 frame_id，则按 0000000.jpg 这种规则自己格式化？这里严格要求存在 frame_file。
                continue
            img_path = img_dir / frame_file
            if not img_path.exists():
                # 忽略缺图的帧
                continue

            objects = frm.get("objects", []) or []
            for obj in objects:
                obj_id = int(obj.get("obj_id", -1))
                stats["total_objects"] += 1
                stats["objid_raw_count"][str(obj_id)] = stats["objid_raw_count"].get(str(obj_id), 0) + 1

                mapped = map_objid_to_5cls(obj_id)
                mapped_id = LABEL_MAP_5CLS["tool_to_id"][mapped]
                pts = obj.get("points", []) or []
                labs = obj.get("labels", []) or []

                # 对齐长度（容错）
                n = min(len(pts), len(labs)) if labs else len(pts)
                pos_points = []
                if labs:
                    for i in range(n):
                        if labs[i] == 1:
                            # 只保留正点
                            x, y = pts[i][:2]
                            pos_points.append([float(x), float(y), 1.0])
                else:
                    # 没有 labels 字段 -> 默认都当正点（少见）
                    for i in range(n):
                        x, y = pts[i][:2]
                        pos_points.append([float(x), float(y), 1.0])

                if len(pos_points) == 0 and not args.keep_empty:
                    stats["skipped_no_positive_points"] += 1
                    continue

                # 若 keep_empty，则把所有点当 1 以保留该对象（可选）
                if len(pos_points) == 0 and args.keep_empty:
                    for i in range(n):
                        x, y = pts[i][:2]
                        pos_points.append([float(x), float(y), 1.0])

                row = {
                    "video": vid,
                    "frame_id": int(frm.get("frame_id", -1)),
                    "frame_file": frame_file,
                    "image_path": str(img_path),
                    "obj_id": obj_id,
                    "obj_name": OBJID_TO_NAME.get(obj_id, f"obj{obj_id}"),
                    "mapped_class": mapped,
                    "class_id": mapped_id,
                    "num_points": len(pos_points),
                    "points_json": json.dumps(pos_points, ensure_ascii=False),
                }
                rows.append(row)
                stats["per_class"][mapped] += 1

    df = pd.DataFrame(rows)
    out_mani = out_root / "manifest.csv"
    out_lbl  = out_root / "label_map.json"
    out_stat = out_root / "stats.json"

    if len(df) == 0:
        print("[WARN] 没有生成任何样本（可能是所有对象都没有 labels==1 的点）")
    else:
        # 按视频、帧排序更直观
        df.sort_values(by=["video", "frame_id", "obj_id"], inplace=True)

    df.to_csv(out_mani, index=False)
    with open(out_lbl, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP_5CLS, f, ensure_ascii=False, indent=2)
    with open(out_stat, "w", encoding="utf-8") as f:
        stats["total_samples"] = int(len(df))
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Done.\n- manifest: {out_mani}\n- label_map: {out_lbl}\n- stats: {out_stat}\nSamples: {len(df)}")

if __name__ == "__main__":
    main()

# 用的是cholec80 video41-50
# python build_ood_5cls_manifest.py \
#   --root /projects/surgical-video-digital-twin/datasets/cholec80_raw/annotated_data \
#   --videos "video41-video50" \
#   --out-root /projects/surgical-video-digital-twin/ood_test_5cls
