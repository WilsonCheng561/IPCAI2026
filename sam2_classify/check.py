#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify SAM2-like mask_tokens format:
Hypothesis:
  - Each file is a (K, 257) float matrix (K≈10).
  - First 256 dims = embedding; last dim = score/logit.
  - If real tokens < K, remaining rows are zero-padded (all-zero embedding and score==0).
  - (Often) valid token rows are sorted by descending score.

Outputs:
  - Per-file summary (effective_K, padding pattern, score sort, stats)
  - Global verdict metrics and a final "LIKELY TRUE / INCONCLUSIVE / LIKELY FALSE" judgment.

Usage:
  python verify_mask_tokens_padding.py \
    --dir /projects/surgical-video-digital-twin/datasets/cholec80_dt/mask_tokens/video01 \
    --limit 100 \
    --report /projects/surgical-video-digital-twin/datasets/cholec80_dt/mask_tokens/video01_verification.json

You can tweak --eps to set the zero threshold (default 1e-8).
"""

import argparse, json, math, sys
from pathlib import Path
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory of *.npy (e.g., .../mask_tokens/video01)")
    ap.add_argument("--limit", type=int, default=200, help="Max files to sample")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    ap.add_argument("--eps", type=float, default=1e-8, help="Zero threshold for padding detection")
    ap.add_argument("--report", default="", help="Optional path to write a JSON report")
    ap.add_argument("--print-first", type=int, default=3, help="Print detailed rows for first N files")
    return ap.parse_args()

def safe_load(path: Path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        return e

def is_matrix(x, cols=None):
    return isinstance(x, np.ndarray) and x.ndim == 2 and (cols is None or x.shape[1] == cols)

def stats_1d(x):
    if x.size == 0: 
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x,25)),
        "p50": float(np.percentile(x,50)),
        "p75": float(np.percentile(x,75)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }

def corr(a, b):
    a = a.ravel(); b = b.ravel()
    if a.size < 3 or b.size < 3 or a.size != b.size: 
        return float("nan")
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0,1])

def analyze_file(path: Path, eps: float, print_detail: bool=False):
    arr = safe_load(path)
    out = {"file": str(path), "ok_shape": False, "K": None, "D": None,
           "effective_K": None, "pad_tail": None, "pad_indices": None,
           "score_descending": None, "zero_rows": None,
           "last_vs_norm_corr": None,
           "notes": []}
    if isinstance(arr, Exception):
        out["notes"].append(f"load_error: {arr}")
        return out

    if not is_matrix(arr):
        out["notes"].append(f"not_2d_matrix: dtype={getattr(arr,'dtype',None)}, shape={getattr(arr,'shape',None)}")
        return out

    K, D = arr.shape
    out["ok_shape"] = True
    out["K"], out["D"] = int(K), int(D)

    if D < 2:
        out["notes"].append("too_few_columns")
        return out

    emb = arr[:, :min(256, D-1)].astype(np.float64)
    score = arr[:, -1].astype(np.float64)
    norms = np.linalg.norm(emb, axis=1)

    # zero row = tiny norm and tiny |score|
    pad = (norms < eps) & (np.abs(score) < eps)
    out["zero_rows"] = int(pad.sum())
    out["effective_K"] = int((~pad).sum())
    out["pad_indices"] = np.where(pad)[0].tolist()

    # padding should be a contiguous tail (all valid first, then all pad)
    if pad.any():
        first_pad = int(np.argmax(pad))  # first True index
        # if no True, argmax returns 0 but pad.any() ensures at least one True; OK.
        head_valid = (~pad[:first_pad]).all()
        tail_all_pad = pad[first_pad:].all()
        out["pad_tail"] = bool(head_valid and tail_all_pad)
    else:
        out["pad_tail"] = True  # no padding present also "OK"

    # descending score check on valid rows
    valid = ~pad
    if valid.sum() >= 2:
        vscore = score[valid]
        out["score_descending"] = bool(np.all(vscore[:-1] >= vscore[1:]))
    else:
        out["score_descending"] = True  # trivial

    # correlation: last-dim vs embedding norm
    c = corr(score[valid], norms[valid]) if valid.sum() >= 3 else float("nan")
    out["last_vs_norm_corr"] = c

    if print_detail:
        np.set_printoptions(suppress=True, linewidth=200, precision=6)
        print("="*100)
        print(f"[FILE] {path.name}  shape=({K},{D})")
        print("norms:", norms)
        print("score:", score)
        print("pad?  ", pad.astype(int))
        print(f"effective_K={out['effective_K']}, pad_tail={out['pad_tail']}, score_desc={out['score_descending']}, corr(last,norm)={c:.3f}")

    return out

def main():
    args = parse_args()
    root = Path(args.dir)
    files = sorted(root.glob("*.npy"))
    if not files:
        print(f"[ERR] No .npy under {root}", file=sys.stderr)
        sys.exit(2)

    rng = np.random.RandomState(args.seed)
    if len(files) > args.limit:
        idx = rng.choice(len(files), size=args.limit, replace=False)
        files = [files[i] for i in sorted(idx)]

    results = []
    printed = 0
    for p in files:
        r = analyze_file(p, args.eps, print_detail=(printed < args.print_first))
        if printed < args.print_first:
            printed += 1
        results.append(r)

    # Aggregate
    K_list = [r["K"] for r in results if r["K"] is not None]
    D_list = [r["D"] for r in results if r["D"] is not None]
    eff_list = [r["effective_K"] for r in results if r["effective_K"] is not None]
    pad_tail_ok = [r["pad_tail"] for r in results if r["pad_tail"] is not None]
    score_desc_ok = [r["score_descending"] for r in results if r["score_descending"] is not None]
    corr_list = [r["last_vs_norm_corr"] for r in results if r["last_vs_norm_corr"] is not None and not math.isnan(r["last_vs_norm_corr"])]

    agg = {
        "dir": str(root),
        "files_checked": len(results),
        "K_unique": sorted(list(set(int(k) for k in K_list))) if K_list else [],
        "D_unique": sorted(list(set(int(d) for d in D_list))) if D_list else [],
        "effective_K_stats": stats_1d(np.array(eff_list)) if eff_list else {},
        "pad_tail_fraction": float(sum(bool(x) for x in pad_tail_ok)/len(pad_tail_ok)) if pad_tail_ok else None,
        "score_desc_fraction": float(sum(bool(x) for x in score_desc_ok)/len(score_desc_ok)) if score_desc_ok else None,
        "corr_last_vs_norm_stats": stats_1d(np.array(corr_list)) if corr_list else {},
        "sample_examples": results[:min(5, len(results))],  # attach a few for quick glance
    }

    # Verdict heuristics:
    # - D==257 for (all or majority)
    # - pad_tail_fraction high (>=0.9)
    # - score_desc_fraction high (>=0.8)
    # - corr_last_vs_norm mean near 0 (|mean| < 0.3)
    d_ok = (agg["D_unique"] == [257]) or (257 in agg["D_unique"] and len(agg["D_unique"])==1)
    pad_ok = (agg["pad_tail_fraction"] is not None and agg["pad_tail_fraction"] >= 0.9)
    desc_ok = (agg["score_desc_fraction"] is not None and agg["score_desc_fraction"] >= 0.8)
    corr_mean = agg["corr_last_vs_norm_stats"].get("mean", 0.0)
    corr_ok = (abs(corr_mean) < 0.3) if corr_mean == corr_mean else True  # NaN -> True (not enough data)

    if d_ok and pad_ok and desc_ok and corr_ok:
        verdict = "LIKELY TRUE: (K,257) with tail zero-padding; last dim behaves like score/logit; valid rows are sorted."
    elif d_ok and (pad_ok or desc_ok):
        verdict = "PLAUSIBLE: (K,257) consistent; partial evidence of tail padding and/or score ordering."
    else:
        verdict = "INCONCLUSIVE or LIKELY FALSE: dimensions or patterns do not match the hypothesis."

    agg["verdict"] = verdict

    print("\n" + "#"*100)
    print("# SUMMARY")
    print("#  K_unique:", agg["K_unique"])
    print("#  D_unique:", agg["D_unique"])
    print("#  effective_K (median/min/max):",
          agg["effective_K_stats"].get("p50"), agg["effective_K_stats"].get("min"), agg["effective_K_stats"].get("max"))
    print(f"#  pad_tail_fraction: {agg['pad_tail_fraction']}")
    print(f"#  score_desc_fraction: {agg['score_desc_fraction']}")
    print("#  corr_last_vs_norm mean/std:",
          agg["corr_last_vs_norm_stats"].get("mean"), agg["corr_last_vs_norm_stats"].get("std"))
    print("#  VERDICT:", verdict)
    print("#"*100)

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[OK] Report saved to: {rp}")

if __name__ == "__main__":
    main()
