#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第一步只训练head Stage-2: unfreeze prompt+mask only, very small LR. Train N epochs with per-epoch evaluation.
- Prints dataset info, trainable parameter counts, LR
- tqdm progress bars for train/val (with running loss)
- Saves epoch ckpts (pmstage_epochXXX.pt) and best by macro-F1 (pmstage_best.pt)
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# reuse components from your finetune script
from train_sam2_classify_finetune import (
    FramePointDataset, collate_varlen, FineTuneSam2Wrapper,
    MLPHead, MLPBNHead, CosineClassifier, load_label_map
)

# ----------------- helpers -----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _parse_hidden_list(s: str):
    out = []
    try:
        for x in str(s).split(","):
            x = x.strip()
            if x: out.append(int(x))
    except Exception:
        pass
    return out

def _build_head_from_ckpt_args(in_dim: int, n_classes: int, args_dict: dict):
    head_type = args_dict.get("head", "cosine")
    drop      = float(args_dict.get("drop", 0.0))
    hidden    = args_dict.get("hidden", "0")
    scale     = float(args_dict.get("scale", 16.0))
    if head_type == "linear": return MLPHead(in_dim, n_classes, hidden=0, drop=drop)
    if head_type == "mlp":
        hs = _parse_hidden_list(hidden); h = (hs[0] if hs else 0)
        return MLPHead(in_dim, n_classes, hidden=h, drop=drop)
    if head_type == "mlp_bn":
        hs = _parse_hidden_list(hidden) or [1024, 512]
        return MLPBNHead(in_dim, n_classes, hidden_layers=hs, drop=drop)
    return CosineClassifier(in_dim, n_classes, scale=scale)

def _apply_logit_adjust(logits, log_prior, tau):
    if (log_prior is None) or (tau is None) or (tau <= 0): return logits
    return logits - float(tau) * log_prior.view(1, -1).to(logits.device)

def _metrics(y_true, y_pred, n_classes: int):
    device = y_true.device
    conf = torch.zeros((n_classes, n_classes), dtype=torch.long, device=device)
    for t, p in zip(y_true, y_pred): conf[t, p] += 1
    eps = 1e-12
    tp = conf.diag().float()
    per_cls_total = conf.sum(dim=1).float()
    per_cls_acc = (tp / (per_cls_total + eps)).cpu().numpy()
    pp = conf.sum(dim=0).float()
    prec = tp / (pp + eps); rec = tp / (per_cls_total + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    macro_f1 = torch.nanmean(f1).item()
    return per_cls_acc, macro_f1

# ----------------- train / eval with progress bars -----------------
def train_one_epoch(extractor, head, loader, device, opt,
                    log_prior=None, tau=0.0,
                    amp=True, amp_dtype=torch.float16,
                    bg_mask_mode="mix", bg_mix_p=0.5,
                    max_grad_norm=1.0, epoch=1, total_epochs=1):
    extractor.train(); head.train()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    running_loss, n = 0.0, 0

    pbar = tqdm(loader, total=len(loader), ncols=110,
                desc=f"[Train {epoch}/{total_epochs}]", leave=False)
    for step, batch in enumerate(pbar, 1):
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=amp, dtype=amp_dtype):
            feats = extractor(imgs, pts, metas, targets=y,
                              bg_mask_mode=bg_mask_mode, bg_mix_p=bg_mix_p)
            logits = head(feats)
            logits = _apply_logit_adjust(logits, log_prior, tau)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(extractor.parameters()) + list(head.parameters()), max_grad_norm)
        scaler.step(opt); scaler.update()

        running_loss += float(loss.item()) * y.size(0); n += y.size(0)
        pbar.set_postfix(loss=f"{running_loss/max(1,n):.4f}")
    pbar.close()
    return running_loss / max(1, n)

@torch.inference_mode()
def evaluate(extractor, head, loader, device, n_classes,
             log_prior=None, tau=0.0,
             amp=True, amp_dtype=torch.float16,
             bg_mask_mode="mix", bg_mix_p=0.5, epoch=1, total_epochs=1):
    extractor.eval(); head.eval()
    total_loss, n, correct = 0.0, 0, 0
    y_true_all, y_pred_all = [], []

    pbar = tqdm(loader, total=len(loader), ncols=110,
                desc=f"[Val   {epoch}/{total_epochs}]", leave=False)
    for batch in pbar:
        imgs, pts, metas = batch["images"], batch["points"], batch["meta"]
        y = batch["targets"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp, dtype=amp_dtype):
            feats = extractor(imgs, pts, metas, targets=y,
                              bg_mask_mode=bg_mask_mode, bg_mix_p=bg_mix_p)
            logits = head(feats)
            logits = _apply_logit_adjust(logits, log_prior, tau)
            loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.size(0); n += y.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        y_true_all.append(y); y_pred_all.append(preds)
        pbar.set_postfix(loss=f"{total_loss/max(1,n):.4f}")
    pbar.close()

    val_loss = total_loss / max(1, n)
    val_acc  = correct / max(1, n)
    if len(y_true_all):
        yt = torch.cat(y_true_all, 0); yp = torch.cat(y_pred_all, 0)
        per_cls_acc, macro_f1 = _metrics(yt, yp, n_classes)
    else:
        per_cls_acc, macro_f1 = np.zeros((n_classes,)), 0.0
    return val_loss, val_acc, per_cls_acc, macro_f1

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data-root", type=str, required=True,
                    help="must contain train_manifest.csv, val_manifest.csv, label_map.json")
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)

    # SAM2 + head
    ap.add_argument("--sam2-cfg", type=str, required=True)
    ap.add_argument("--sam2-ckpt", type=str, required=True)
    ap.add_argument("--head-ckpt", type=str, required=True)

    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-backbone", type=float, default=3e-6, help="for prompt+mask")
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--wd-backbone", type=float, default=0.05)
    ap.add_argument("--wd-head", type=float, default=0.05)
    ap.add_argument("--bg-mask-mode", choices=["pos","global","mix"], default="mix")
    ap.add_argument("--bg-mix-p", type=float, default=0.5)
    ap.add_argument("--logit-adjust", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    data_root = Path(args.data_root)
    label_map_path = data_root / "label_map.json"
    train_csv = data_root / "train_manifest.csv"
    val_csv   = data_root / "val_manifest.csv"

    # datasets / loaders
    ds_train = FramePointDataset(train_csv, label_map_path, resize=args.resize)
    ds_val   = FramePointDataset(val_csv,   label_map_path, resize=args.resize)
    print(f"[DATA] train={len(ds_train)}  val={len(ds_val)}  batch_size={args.batch_size}  workers={args.workers}")

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_varlen, pin_memory=True,
        persistent_workers=(args.workers > 0), prefetch_factor=2 if args.workers>0 else None
    )
    dl_val   = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_varlen, pin_memory=True,
        persistent_workers=(args.workers > 0), prefetch_factor=2 if args.workers>0 else None
    )

    # label map
    lm = load_label_map(label_map_path)
    tool2id = lm["tool_to_id"]; n_classes = len(tool2id)
    id2name = {int(v): k for k, v in tool2id.items()}

    # SAM2 wrapper: unfreeze prompt+mask only
    extractor = FineTuneSam2Wrapper(args.sam2_cfg, args.sam2_ckpt, device=device)
    extractor.set_trainable("none")
    extractor.set_trainable("prompt+mask")
    extractor.to(device)

    # probe feature dim
    probe = next(iter(dl_train))
    with torch.no_grad():
        feat_probe = extractor(probe["images"][:1], probe["points"][:1], probe["meta"][:1])
    in_dim = int(feat_probe.shape[-1])

    # load head from Stage-1 ckpt
    ckpt = torch.load(args.head_ckpt, map_location="cpu")
    ck_args   = ckpt.get("args", {})
    in_dim_ck = int(ckpt.get("in_dim", in_dim))
    n_cls_ck  = int(ckpt.get("n_classes", n_classes))
    if n_cls_ck != n_classes:
        print(f"[WARN] n_classes mismatch: ckpt={n_cls_ck}, now={n_classes}. Use now={n_classes}.")
        n_cls_ck = n_classes
    head = _build_head_from_ckpt_args(in_dim_ck, n_cls_ck, ck_args).to(device)
    try:
        head.load_state_dict(ckpt.get("head_state", ckpt), strict=True)
    except Exception:
        head.load_state_dict(ckpt.get("head_state", ckpt), strict=False)
        print("[INFO] Loaded head with non-strict=True.")

    # logit-adjust prior
    log_prior, tau = None, float(args.logit_adjust or 0.0)
    if tau > 0:
        import pandas as pd
        df_tr = pd.read_csv(train_csv)
        ids = [tool2id[t] for t in df_tr["tool"] if t in tool2id]
        cnt = np.bincount(ids, minlength=n_classes).astype(np.float64)
        pri = cnt / max(1.0, cnt.sum())
        log_prior = torch.log(torch.tensor(pri + 1e-12, dtype=torch.float32, device=device))
        print(f"[LOGIT-ADJUST] tau={tau:.3f} priors={np.round(pri,4)}")

    # optimizer (diff LR)
    back_params = [p for p in extractor.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": back_params, "lr": args.lr_backbone, "weight_decay": args.wd_backbone},
        {"params": head_params, "lr": args.lr_head,     "weight_decay": args.wd_head},
    ])
    n_trainable = sum(p.numel() for p in back_params) + sum(p.numel() for p in head_params)
    print(f"[MODEL] trainable params={n_trainable/1e6:.2f}M "
          f"(backbone={sum(p.numel() for p in back_params)/1e6:.2f}M, head={sum(p.numel() for p in head_params)/1e6:.2f}M)")
    print(f"[LR] backbone={args.lr_backbone}  head={args.lr_head}")

    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    best_score, best_path = -1.0, out_dir / "pmstage_best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        for i, pg in enumerate(opt.param_groups):
            print(f"  ParamGroup{i}: lr={pg['lr']}")
        tr_loss = train_one_epoch(
            extractor, head, dl_train, device, opt,
            log_prior=log_prior, tau=tau,
            amp=args.amp, amp_dtype=amp_dtype,
            bg_mask_mode=args.bg_mask_mode, bg_mix_p=args.bg_mix_p,
            max_grad_norm=args.max_grad_norm, epoch=epoch, total_epochs=args.epochs
        )
        va_loss, va_acc, per_cls_acc, macro_f1 = evaluate(
            extractor, head, dl_val, device, n_classes,
            log_prior=log_prior, tau=tau,
            amp=args.amp, amp_dtype=amp_dtype,
            bg_mask_mode=args.bg_mask_mode, bg_mix_p=args.bg_mix_p,
            epoch=epoch, total_epochs=args.epochs
        )

        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} | "
              f"val_loss={va_loss:.4f} | val_acc={va_acc:.4f} | macroF1={macro_f1:.4f}")
        print("  per-class acc:")
        for cid, acc in enumerate(per_cls_acc):
            name = id2name.get(cid, f"class{cid}")
            print(f"    {cid:2d} {name:>12s}: {acc:.4f}")

        # save epoch ckpt
        ep_path = out_dir / f"pmstage_epoch{epoch:03d}.pt"
        torch.save({
            "sam2_state": extractor.state_dict(),
            "head_state": head.state_dict(),
            "in_dim": in_dim_ck,
            "n_classes": n_cls_ck,
            "tool_to_id": tool2id,
            "args": {
                "head": ck_args.get("head", "cosine"),
                "hidden": ck_args.get("hidden", "0"),
                "drop": ck_args.get("drop", 0.0),
                "scale": ck_args.get("scale", 16.0),
                "stage2_trainable": "prompt+mask",
                "lr_backbone": args.lr_backbone,
                "lr_head": args.lr_head,
                "logit-adjust": tau,
            },
        }, str(ep_path))

        # best by macro-F1
        score = macro_f1
        if score > best_score:
            best_score = score
            torch.save({
                "sam2_state": extractor.state_dict(),
                "head_state": head.state_dict(),
                "in_dim": in_dim_ck,
                "n_classes": n_cls_ck,
                "tool_to_id": tool2id,
                "args": {
                    "head": ck_args.get("head", "cosine"),
                    "hidden": ck_args.get("hidden", "0"),
                    "drop": ck_args.get("drop", 0.0),
                    "scale": ck_args.get("scale", 16.0),
                    "stage2_trainable": "prompt+mask",
                    "lr_backbone": args.lr_backbone,
                    "lr_head": args.lr_head,
                    "logit-adjust": tau,
                },
            }, str(best_path))
            print(f"  ✓ saved BEST (macroF1={best_score:.4f}) -> {best_path}")

    print(f"\nDone. Best macroF1={best_score:.4f}. Best path: {best_path}")

if __name__ == "__main__":
    main()


# python /home/wcheng31/sam2_classify/train_sam2_classify_finetune_stage2.py \
#   --sam2-cfg sam2_hiera_l.yaml \
#   --sam2-ckpt /projects/surgical-video-digital-twin/pretrain_params/sam2_hiera_large.pt \
#   --head-ckpt /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier/head_epoch010.pt \
#   --data-root /home/wcheng31/sam2_classify/config \
#   --out-dir   /projects/surgical-video-digital-twin/pretrain_params/cwz/sam2_classifier \
#   --epochs 3 --batch-size 16 --workers 4 \
#   --lr-backbone 3e-6 --lr-head 2e-4 \
#   --bg-mask-mode mix --bg-mix-p 0.5 \
#   --logit-adjust 0.5 \
#   --amp
