"""Train the ICDR grader on development data only (APTOS + DDR train).

    python -m drishti.grading.train --name b0_512 --backbone efficientnet_b0

Early stopping and temperature use valA only. After training, predictions
for every split (with horizontal-flip test-time augmentation) are written to
runs/<name>/preds.csv; all later calibration and evaluation read that file,
so the model is never re-run on external data during tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..config import PROCESSED, RUNS, SEED
from ..data.datasets import GradingDataset
from ..eval.metrics import auc_ci, expected_grade, qwk
from .model import RETFOUND, Grader, class_probs, corn_loss, referable_prob, soft_corn_loss


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "manifest.csv", low_memory=False)
    return df[df.ok == 1].reset_index(drop=True)


def make_loader(df, train, bs, workers, size, capture_p=0.0, seed=SEED, graham=True, soft=None):
    ds = GradingDataset(df, train=train, size=size, capture_p=capture_p, seed=seed, graham=graham, soft=soft)
    if train:
        counts = df.grade.value_counts().to_dict()
        w = df.grade.map(lambda g: 1.0 / np.sqrt(counts[g])).to_numpy()
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df), replacement=True)
        return DataLoader(ds, bs, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    # on Windows every worker is a full process with its own torch import
    # (~1.8 GB committed), so evaluation loaders stay in-process there
    return DataLoader(ds, bs, shuffle=False, num_workers=0 if os.name == "nt" else workers,
                      pin_memory=True, persistent_workers=False)


@torch.no_grad()
def predict_logits(model, loader, device, tta=True) -> np.ndarray:
    model.eval()
    out = []
    for x, *_ in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x).float()
            if tta:
                lg = (lg + model(torch.flip(x, dims=[3])).float()) / 2
        out.append(lg.cpu())
    return torch.cat(out).numpy()


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    lg = torch.tensor(logits)
    yt = torch.tensor(y)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        p = class_probs(lg / t).clamp_min(1e-7)
        nll = -torch.log(p[torch.arange(len(yt)), yt]).mean().item()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def evaluate(logits, y):
    lg = torch.tensor(logits)
    probs = class_probs(lg).numpy()
    pref = referable_prob(lg).numpy()
    auc = auc_ci(y >= 2, pref, n_boot=200)[0]
    return {"qwk": qwk(y, expected_grade(probs)), "auc_ref": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap images per split")
    ap.add_argument("--init", default="", help="warm-start from another run's best.pt (e.g. after an interrupted run)")
    ap.add_argument("--capture-p", type=float, default=0.0,
                    help="share of training images re-imaged through a simulated cheap camera")
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay. 1e-4 suits the EfficientNets; ConvNeXt and the "
                         "transformer backbones expect roughly 0.05 and train poorly without it")
    ap.add_argument("--sources", default="",
                    help="comma-separated training sources; default is every development source. "
                         "Pin this when a run is one arm of a comparison, so that adding a dataset "
                         "to the manifest later cannot quietly change what the run trained on")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decay of an exponential moving average of the weights (e.g. 0.9998); "
                         "when set, the average is what is validated, checkpointed and exported")
    ap.add_argument("--no-graham", action="store_true",
                    help="feed plain crops instead of Graham-normalised ones (RETFound was pretrained on photographs)")
    ap.add_argument("--layer-decay", type=float, default=0.65,
                    help="ViT backbones only: learning rate shrinks by this factor per block from the top")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (large ViTs at high resolution)")
    ap.add_argument("--teacher", default="",
                    help="distil from this run's preds_train.csv (teacher class probabilities on the training images)")
    ap.add_argument("--kd-alpha", type=float, default=0.5, help="weight of the teacher term; 1 - alpha goes to the hard grades")
    ap.add_argument("--predict-train", action="store_true",
                    help="also write preds_train.csv (flip-TTA class probabilities on the training images), "
                         "which a later run can use as --teacher")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="training seed (initialisation of the head, sampling, augmentation); "
                         "the data splits do not depend on it")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    df = load_manifest()
    dev = df[(df.role == "dev") & (df.grade >= 0)]
    train_df = dev[dev.split == "train"]
    if args.sources:
        want = [x.strip() for x in args.sources.split(",")]
        missing = set(want) - set(train_df.source.unique())
        if missing:
            sys.exit(f"--sources names {sorted(missing)}, which is not in the manifest's training rows")
        train_df = train_df[train_df.source.isin(want)]
    val_df = dev[dev.split == "valA"]
    if args.limit:
        train_df = train_df.sample(min(args.limit, len(train_df)), random_state=SEED)
        val_df = val_df.sample(min(args.limit // 4, len(val_df)), random_state=SEED)
    col = f"cache{args.size}"
    if args.size != 512 and not (col in df and (df[col].astype(str) != "").all()):
        sys.exit(f"no {col} column in the manifest: run scripts/cache_size.py --size {args.size} first "
                 f"(training at {args.size} from 512-px crops would just upsample)")
    print(f"train {len(train_df)}  valA {len(val_df)}  crops from {col if args.size != 512 else 'cache'}")
    print(train_df.groupby(["source", "grade"]).size().unstack(fill_value=0).to_string())

    device = "cuda"
    graham = not args.no_graham
    model = Grader(args.backbone, img_size=args.size).to(device).to(memory_format=torch.channels_last)
    if args.grad_ckpt:
        model.body.set_grad_checkpointing(True)
    if args.init:
        ck0 = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck0["state"])
        print(f"warm-started from {args.init} (epoch {ck0.get('epoch')}, val {ck0.get('val')})")
    soft = None
    if args.teacher:
        tp = pd.read_csv(RUNS / args.teacher / "preds_train.csv").set_index("image_id")
        missing = ~train_df.image_id.isin(tp.index)
        if missing.any():
            sys.exit(f"teacher {args.teacher} has no prediction for {int(missing.sum())} training images")
        soft = tp.loc[train_df.image_id, [f"p{k}" for k in range(5)]].to_numpy()
        print(f"distilling from {args.teacher}, alpha {args.kd_alpha}")
    tl = make_loader(train_df, True, args.bs, args.workers, args.size, capture_p=args.capture_p, seed=args.seed,
                     graham=graham, soft=soft)
    vl = make_loader(val_df, False, args.bs * 2, args.workers, args.size, graham=graham)

    if args.backbone in RETFOUND:
        # layer-wise learning-rate decay, the standard recipe for fine-tuning
        # a pretrained ViT: early blocks move least
        from timm.optim._param_groups import param_groups_layer_decay
        groups = param_groups_layer_decay(model.body, weight_decay=args.weight_decay, layer_decay=args.layer_decay)
        for g in groups:
            g["lr"] = args.lr * g.pop("lr_scale", 1.0)
        groups.append({"params": list(model.head.parameters()), "lr": args.lr * 3, "weight_decay": args.weight_decay})
    else:
        groups = [{"params": model.body.parameters(), "lr": args.lr},
                  {"params": model.head.parameters(), "lr": args.lr * 3}]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                total_steps=steps, pct_start=0.1)

    ema = None
    if args.ema:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=args.ema, use_warmup=True)

    best, log, gstep = -1.0, [], 0
    yv = val_df.grade.to_numpy()
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in tl:
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x)
            loss = corn_loss(lg.float(), y)
            if soft is not None:
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * soft_corn_loss(lg.float(), batch[2].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            gstep += 1
            if ema is not None:
                ema.update(model, step=gstep)
            tot += loss.item() * len(y)
            n += len(y)
        judged = ema.module if ema is not None else model
        lv = predict_logits(judged, vl, device, tta=False)
        m = evaluate(lv, yv)
        score = m["auc_ref"] + 0.25 * m["qwk"]
        rec = {"epoch": ep, "loss": tot / n, **m, "sec": round(time.time() - t0)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if score > best:
            best = score
            torch.save({"state": judged.state_dict(), "backbone": args.backbone, "size": args.size, "epoch": ep, "val": m,
                        "graham": graham}, out / "best.pt")
    (out / "log.json").write_text(json.dumps(log, indent=2))

    # reload best, fit temperature on valA, predict every split
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state"])
    lv = predict_logits(model, make_loader(val_df, False, args.bs * 2, args.workers, args.size, graham=graham), device)
    t = fit_temperature(lv, yv)
    model.temperature.fill_(t)
    ck["state"] = model.state_dict()
    ck["temperature"] = t
    torch.save(ck, out / "best.pt")
    print(f"best epoch {ck['epoch']}  temperature {t:.2f}")

    pred_df = df[df.split.isin(["valA", "valB", "devext", "devext2", "test", "external"])].copy()
    if args.limit:
        pred_df = pred_df.groupby("source").head(args.limit // 4)
    pred_df = pred_df.assign(grade_label=pred_df.grade)
    lg = predict_logits(model, make_loader(pred_df.assign(grade=0), False, args.bs * 2, args.workers, args.size,
                                           graham=graham), device)
    lg_t = torch.tensor(lg) / t
    probs = class_probs(lg_t).numpy()
    res = pred_df[["image_id", "source", "split", "grade", "dme", "gradable", "exam_id", "eye", "dup_of_dev"]].copy()
    for k in range(5):
        res[f"p{k}"] = probs[:, k]
    res["p_ref"] = referable_prob(lg_t).numpy()
    for k in range(4):
        res[f"logit{k}"] = lg[:, k]
    res.to_csv(out / "preds.csv", index=False)
    print(f"wrote {out / 'preds.csv'} ({len(res)} rows)")

    if args.predict_train:
        tr = dev[dev.split == "train"]
        lg = predict_logits(model, make_loader(tr.assign(grade=0), False, args.bs * 2, args.workers, args.size,
                                               graham=graham), device)
        pt = class_probs(torch.tensor(lg) / t).numpy()
        tres = tr[["image_id", "source", "grade"]].copy()
        for k in range(5):
            tres[f"p{k}"] = pt[:, k]
        tres.to_csv(out / "preds_train.csv", index=False)
        print(f"wrote {out / 'preds_train.csv'} ({len(tres)} rows)")


if __name__ == "__main__":
    main()
