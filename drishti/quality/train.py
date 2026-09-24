"""Learned gradability model + rule-threshold fitting for the quality gate.

The CNN learns DDR's gradable-vs-ungradable split only within DDR, so it
cannot shortcut on "which dataset is this" (every other source is all
gradable). It sees the raw crop, not the Graham-normalised one, because
exposure is exactly what normalisation removes.

Rule thresholds (dark / bright / evenness / focus) are set at the 3rd / 97th
percentile of *gradable* DDR-valid images, so a rule fires only for images
worse than almost every gradable one.

    python -m drishti.quality.train --name qual_b0
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..config import PROCESSED, RESULTS, RUNS, SEED
from ..data.datasets import field_augment, to_tensor
from ..eval.metrics import binary_report
from ..preprocess import read_rgb
from .gate import CFG_PATH, DEFAULTS, metrics

SIZE = 384


class QData(Dataset):
    def __init__(self, df, train):
        self.paths, self.y = df.cache.tolist(), df.gradable.to_numpy()
        self.aug = field_augment(strong=False) if train else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        img = cv2.resize(read_rgb(self.paths[i]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if self.aug is not None:
            img = self.aug(image=img)["image"]
        return to_tensor(img), float(self.y[i])


def build(pretrained=True):
    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)


@torch.no_grad()
def predict(model, df, bs=48):
    model.eval()
    out = []
    for x, _ in DataLoader(QData(df, False), bs, num_workers=2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(x.cuda()).float()).squeeze(1).cpu())
    return torch.cat(out).numpy()


def fit_rule_thresholds(df_valid: pd.DataFrame) -> dict:
    ok = df_valid[df_valid.gradable == 1].sample(min(800, int((df_valid.gradable == 1).sum())), random_state=SEED)

    def one(path):
        return metrics(read_rgb(path))

    with ThreadPoolExecutor(8) as ex:
        ms = [m for m in ex.map(one, ok.path) if not m.get("no_retina")]
    m = pd.DataFrame(ms)
    return {
        "dark_median": float(m["median"].quantile(0.03)),
        "bright_median": float(m["median"].quantile(0.97)),
        "max_clipped": float(max(m["clipped"].quantile(0.97), 0.02)),
        "min_evenness": float(m["evenness"].quantile(0.03)),
        "min_focus": float(m["focus"].quantile(0.03)),
        "min_fov_fraction": float(min(m["fov_fraction"].quantile(0.01), DEFAULTS["min_fov_fraction"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="qual_b0")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED / "manifest.csv")
    ddr = df[(df.source == "ddr") & (df.ok == 1)]
    tr, va, te = ddr[ddr.split == "train"], ddr[ddr.split.isin(["valA", "valB"])], ddr[ddr.split == "test"]
    print("ungradable in train/valid/test:", int((tr.gradable == 0).sum()), int((va.gradable == 0).sum()), int((te.gradable == 0).sum()))

    w = np.where(tr.gradable == 0, 1.0 / (tr.gradable == 0).mean(), 1.0 / (tr.gradable == 1).mean())
    dl = DataLoader(QData(tr, True), 32, sampler=WeightedRandomSampler(torch.as_tensor(w), len(tr), True), num_workers=4, drop_last=True, persistent_workers=True)
    model = build().cuda()
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.epochs * len(dl), pct_start=0.1)
    for ep in range(args.epochs):
        model.train()
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x.cuda()).float().squeeze(1)
            loss = F.binary_cross_entropy_with_logits(lg, y.float().cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(model, va)
        print(json.dumps({"epoch": ep, "valid_auc": round(roc_auc_score(va.gradable, pv), 4)}), flush=True)
    torch.save({"state": model.state_dict()}, out / "best.pt")

    # threshold: keep >= 97% of gradable images (valid), i.e. few needless retakes
    pv = predict(model, va)
    thr = float(np.quantile(pv[va.gradable.to_numpy() == 1], 0.03))
    cfg = dict(DEFAULTS)
    cfg.update(fit_rule_thresholds(va))
    cfg["gradability_threshold"] = thr
    CFG_PATH.parent.mkdir(exist_ok=True)
    CFG_PATH.write_text(json.dumps(cfg, indent=2))

    rep = {"threshold": thr, "rule_thresholds": cfg}
    pt = predict(model, te)
    rep["ddr_test"] = {"auc": roc_auc_score(te.gradable, pt),
                       **{k: v for k, v in binary_report(te.gradable == 0, pt < thr).items()}}
    # every DDR ungradable image is in the 007 series, so the fair number is
    # the one within that series (no "which camera" shortcut available)
    s7 = (te.ddr_series == "007").to_numpy()
    rep["ddr_test_007_series_only"] = {"auc": roc_auc_score(te.gradable[s7], pt[s7]),
                                       **binary_report(te.gradable[s7] == 0, pt[s7] < thr)}
    m2 = df[(df.source == "messidor2") & (df.ok == 1)]
    if len(m2) and (m2.gradable == 0).any():
        pm = predict(model, m2)
        rep["messidor2"] = {"auc": roc_auc_score(m2.gradable, pm), **binary_report(m2.gradable == 0, pm < thr)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"quality_{args.name}.json").write_text(json.dumps(rep, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
