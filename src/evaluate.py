"""
evaluate.py  —  test-set evaluation of CNN vs baselines on IDENTICAL crops.

Produces (in --out):
  predictions.csv     one row per crop: parent, dose, position, truth, CNN, Otsu, GMM
  metrics.csv         MAE / RMSE / R^2 / bias per method x phase x dose, with 95% bootstrap CIs
                      (bootstrap resamples PARENT phantoms, because crops from one parent are
                      correlated - resampling crops would give falsely narrow intervals)
  scatter_*.png       predicted vs true, per phase
  bland_altman_*.png  CNN vs truth and CNN vs Otsu agreement plots

Run:  python evaluate.py --data ../data --ckpt ../runs/c64/best.pt --crop 64 --split test
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import PhaseFractionResNet3D
from train import CropDataset, evaluate, DOSES
from baselines import otsu_count, gmm_fractions

PH = ["fibre", "matrix", "void"]


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan


def boot_ci(df, fn, n=1000, seed=0):
    """95% CI of statistic fn(df) by resampling parent phantoms (cluster bootstrap)."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby("parent")}
    keys = list(groups)
    vals = [fn(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def bland_altman(a, b, title, path, units="percentage points"):
    m, d = (a + b) / 2, a - b
    bias, sd = d.mean(), d.std(ddof=1)
    plt.figure(figsize=(5.5, 4))
    plt.scatter(m, d, s=8, alpha=0.5)
    for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
        plt.axhline(y, ls=ls, c="k", lw=1)
    plt.title(f"{title}\nbias {bias:.3f}, 95% LoA [{bias-1.96*sd:.3f}, {bias+1.96*sd:.3f}]",
              fontsize=10)
    plt.xlabel(f"mean of methods ({units})"); plt.ylabel(f"difference ({units})")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--ckpt", default="../runs/c64/best.pt")
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or Path(a.ckpt).parent / f"eval_{a.split}_c{a.crop}"); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = CropDataset(a.data, a.split, a.crop, DOSES, grid=True)
    model = PhaseFractionResNet3D().to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    P, T, M = evaluate(model, DataLoader(ds, 16), dev)

    rows = []
    for k, (i, d, z, y, x) in enumerate(M):
        c = a.crop
        raw = np.asarray(ds.rec[(int(i), DOSES[d])][z:z + c, y:y + c, x:x + c], np.float32)
        fo, _ = otsu_count(raw)
        fg = gmm_fractions(raw)
        row = {"parent": int(i), "dose": DOSES[d], "z": z, "y": y, "x": x}
        for j, ph in enumerate(PH):
            row[f"true_{ph}"] = T[k, j] * 100
            row[f"cnn_{ph}"] = P[k, j] * 100
            row[f"otsu_{ph}"] = fo[ph] * 100
            row[f"gmm_{ph}"] = fg[ph] * 100
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(out / "predictions.csv", index=False)

    met = []
    for dose in ["all", *DOSES]:
        sub = df if dose == "all" else df[df.dose == dose]
        for meth in ["cnn", "otsu", "gmm"]:
            for ph in PH:
                e = sub[f"{meth}_{ph}"] - sub[f"true_{ph}"]
                mae_fn = lambda g, m=meth, p=ph: (g[f"{m}_{p}"] - g[f"true_{p}"]).abs().mean()
                lo, hi = boot_ci(sub, mae_fn, n=500)
                met.append({"dose": dose, "method": meth, "phase": ph, "n_crops": len(sub),
                            "MAE_pp": e.abs().mean(), "MAE_CI_lo": lo, "MAE_CI_hi": hi,
                            "RMSE_pp": np.sqrt((e ** 2).mean()), "bias_pp": e.mean(),
                            "R2": r2(sub[f"true_{ph}"].values, sub[f"{meth}_{ph}"].values)})
    mt = pd.DataFrame(met); mt.to_csv(out / "metrics.csv", index=False)
    print(mt[mt.dose == "all"].round(3).to_string(index=False))

    for ph in PH:
        plt.figure(figsize=(5, 5))
        for meth, mk in [("otsu", "x"), ("gmm", "+"), ("cnn", "o")]:
            plt.scatter(df[f"true_{ph}"], df[f"{meth}_{ph}"], s=10, marker=mk, alpha=0.6, label=meth)
        lim = [df[f"true_{ph}"].min(), df[f"true_{ph}"].max()]
        plt.plot(lim, lim, "k-", lw=1); plt.legend()
        plt.xlabel(f"true {ph} (%)"); plt.ylabel(f"predicted {ph} (%)")
        plt.tight_layout(); plt.savefig(out / f"scatter_{ph}.png", dpi=130); plt.close()
        bland_altman(df[f"cnn_{ph}"].values, df[f"true_{ph}"].values,
                     f"{ph}: CNN vs ground truth", out / f"bland_altman_{ph}_cnn_vs_truth.png")
        bland_altman(df[f"cnn_{ph}"].values, df[f"otsu_{ph}"].values,
                     f"{ph}: CNN vs Otsu counting", out / f"bland_altman_{ph}_cnn_vs_otsu.png")


if __name__ == "__main__":
    main()
