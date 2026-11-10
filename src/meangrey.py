"""
meangrey.py  —  physics baseline: phase fractions from the crop's MEAN grey value.

Because the simulator is linear and monochromatic, the mean reconstructed attenuation of a crop is
  mu_bar = mu_f*f + mu_m*m + mu_v*v = 0.8 + 0.2 f - 0.8 v   (mu_f=1.0, mu_m=0.8, mu_v=0)
so f = 5 mu_bar - 4 + 4 v. Void v comes from the GMM histogram baseline (or the truth, as an
oracle bound). A calibrated variant fits f = a + b mu_bar + c v_gmm on VALIDATION crops.
It needs absolute grey-level calibration; the CNN does not (its input is z-scored per crop),
which the gain test below quantifies.

Also re-runs the trained CNN on the same crops (projector-change robustness check).

Run: python meangrey.py --data <regen dir> --ckpt best.pt --gpu_preds predictions_plus.csv --out <dir>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from model import PhaseFractionResNet3D
from baselines import gmm_fractions

DOSES = ("high", "medium", "low")


def crops(data, split, c=64):
    meta = pd.read_csv(data / "parents.csv")
    for pid in meta.loc[meta.split == split, "id"]:
        cnt = np.load(data / f"p{pid:03d}_counts.npy")
        L = cnt.shape[-1]
        for d in DOSES:
            rec = np.load(data / f"p{pid:03d}_rec_{d}.npy").astype(np.float32)
            for z in range(0, L - c + 1, c):
                for y in range(0, L - c + 1, c):
                    for x in range(0, L - c + 1, c):
                        blk = cnt[:, z:z+c, y:y+c, x:x+c].sum(axis=(1, 2, 3)) / (c**3 * 8)
                        yield (pid, d, z, y, x, 100 * blk[0], 100 * blk[1],
                               rec[z:z+c, y:y+c, x:x+c])


def cnn_predict(model, v):
    with torch.no_grad():
        t = torch.from_numpy((v - v.mean()) / (v.std() + 1e-6))[None, None]
        return model(t)[0].numpy() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu_preds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = PhaseFractionResNet3D(); model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); model.eval()

    rows = []
    for split in ["val", "test"]:
        for pid, d, z, y, x, tf, tv, v in crops(data, split):
            g = gmm_fractions(v)
            r = {"split": split, "parent": pid, "dose": d, "z": z, "y": y, "x": x,
                 "true_fibre": tf, "true_void": tv, "mu_bar": float(v.mean()),
                 "gmm_void": 100 * g["void"]}
            if split == "test":
                p = cnn_predict(model, v)
                r |= {"cnn_fibre": p[0], "cnn_void": p[2]}
                # gain test: 2% and 5% calibration error
                for gain in (1.02, 1.05):
                    pg = cnn_predict(model, v * gain)
                    r[f"cnn_fibre_g{gain}"] = pg[0]
                    r[f"mu_bar_g{gain}"] = float(v.mean() * gain)
            rows.append(r)
    df = pd.DataFrame(rows)

    # estimators (percent units)
    phys = lambda mu, vv: 100 * (5 * mu - 4) + 4 * vv
    df["mg_phys_fibre"] = phys(df.mu_bar, df.gmm_void)
    df["mg_oracle_fibre"] = phys(df.mu_bar, df.true_void)
    val = df[df.split == "val"]
    X = np.c_[np.ones(len(val)), val.mu_bar, val.gmm_void]
    coef, *_ = np.linalg.lstsq(X, val.true_fibre, rcond=None)
    df["mg_cal_fibre"] = coef[0] + coef[1] * df.mu_bar + coef[2] * df.gmm_void
    te = df[df.split == "test"].copy()
    for gain in (1.02, 1.05):
        te[f"mg_cal_fibre_g{gain}"] = coef[0] + coef[1] * te[f"mu_bar_g{gain}"] + coef[2] * te.gmm_void

    # GPU-run CNN predictions for the same crops (from Colab)
    gp = pd.read_csv(a.gpu_preds)[["parent", "dose", "z", "y", "x", "true_fibre", "cnn_fibre"]]
    gp = gp.rename(columns={"true_fibre": "gpu_true_fibre", "cnn_fibre": "gpu_cnn_fibre"})
    te = te.merge(gp, on=["parent", "dose", "z", "y", "x"], how="left")
    te.to_csv(out / "meangrey_test_crops.csv", index=False)

    def mae(col, sub=te):
        return (sub[col] - sub.true_fibre).abs().mean()

    res = {
        "n_test_crops": len(te),
        "max |true fibre CPU - GPU| (pp)": float((te.true_fibre - te.gpu_true_fibre).abs().max()),
        "CNN on GPU recon (Colab)": mae("gpu_cnn_fibre"),
        "CNN on CPU recon (re-simulated)": mae("cnn_fibre"),
        "Mean-grey, physics constants + GMM void": mae("mg_phys_fibre"),
        "Mean-grey, calibrated on val + GMM void": mae("mg_cal_fibre"),
        "Mean-grey, physics constants + TRUE void (oracle)": mae("mg_oracle_fibre"),
        "calibration coefficients a,b,c": [round(float(c_), 4) for c_ in coef],
    }
    for gain in (1.02, 1.05):
        res[f"gain {gain}: mean-grey calibrated"] = mae(f"mg_cal_fibre_g{gain}")
        res[f"gain {gain}: CNN"] = mae(f"cnn_fibre_g{gain}")
        res[f"gain {gain}: max |CNN change| (pp)"] = float((te[f"cnn_fibre_g{gain}"] - te.cnn_fibre).abs().max())
    by_dose = {d: {"cnn": mae("cnn_fibre", te[te.dose == d]), "mg_cal": mae("mg_cal_fibre", te[te.dose == d])}
               for d in DOSES}
    per = te.groupby("parent").apply(lambda g: pd.Series({
        "cnn": (g.cnn_fibre - g.true_fibre).abs().mean(),
        "mg_cal": (g.mg_cal_fibre - g.true_fibre).abs().mean()}), include_groups=False)
    w = wilcoxon(per.cnn, per.mg_cal)
    res["per-parent: CNN better on"] = f"{int((per.cnn < per.mg_cal).sum())}/{len(per)}"
    res["Wilcoxon p (CNN vs calibrated mean-grey)"] = float(w.pvalue)
    for k, v in res.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("by dose:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in by_dose.items()})
    pd.Series({k: str(v) for k, v in res.items()}).to_csv(out / "meangrey_summary.csv")
    pd.DataFrame(by_dose).T.to_csv(out / "meangrey_by_dose.csv")


if __name__ == "__main__":
    main()
