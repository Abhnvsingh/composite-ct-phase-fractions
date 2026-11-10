"""
stats.py  —  strengthen Experiment 1 for review:
  1. a FAIR classical baseline: multi-Otsu after Gaussian smoothing (sigma=1 voxel), computed on
     exactly the same test crops as evaluate.py (positions read from predictions.csv)
  2. paired significance: per-parent MAE of the CNN vs each baseline, Wilcoxon signed-rank test
     (pairs = test parent phantoms, because crops from one parent are not independent)
  3. seed stability: CNN MAE mean +- SD over several training seeds (optional)

Run:
  python stats.py --data ../data --eval ../runs/c64/eval_test_c64 --out ../runs/c64/stats \
                  --seed_evals ../runs/c64/eval_test_c64 ../runs/c64_seed1/eval_test_c64 ../runs/c64_seed2/eval_test_c64
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from baselines import otsu_count
from evaluate import boot_ci

PH = ["fibre", "matrix", "void"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--eval", required=True, help="evaluate.py output folder (64^3 test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed_evals", nargs="*", default=[])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    crop = int(str(a.eval).rstrip("/").split("_c")[-1])

    # 1. smoothed multi-Otsu on the identical crops
    df = pd.read_csv(Path(a.eval) / "predictions.csv")
    cache = {}
    rows = []
    for r in df.itertuples():
        key = (r.parent, r.dose)
        if key not in cache:
            cache = {key: np.load(data / f"p{r.parent:03d}_rec_{r.dose}.npy", mmap_mode="r")}
        v = np.asarray(cache[key][r.z:r.z + crop, r.y:r.y + crop, r.x:r.x + crop], np.float32)
        f, _ = otsu_count(v, smooth=a.sigma)
        rows.append({f"otsus_{ph}": f[ph] * 100 for ph in PH})
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    df.to_csv(out / "predictions_plus.csv", index=False)

    methods = {"cnn": "3D ResNet (CNN)", "otsu": "Multi-Otsu, raw",
               "otsus": f"Multi-Otsu, Gaussian-smoothed (sigma={a.sigma})", "gmm": "GMM histogram"}
    tab = []
    for m, name in methods.items():
        for ph in PH:
            e = df[f"{m}_{ph}"] - df[f"true_{ph}"]
            lo, hi = boot_ci(df, lambda g, m=m, p=ph: (g[f"{m}_{p}"] - g[f"true_{p}"]).abs().mean(), n=1000)
            tab.append({"method": name, "phase": ph, "MAE_pp": e.abs().mean(), "CI95_lo": lo,
                        "CI95_hi": hi, "bias_pp": e.mean()})
    tab = pd.DataFrame(tab); tab.to_csv(out / "exp1_with_fair_baseline.csv", index=False)

    # 2. paired test per parent phantom
    per = df.groupby("parent").apply(
        lambda g: pd.Series({f"{m}_{ph}": (g[f"{m}_{ph}"] - g[f"true_{ph}"]).abs().mean()
                             for m in methods for ph in PH}), include_groups=False)
    tests = []
    for ph in PH:
        for m in ["otsu", "otsus", "gmm"]:
            diff = per[f"{m}_{ph}"] - per[f"cnn_{ph}"]          # >0 means CNN better
            stat, p = wilcoxon(per[f"cnn_{ph}"], per[f"{m}_{ph}"])
            tests.append({"phase": ph, "CNN vs": methods[m], "n_parents": len(per),
                          "parents_where_CNN_better": int((diff > 0).sum()),
                          "median_MAE_gain_pp": diff.median(), "wilcoxon_p": p})
    tests = pd.DataFrame(tests); tests.to_csv(out / "paired_tests.csv", index=False)

    # 3. seed stability
    if a.seed_evals:
        s = []
        for e in a.seed_evals:
            d = pd.read_csv(Path(e) / "predictions.csv")
            s.append({ph: (d[f"cnn_{ph}"] - d[f"true_{ph}"]).abs().mean() for ph in PH} | {"run": str(e)})
        s = pd.DataFrame(s)
        summ = pd.DataFrame({"phase": PH, "n_seeds": len(s), "MAE_mean_pp": [s[p].mean() for p in PH],
                             "MAE_sd_pp": [s[p].std(ddof=1) if len(s) > 1 else np.nan for p in PH]})
        s.to_csv(out / "seeds_runs.csv", index=False); summ.to_csv(out / "seeds_summary.csv", index=False)
        print("\n== CNN across training seeds =="); print(summ.round(3).to_string(index=False))

    print("== Experiment 1 incl. fair baseline =="); print(tab.round(3).to_string(index=False))
    print("\n== Paired Wilcoxon tests (per parent) =="); print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
