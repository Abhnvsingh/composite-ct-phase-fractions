"""
experiments.py  —  Experiments 2-5 from evaluate.py outputs, no retraining needed.

Exp 2 (sub-volume size): run evaluate.py with the SAME checkpoint at --crop 32, 64, 96 (the
        network's global average pool accepts any cube size), then pass all eval folders here.
Exp 3 (void level), Exp 4 (packing / diameter variability), Exp 5 (noise): bin the 64^3 test
        predictions by the parent phantom's parameters and by dose.

Run:
  python evaluate.py --data ../data --ckpt ../runs/c64/best.pt --crop 32 --split test
  python evaluate.py --data ../data --ckpt ../runs/c64/best.pt --crop 64 --split test
  python evaluate.py --data ../data --ckpt ../runs/c64/best.pt --crop 96 --split test
  python experiments.py --data ../data --evals ../runs/c64/eval_test_c32 ../runs/c64/eval_test_c64 \
                        ../runs/c64/eval_test_c96 --main ../runs/c64/eval_test_c64 --out ../runs/c64/experiments
Outputs: exp2_size.csv/png, exp3_void.csv/png, exp4_packing.csv/png, exp5_noise.csv/png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PH = ["fibre", "matrix", "void"]
METH = {"cnn": "3D ResNet", "otsu": "Multi-Otsu count", "gmm": "GMM histogram"}


def mae_table(df, by):
    rows = []
    for key, g in df.groupby(by, observed=True):
        for m in METH:
            r = {by: key, "method": m, "n_crops": len(g), "n_parents": g.parent.nunique()}
            for ph in PH:
                r[f"MAE_{ph}_pp"] = (g[f"{m}_{ph}"] - g[f"true_{ph}"]).abs().mean()
            rows.append(r)
    return pd.DataFrame(rows)


def plot(tab, by, phases, path, xlabel, order=None):
    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 4))
    axes = np.atleast_1d(axes)
    xs = order or list(dict.fromkeys(tab[by]))
    for ax, ph in zip(axes, phases):
        for m, name in METH.items():
            t = tab[tab.method == m].set_index(by).reindex(xs)
            ax.plot([str(x) for x in xs], t[f"MAE_{ph}_pp"], marker="o", label=name)
        ax.set_title(f"{ph} fraction"); ax.set_xlabel(xlabel); ax.set_ylabel("MAE (percentage points)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--evals", nargs="+", required=True, help="eval folders at different crop sizes")
    ap.add_argument("--main", required=True, help="eval folder used for exps 3-5 (e.g. 64^3)")
    ap.add_argument("--out", default="experiments")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    parents = pd.read_csv(Path(a.data) / "parents.csv")

    # Exp 2: crop size
    frames = []
    for e in a.evals:
        d = pd.read_csv(Path(e) / "predictions.csv")
        d["crop"] = int(str(e).rstrip("/").split("_c")[-1])
        frames.append(d)
    t2 = mae_table(pd.concat(frames), "crop")
    t2.to_csv(out / "exp2_size.csv", index=False)
    plot(t2, "crop", PH, out / "exp2_size.png", "crop edge (voxels)", sorted(t2.crop.unique()))

    df = pd.read_csv(Path(a.main) / "predictions.csv").merge(
        parents[["id", "true_void", "true_fibre", "r_cv", "r_mean"]].rename(
            columns={"id": "parent", "true_void": "parent_void", "true_fibre": "parent_fibre"}),
        on="parent")

    # Exp 3: void level of the crop itself
    df["void_bin"] = pd.cut(df.true_void, [-0.01, 1, 3, 100], labels=["<1%", "1-3%", ">3%"])
    t3 = mae_table(df, "void_bin"); t3.to_csv(out / "exp3_void.csv", index=False)
    plot(t3, "void_bin", ["void"], out / "exp3_void.png", "true void content of crop",
         ["<1%", "1-3%", ">3%"])

    # Exp 4: packing density and fibre-diameter variability
    df["fibre_bin"] = pd.cut(df.true_fibre, [0, 50, 58, 100], labels=["<50%", "50-58%", ">58%"])
    t4a = mae_table(df, "fibre_bin")
    df["cv_bin"] = pd.cut(df.r_cv, [0, 0.085, 1], labels=["low CV", "high CV"])
    t4b = mae_table(df, "cv_bin")
    pd.concat([t4a.assign(factor="fibre fraction"), t4b.assign(factor="radius CV")]).to_csv(
        out / "exp4_packing.csv", index=False)
    plot(t4a, "fibre_bin", ["fibre", "void"], out / "exp4_packing.png", "true fibre fraction",
         ["<50%", "50-58%", ">58%"])

    # Exp 5: noise (dose -> measured CNR)
    cnr = parents[["cnr_high", "cnr_medium", "cnr_low"]].mean()
    df["noise"] = df.dose.map({d: f"{d} (CNR {cnr[f'cnr_{d}']:.1f})" for d in ["high", "medium", "low"]})
    order = [f"{d} (CNR {cnr[f'cnr_{d}']:.1f})" for d in ["high", "medium", "low"]]
    t5 = mae_table(df, "noise"); t5.to_csv(out / "exp5_noise.csv", index=False)
    plot(t5, "noise", PH, out / "exp5_noise.png", "dose", order)

    for name, t in [("Exp 2 size", t2), ("Exp 3 void", t3), ("Exp 4 packing", t4a), ("Exp 5 noise", t5)]:
        print(f"\n== {name} ==")
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
