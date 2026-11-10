"""
regen_split.py  —  re-create selected splits of the full dataset (same master seed, same parameter
draws as make_dataset.py) so post-hoc analyses can run without the original Colab files.
Only the CT projector may differ (CPU 'linear' vs GPU 'cuda'); phantom geometry and noise
draws are identical because both are seeded.

Run: python regen_split.py --splits val test --out <dir>
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from phantom import generate_phantom
from ct_sim import simulate, DOSE, measure_cnr
from make_dataset import sample_params, counts_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n_ood", type=int, default=6)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--angles", type=int, default=360)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(np.arange(a.n))
    n_tr, n_va = int(0.70 * a.n), int(0.15 * a.n)
    split = {int(i): "train" for i in perm[:n_tr]}
    split |= {int(i): "val" for i in perm[n_tr:n_tr + n_va]}
    split |= {int(i): "test" for i in perm[n_tr + n_va:]}
    for j in range(a.n_ood):
        split[a.n + j] = "ood"
    csv = out / "parents.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {int(r["id"]) for r in rows}
    for pid in range(a.n + a.n_ood):
        p = sample_params(rng, pid, a.size, ood=(split[pid] == "ood"))
        if split[pid] not in a.splits or pid in done:
            continue
        t = time.time()
        ph = generate_phantom(p)
        np.save(out / f"p{pid:03d}_counts.npy", counts_field(ph["labels_fine"], p.size, p.ss))
        np.save(out / f"p{pid:03d}_labels.npy", ph["labels"])
        row = {"id": pid, "split": split[pid]} | {f"true_{k}": v for k, v in ph["fractions"].items()}
        row |= dict(ph["meta"])
        for d, I0 in DOSE.items():
            rec = simulate(ph["mu"], I0=I0, n_angles=a.angles, seed=p.seed + 1, use_gpu=False)
            np.save(out / f"p{pid:03d}_rec_{d}.npy", rec.astype(np.float16))
            row[f"cnr_{d}"] = measure_cnr(rec, ph["labels"])
        rows.append(row); rows.sort(key=lambda r: r["id"])
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"{pid} {split[pid]} fibre={row['true_fibre']:.4f} {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
