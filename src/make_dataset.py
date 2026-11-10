"""
make_dataset.py  —  build the full Project 65 dataset, reproducibly from a single seed.

Design (matches the project sheet: "200 (sub-)volumes, or crops from ~50 larger phantoms")
------------------------------------------------------------------------------------------
* N parent phantoms (default 60) of 128^3, parameters sampled per phantom:
      fibre fraction target U(0.40, 0.66), void target U(0, 0.05),
      fibre radius U(3, 6) voxels, radius CV U(0.02, 0.15), waviness U(0, 2) voxels
* each parent is scanned at 3 doses (high/medium/low CNR)
* for EXACT ground truth of ANY crop we store per-voxel counts of fine fibre/void sub-voxels
  (uint8, 0..ss^3). The fraction of a crop = sum(counts in crop) / (n_voxels * ss^3). Exact.
* split BY PARENT (70/15/15) so overlapping crops never leak across train/val/test
* plus an out-of-distribution (OOD) set: fibre radius 6.5-7.5 voxels (never seen in training)

Output layout
-------------
data/
  parents.csv                 one row per parent: id, split, all parameters, true fractions
  p{id:03d}_rec_{dose}.npy    float16 reconstructions (128^3)
  p{id:03d}_counts.npy        uint8 (2,128,128,128): fine-voxel counts of fibre, void
  p{id:03d}_labels.npy        uint8 coarse majority labels (for visual QA / seg baselines)

Run from Project65_Colab.ipynb (recommended), or:
      python make_dataset.py --n 60 --size 128 --out ../data     (uses the Colab GPU automatically)
      python make_dataset.py --n 4 --size 64 --out ../data_smoke (quick smoke test)
Resumable: rerun the same command after a disconnect; finished parents are skipped.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from phantom import PhantomParams, generate_phantom
from ct_sim import simulate, DOSE, measure_cnr, gpu_available


def sample_params(rng, pid, size, ood=False):
    return PhantomParams(
        size=size, ss=2, seed=int(rng.integers(1 << 31)),
        vf_target=float(rng.uniform(0.40, 0.66)),
        void_target=float(rng.uniform(0.0, 0.05)),
        r_mean=float(rng.uniform(6.5, 7.5) if ood else rng.uniform(3.0, 6.0)),
        r_cv=float(rng.uniform(0.02, 0.15)),
        waviness_amp=float(rng.uniform(0.0, 2.0)),
    )


def counts_field(labels_fine, L, ss):
    blk = labels_fine.reshape(L, ss, L, ss, L, ss)
    fib = (blk == 1).sum(axis=(1, 3, 5)).astype(np.uint8)
    voi = (blk == 2).sum(axis=(1, 3, 5)).astype(np.uint8)
    return np.stack([fib, voi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n_ood", type=int, default=6)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--angles", type=int, default=360)
    ap.add_argument("--backend", default="parallel2d", choices=["parallel2d", "cone3d"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="../data")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    ids = np.arange(a.n)
    perm = rng.permutation(ids)
    n_tr, n_va = int(0.70 * a.n), int(0.15 * a.n)
    split = {int(i): "train" for i in perm[:n_tr]}
    split |= {int(i): "val" for i in perm[n_tr:n_tr + n_va]}
    split |= {int(i): "test" for i in perm[n_tr + n_va:]}
    for j in range(a.n_ood):
        split[a.n + j] = "ood"

    # Resumable: Colab sessions disconnect. Parents already listed in parents.csv are skipped.
    # Parameters are still drawn for skipped parents, so the random sequence (and every
    # phantom) is identical whether the run was interrupted or not.
    device = "gpu" if (a.backend == "cone3d" or gpu_available()) else "cpu"
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        for key in ("n", "n_ood", "size", "angles", "backend", "seed", "device"):
            new_val = device if key == "device" else getattr(a, key)
            if old.get(key) != new_val:
                raise SystemExit(f"Existing dataset in {out} was built with {key}={old.get(key)}, "
                                 f"now {new_val}. Use a new --out folder or the same settings.")
    cfg_path.write_text(json.dumps(vars(a) | {"dose_I0": DOSE, "device": device}, indent=2))
    print(f"CT simulation device: {device}", flush=True)
    csv_path = out / "parents.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {int(r["id"]) for r in rows}
    if done:
        print(f"resuming: {len(done)} parents already built", flush=True)

    for pid in range(a.n + a.n_ood):
        t = time.time()
        p = sample_params(rng, pid, a.size, ood=(split[pid] == "ood"))
        if pid in done:
            continue
        ph = generate_phantom(p)
        np.save(out / f"p{pid:03d}_counts.npy", counts_field(ph["labels_fine"], p.size, p.ss))
        np.save(out / f"p{pid:03d}_labels.npy", ph["labels"])
        row = {"id": pid, "split": split[pid]} | {f"true_{k}": v for k, v in ph["fractions"].items()}
        row |= {k: v for k, v in ph["meta"].items()}
        for d, I0 in DOSE.items():
            rec = simulate(ph["mu"], backend=a.backend, I0=I0, n_angles=a.angles, seed=p.seed + 1)
            np.save(out / f"p{pid:03d}_rec_{d}.npy", rec.astype(np.float16))
            row[f"cnr_{d}"] = measure_cnr(rec, ph["labels"])
        rows.append(row)
        print(f"[{pid+1}/{a.n + a.n_ood}] {split[pid]:5s} fibre={row['true_fibre']:.3f} "
              f"void={row['true_void']:.4f} cnr(h/m/l)={row['cnr_high']:.1f}/"
              f"{row['cnr_medium']:.1f}/{row['cnr_low']:.1f}  {time.time()-t:.0f}s", flush=True)
        rows.sort(key=lambda r: r["id"])
        pd.DataFrame(rows).to_csv(csv_path, index=False)   # written only after all files exist
    print(f"done: {len(rows)} parents in {out}", flush=True)


if __name__ == "__main__":
    main()
