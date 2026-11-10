"""
train.py  —  train the 3D ResNet phase-fraction regressor.

Key choices (each is something to state in the Methods section):
* crops are drawn ONLY from parents in the requested split (no leakage)
* ground truth of every random crop is exact (from the stored fine-voxel counts)
* input normalisation: per-crop z-score (robust to dose/contrast differences between scans)
* augmentation: random 90-degree rotations/flips (fractions are invariant to these)
* training mixes all three doses unless --dose is given
* model selection: lowest validation MAE (mean over the 3 phases), early stopping

Run:  python train.py --data ../data --crop 64 --epochs 120 --out ../runs/c64
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import PhaseFractionResNet3D, fraction_loss

DOSES = ("high", "medium", "low")


class CropDataset(Dataset):
    """Random (train) or fixed-grid (eval) cubic crops with exact [fibre, matrix, void] targets."""

    def __init__(self, data_dir, split, crop=64, doses=DOSES, n_random=2000, seed=0,
                 grid=False, ss=2):
        self.dir = Path(data_dir)
        meta = pd.read_csv(self.dir / "parents.csv")
        self.ids = meta.loc[meta.split == split, "id"].tolist()
        self.crop, self.doses, self.grid, self.ss3 = crop, doses, grid, ss ** 3
        self.rec = {(i, d): np.load(self.dir / f"p{i:03d}_rec_{d}.npy", mmap_mode="r")
                    for i in self.ids for d in doses}
        self.cnt = {i: np.load(self.dir / f"p{i:03d}_counts.npy", mmap_mode="r") for i in self.ids}
        L = self.cnt[self.ids[0]].shape[-1]
        self.seed, self.epoch = seed, 0
        if grid:   # non-overlapping tiling: every voxel used once -> unbiased test metrics
            starts = range(0, L - crop + 1, crop)
            self.items = [(i, d, z, y, x) for i in self.ids for d in doses
                          for z in starts for y in starts for x in starts]
        else:
            self.n_random, self.L = n_random, L

    def set_epoch(self, epoch):
        """Call before each epoch. Each crop's randomness is seeded by (seed, epoch, index), so
        every epoch sees new crops, DataLoader workers never repeat each other, and a resumed
        run continues exactly as an uninterrupted one would."""
        self.epoch = epoch

    def __len__(self):
        return len(self.items) if self.grid else self.n_random

    def target(self, i, z, y, x):
        c = self.crop
        blk = self.cnt[i][:, z:z + c, y:y + c, x:x + c]
        tot = c ** 3 * self.ss3
        fib = blk[0].sum(dtype=np.int64) / tot
        voi = blk[1].sum(dtype=np.int64) / tot
        return np.array([fib, 1.0 - fib - voi, voi], np.float32)

    def __getitem__(self, k):
        c = self.crop
        if self.grid:
            i, d, z, y, x = self.items[k]
        else:
            r = np.random.default_rng((self.seed, self.epoch, k))
            i = self.ids[r.integers(len(self.ids))]
            d = self.doses[r.integers(len(self.doses))]
            z, y, x = r.integers(0, self.L - c + 1, 3)
        v = np.asarray(self.rec[(i, d)][z:z + c, y:y + c, x:x + c], np.float32)
        if not self.grid:      # augmentation that leaves fractions unchanged
            v = np.rot90(v, r.integers(4), axes=(1, 2))
            if r.random() < 0.5: v = v[::-1]
            if r.random() < 0.5: v = v[:, ::-1]
        v = (v - v.mean()) / (v.std() + 1e-6)
        meta = np.array([i, DOSES.index(d), z, y, x])
        return torch.from_numpy(v.copy())[None], torch.from_numpy(self.target(i, z, y, x)), meta


def evaluate(model, loader, device):
    model.eval(); P, T, M = [], [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                         enabled=(device == "cuda")):
        for x, t, m in loader:
            P.append(model(x.to(device)).float().cpu().numpy()); T.append(t.numpy()); M.append(m.numpy())
    return np.concatenate(P), np.concatenate(T), np.concatenate(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../runs/c64")
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps", type=int, default=400, help="random crops per epoch")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_void", type=float, default=10.0)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--dose", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--shuffle_labels", action="store_true",
                    help="leakage test: pair each crop with another crop's target; val MAE must "
                         "stay at the predict-the-mean level")
    ap.add_argument("--fresh", action="store_true", help="ignore last.pt and start over")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    doses = DOSES if a.dose == "all" else (a.dose,)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = CropDataset(a.data, "train", a.crop, doses, n_random=a.steps, seed=a.seed)
    va = CropDataset(a.data, "val", a.crop, doses, grid=True)
    pin = dev == "cuda"
    tl = DataLoader(tr, a.bs, shuffle=False, num_workers=a.workers, drop_last=True, pin_memory=pin)
    vl = DataLoader(va, a.bs, shuffle=False, num_workers=a.workers, pin_memory=pin)

    model = PhaseFractionResNet3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    amp = dev == "cuda"                      # mixed precision: ~2x faster on Colab T4
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if amp:
        torch.backends.cudnn.benchmark = True
    w = (1.0, 1.0, a.w_void)
    best, bad, log, start = 1e9, 0, [], 0

    # Resume after a Colab disconnect: last.pt holds everything needed to continue.
    last = out / "last.pt"
    if last.exists() and not a.fresh:
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        best, bad, log, start = ck["best"], ck["bad"], ck["log"], ck["epoch"] + 1
        print(f"resumed from epoch {ck['epoch']} (best val MAE {best:.3f})", flush=True)
        if bad >= a.patience or start >= a.epochs:
            print("training already finished"); return

    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); tl_sum = 0
        tr.set_epoch(ep)
        for x, t, _ in tl:
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            if a.shuffle_labels:
                t = t[torch.randperm(len(t), device=t.device)]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                pred = model(x)
            loss = fraction_loss(pred.float(), t, w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl_sum += loss.item()
        sched.step()
        P, T, _ = evaluate(model, vl, dev)
        mae = np.abs(P - T).mean(0) * 100          # percentage points
        score = mae.mean()
        log.append({"epoch": ep, "train_loss": tl_sum / len(tl), "val_mae_fibre": mae[0],
                    "val_mae_matrix": mae[1], "val_mae_void": mae[2], "sec": time.time() - t0})
        print(f"ep {ep:3d} loss {log[-1]['train_loss']:.5f} val MAE (pp) fibre {mae[0]:.2f} "
              f"matrix {mae[1]:.2f} void {mae[2]:.3f}  {log[-1]['sec']:.0f}s", flush=True)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        if score < best:
            best, bad = score, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep,
                    "best": best, "bad": bad, "log": log}, out / "last.tmp")
        (out / "last.tmp").replace(last)     # atomic: a disconnect never leaves a half file
        if bad >= a.patience:
            print("early stop"); break
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))


if __name__ == "__main__":
    main()
