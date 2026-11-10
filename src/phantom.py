"""
phantom.py  —  Project 65: synthetic fibre-composite phantoms with exact phase-fraction ground truth.

What it builds
--------------
A 3D block of matrix (epoxy) containing parallel cylindrical fibres (with optional global
waviness along z) and spherical voids. Everything is drawn on a FINE grid (supersampling
factor `ss`), so:
  * ground-truth fractions are counted on the fine grid  -> near-analytic accuracy
  * the attenuation map is block-averaged down to the target grid -> a physically correct
    partial-volume effect (PVE) in the voxels that straddle a boundary

Labels:  0 = matrix, 1 = fibre, 2 = void

Packing method
--------------
Random sequential addition (RSA) alone jams at ~55% area fraction for disks, below the 65%
the project needs. So we use a simple force-biased relaxation (Lubachevsky-Stillinger /
Moscinski style, simplified): drop all N fibres at random, then repeatedly push overlapping
pairs apart until no overlaps remain (or an iteration cap is hit). Fibres may cross the
volume boundary, exactly as a crop from a large laminate would show.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from scipy.spatial import cKDTree

# Relative linear attenuation coefficients at a ~30-40 keV effective energy.
# For low-Z materials mu scales roughly with density: carbon fibre ~1.8 g/cc, epoxy ~1.2 g/cc.
# (Set fibre higher, e.g. 2.0, to mimic glass fibre with stronger contrast.)
MU = {"matrix": 0.80, "fibre": 1.00, "void": 0.0}


@dataclass
class PhantomParams:
    size: int = 96                 # target grid edge (voxels), volume is size^3
    ss: int = 2                    # supersampling factor for GT and PVE
    vf_target: float = 0.55        # fibre volume fraction target (0.40-0.65)
    void_target: float = 0.02      # void volume fraction target (0.00-0.05)
    r_mean: float = 4.0            # fibre radius, target-grid voxels (3-6)
    r_cv: float = 0.08             # radius coefficient of variation (diameter variability)
    gap: float = 0.15              # minimum fibre-fibre gap, voxels
    void_r_logmean: float = 0.6    # log-normal void radius: exp(0.6) ~ 1.8 voxels median
    void_r_logstd: float = 0.45
    waviness_amp: float = 0.0      # global in-plane waviness amplitude along z (voxels)
    seed: int = 0


def _pack_fibres(p: PhantomParams, rng: np.random.Generator, max_iter: int = 4000):
    """Return centres (N,2) and radii (N,) in target-grid voxel units."""
    L = p.size
    pad = p.r_mean * 1.5                      # let fibres straddle the boundary
    side = L + 2 * pad
    area_needed = p.vf_target * side * side
    radii = []
    while np.pi * np.sum(np.square(radii)) < area_needed:
        radii.append(max(1.0, rng.normal(p.r_mean, p.r_cv * p.r_mean)))
    radii = np.array(radii)
    xy = rng.uniform(-pad, L + pad, size=(len(radii), 2))
    rmax = radii.max()
    for it in range(max_iter):
        tree = cKDTree(xy)
        pairs = tree.query_pairs(2 * rmax + p.gap, output_type="ndarray")
        if len(pairs) == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        d = xy[j] - xy[i]
        dist = np.linalg.norm(d, axis=1) + 1e-9
        need = radii[i] + radii[j] + p.gap
        ov = need - dist
        m = ov > 0
        if not m.any():
            break
        i, j, d, dist, ov = i[m], j[m], d[m], dist[m], ov[m]
        push = (d / dist[:, None]) * (0.5 * ov[:, None] * 1.05)
        # random jitter breaks symmetric deadlocks
        push += rng.normal(0, 0.01, push.shape)
        np.add.at(xy, i, -push)
        np.add.at(xy, j, push)
        xy = np.clip(xy, -pad, L + pad)
    return xy, radii, it


def generate_phantom(p: PhantomParams):
    """Build one phantom.

    Returns dict with:
      labels_fine  (S,S,S) uint8   fine-grid labels, S = size*ss   (z, y, x)
      mu           (L,L,L) float32 block-averaged attenuation map on the target grid (has PVE)
      labels       (L,L,L) uint8   target-grid labels by majority vote (for the seg baseline)
      fractions    dict  exact fibre/matrix/void fractions from the fine grid
      meta         dict  parameters + packing diagnostics
    """
    rng = np.random.default_rng(p.seed)
    L, ss = p.size, p.ss
    S = L * ss
    xy, radii, iters = _pack_fibres(p, rng)

    # fine-grid coordinates of voxel centres, in target-voxel units
    c = (np.arange(S) + 0.5) / ss
    Y, X = np.meshgrid(c, c, indexing="ij")
    labels = np.zeros((S, S, S), np.uint8)

    # fibres: parallel to z, optional global sinusoidal waviness (all fibres shift together,
    # so the waviness never creates fibre-fibre intersections)
    phase = rng.uniform(0, 2 * np.pi)
    for zi in range(S):
        z = c[zi]
        dx = p.waviness_amp * np.sin(2 * np.pi * z / L + phase)
        sl = np.zeros((S, S), bool)
        for (x0, y0), r in zip(xy, radii):
            x0 += dx
            # bounding box on fine grid for speed
            xa, xb = int(max(0, (x0 - r) * ss)), int(min(S, (x0 + r) * ss + 2))
            ya, yb = int(max(0, (y0 - r) * ss)), int(min(S, (y0 + r) * ss + 2))
            if xa >= xb or ya >= yb:
                continue
            sub = (X[ya:yb, xa:xb] - x0) ** 2 + (Y[ya:yb, xa:xb] - y0) ** 2 <= r * r
            sl[ya:yb, xa:xb] |= sub
        labels[zi][sl] = 1

    # voids: spheres placed with centre in matrix; carve matrix voxels only
    # (voids in cured laminates sit in resin-rich regions between fibres)
    n_total = labels.size
    target_void_vox = p.void_target * n_total
    void_vox = 0
    voids = []
    zz = np.arange(S)
    attempts = 0
    while void_vox < target_void_vox and attempts < 20000:
        attempts += 1
        cz, cy, cx = rng.uniform(0, L, 3)
        iz, iy, ix = (int(cz * ss), int(cy * ss), int(cx * ss))
        if labels[iz, iy, ix] != 0:
            continue
        r = float(np.exp(rng.normal(p.void_r_logmean, p.void_r_logstd)))
        r = min(r, 6.0)
        za, zb = max(0, int((cz - r) * ss)), min(S, int((cz + r) * ss) + 2)
        ya, yb = max(0, int((cy - r) * ss)), min(S, int((cy + r) * ss) + 2)
        xa, xb = max(0, int((cx - r) * ss)), min(S, int((cx + r) * ss) + 2)
        Zs, Ys, Xs = np.meshgrid(c[za:zb], c[ya:yb], c[xa:xb], indexing="ij")
        ball = (Zs - cz) ** 2 + (Ys - cy) ** 2 + (Xs - cx) ** 2 <= r * r
        block = labels[za:zb, ya:yb, xa:xb]
        carve = ball & (block == 0)
        block[carve] = 2
        void_vox += int(carve.sum())
        voids.append((cz, cy, cx, r))

    counts = np.bincount(labels.ravel(), minlength=3).astype(float)
    fr = counts / counts.sum()
    fractions = {"matrix": fr[0], "fibre": fr[1], "void": fr[2]}

    # attenuation on fine grid -> block mean to target grid (exact PVE for a box detector)
    mu_lut = np.array([MU["matrix"], MU["fibre"], MU["void"]], np.float32)
    mu_fine = mu_lut[labels]
    mu = mu_fine.reshape(L, ss, L, ss, L, ss).mean(axis=(1, 3, 5)).astype(np.float32)

    # majority-vote coarse labels (only for a segmentation baseline/visual QA)
    onehot = np.stack([(labels == k) for k in range(3)]).reshape(3, L, ss, L, ss, L, ss)
    labels_coarse = onehot.sum(axis=(2, 4, 6)).argmax(0).astype(np.uint8)

    meta = asdict(p) | {"n_fibres": int(len(radii)), "pack_iters": int(iters),
                        "n_voids": len(voids)}
    return {"labels_fine": labels, "mu": mu, "labels": labels_coarse,
            "fractions": fractions, "meta": meta}


def subvolume_fractions(labels_fine: np.ndarray, ss: int, z0: int, y0: int, x0: int, n: int):
    """Exact fractions of a target-grid crop [z0:z0+n, y0:y0+n, x0:x0+n] from the fine labels.
    Use this when you crop 32^3 / 64^3 sub-volumes for the sub-volume-size experiment."""
    blk = labels_fine[z0*ss:(z0+n)*ss, y0*ss:(y0+n)*ss, x0*ss:(x0+n)*ss]
    cnt = np.bincount(blk.ravel(), minlength=3).astype(float)
    cnt /= cnt.sum()
    return {"matrix": cnt[0], "fibre": cnt[1], "void": cnt[2]}


if __name__ == "__main__":
    import time
    t = time.time()
    out = generate_phantom(PhantomParams(seed=1))
    print({k: round(v, 4) for k, v in out["fractions"].items()}, out["meta"]["n_fibres"],
          out["meta"]["pack_iters"], f"{time.time()-t:.1f}s")
