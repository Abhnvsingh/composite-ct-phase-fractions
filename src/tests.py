"""
tests.py  —  validation level 1: is the ground truth itself correct?  Run: python tests.py
Every check must print PASS before you trust any model result.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from phantom import PhantomParams, generate_phantom, subvolume_fractions, _pack_fibres
from make_dataset import counts_field


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main():
    ok = True
    p = PhantomParams(size=48, vf_target=0.6, void_target=0.03, seed=11)
    ph = generate_phantom(p)
    f = ph["fractions"]
    ok &= check("fractions sum to 1", abs(sum(f.values()) - 1) < 1e-12)
    ok &= check("void close to target (+-0.5 pp)", abs(f["void"] - 0.03) < 0.005)

    # crop ground truth from stored counts == direct recount on fine labels
    cnt = counts_field(ph["labels_fine"], p.size, p.ss)
    z, y, x, n = 5, 9, 13, 24
    blk = cnt[:, z:z+n, y:y+n, x:x+n].sum(axis=(1, 2, 3)) / (n**3 * p.ss**3)
    direct = subvolume_fractions(ph["labels_fine"], p.ss, z, y, x, n)
    ok &= check("crop GT from counts == direct recount",
                abs(blk[0] - direct["fibre"]) < 1e-12 and abs(blk[1] - direct["void"]) < 1e-12)

    # packing: no overlapping fibres (centre distance >= r_i + r_j)
    rng = np.random.default_rng(p.seed)
    xy, r, _ = _pack_fibres(PhantomParams(size=48, vf_target=0.65, seed=3), np.random.default_rng(3))
    D = squareform(pdist(xy)); R = r[:, None] + r[None, :]
    np.fill_diagonal(D, np.inf)
    ok &= check(f"no fibre overlaps at 65% target (worst overlap {max(0,(R-D).max()):.3f} vox)",
                (R - D).max() <= 1e-6)

    # zero-void phantom has exactly zero void
    ph0 = generate_phantom(PhantomParams(size=32, void_target=0.0, seed=5))
    ok &= check("void_target=0 gives exactly 0 void", ph0["fractions"]["void"] == 0.0)

    # same seed -> identical phantom (reproducibility)
    a = generate_phantom(PhantomParams(size=32, seed=9))["labels_fine"]
    b = generate_phantom(PhantomParams(size=32, seed=9))["labels_fine"]
    ok &= check("same seed reproduces the phantom exactly", np.array_equal(a, b))
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    if not ok:
        raise SystemExit(1)          # non-zero exit stops the Colab notebook here


if __name__ == "__main__":
    main()
