"""
baselines.py  —  the classical methods the CNN must beat.

1. Multi-Otsu segmentation + voxel counting (with connected-component void filtering)
2. Gaussian-mixture fit to the intensity histogram, integrating each component's weight

Both take a reconstructed CT (sub-)volume and return {'fibre','matrix','void'} fractions.
"""
from __future__ import annotations
import numpy as np
from skimage.filters import threshold_multiotsu
from scipy import ndimage
from sklearn.mixture import GaussianMixture


def _drop_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components with fewer than `min_size` voxels (face connectivity, same
    as skimage's remove_small_objects default). Written with scipy so it behaves the same on
    every scikit-image version (that function's API changed in scikit-image 0.26)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def otsu_count(rec: np.ndarray, min_void_vox: int = 4, smooth: float = 0.0):
    """3-class multi-Otsu: darkest = void, middle = matrix, brightest = fibre.
    Void components smaller than `min_void_vox` are treated as noise and given back to matrix.
    Returns fractions and the label map (0 matrix, 1 fibre, 2 void)."""
    x = rec
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        x = gaussian_filter(rec, smooth)
    t1, t2 = threshold_multiotsu(x, classes=3)
    lab = np.zeros(x.shape, np.uint8)          # matrix
    lab[x >= t2] = 1                           # fibre
    void = x < t1
    if min_void_vox > 1:
        void = _drop_small(void, min_void_vox)
    lab[void] = 2
    cnt = np.bincount(lab.ravel(), minlength=3).astype(float)
    cnt /= cnt.sum()
    return {"matrix": cnt[0], "fibre": cnt[1], "void": cnt[2]}, lab


def gmm_fractions(rec: np.ndarray, n_sample: int = 200_000, seed: int = 0):
    """Fit a 3-component 1D Gaussian mixture to voxel intensities; component weights =
    phase fractions. Components sorted by mean: void < matrix < fibre.
    Known weakness: with 0-1% voids the void component is poorly identified."""
    rng = np.random.default_rng(seed)
    v = rec.ravel()
    if v.size > n_sample:
        v = rng.choice(v, n_sample, replace=False)
    # Initialisation matters: a free 1D GMM collapses matrix+fibre into one component because
    # partial-volume voxels fill the gap between the two peaks. Start void near 0 and the two
    # solid phases at the 25th/75th intensity percentiles; 'tied' = one shared variance,
    # which is physically right when noise (not material) sets the peak widths.
    m_lo, m_hi = np.percentile(v, [25, 75])
    gm = GaussianMixture(3, covariance_type="tied", random_state=seed, max_iter=500,
                         means_init=np.array([[0.05], [m_lo], [m_hi]]),
                         weights_init=np.array([0.02, 0.49, 0.49])).fit(v[:, None])
    order = np.argsort(gm.means_.ravel())
    w = gm.weights_[order]
    return {"void": w[0], "matrix": w[1], "fibre": w[2]}
