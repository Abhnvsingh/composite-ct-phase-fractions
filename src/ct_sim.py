"""
ct_sim.py  —  simulate a CT scan of a phantom: forward projection -> Poisson noise -> detector
blur -> reconstruction.

Two backends
------------
'parallel2d' (default)        slice-by-slice parallel-beam FBP. Uses ASTRA's CUDA projector +
                              FBP_CUDA automatically on a Colab GPU runtime, CPU otherwise.
                              A standard approximation of a micro-CT with small cone angle.
                              Keep ONE device for a whole dataset (make_dataset.py records it):
                              CPU 'linear' and GPU 'cuda' projectors differ slightly.
'cone3d'     (NVIDIA GPU)     true cone-beam forward projection + FDK, as the project sheet asks.
                              Needs ASTRA built with CUDA (Colab/Kaggle GPU: `pip install astra-toolbox`).

Use 'parallel2d' to develop and debug; switch to 'cone3d' for the final dataset and say so in
the report (and optionally compare the two as a robustness check).

Physics
-------
line integral p = sum(mu * step) ; detected counts N ~ Poisson(I0 * exp(-p)) ;
p_noisy = -ln(N / I0). I0 (photons per detector pixel) is the dose / SNR knob.
"""
from __future__ import annotations
import numpy as np
import astra
from scipy.ndimage import gaussian_filter1d

# scale so that the longest path through a 96-voxel block of average material
# gives p ~ 2.5 (about 8% transmission), a typical well-set micro-CT exposure
def _mu_scale(size: int, mean_mu: float = 0.8, p_max: float = 2.5) -> float:
    return p_max / (size * mean_mu)


def _noisy(sino: np.ndarray, I0: float | None, rng: np.random.Generator) -> np.ndarray:
    if I0 is None:                     # noiseless
        return sino
    counts = rng.poisson(I0 * np.exp(-sino)).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid log(0) (photon starvation)
    return -np.log(counts / I0)


_GPU_OK = None


def gpu_available() -> bool:
    """True if ASTRA can run its CUDA projector here (e.g. a Colab GPU runtime). Probed once with
    a tiny real projection, so a CUDA build without a working GPU falls back to CPU cleanly."""
    global _GPU_OK
    if _GPU_OK is None:
        try:
            vg = astra.create_vol_geom(8, 8)
            pg = astra.create_proj_geom("parallel", 1.0, 12, np.linspace(0, np.pi, 4, endpoint=False))
            pid = astra.create_projector("cuda", pg, vg)
            sid, _ = astra.create_sino(np.ones((8, 8), np.float32), pid)
            astra.data2d.delete(sid); astra.projector.delete(pid)
            _GPU_OK = True
        except Exception:
            _GPU_OK = False
    return _GPU_OK


def simulate_parallel2d(mu: np.ndarray, n_angles: int = 360, I0: float | None = 2e4,
                        blur_sigma: float = 0.7, seed: int = 0, filt: str = "ram-lak",
                        use_gpu: bool | None = None):
    """mu: (Z,Y,X) target-grid attenuation. Returns reconstruction in the same relative units.
    use_gpu=None picks the GPU automatically when available (Colab GPU runtime)."""
    if use_gpu is None:
        use_gpu = gpu_available()
    rng = np.random.default_rng(seed)
    Z, Ny, Nx = mu.shape
    # Embed the block in a cylindrical specimen: a real scan is a crop from INSIDE a larger
    # part, so the block must not have hard square edges (they cause diagonal streaks).
    # Padding = homogenised composite (block mean); outside the cylinder = air.
    P = int(np.ceil(np.sqrt(2) * max(Ny, Nx))) + 6
    P += P % 2
    oy, ox = (P - Ny) // 2, (P - Nx) // 2
    yy, xx = np.mgrid[:P, :P] - (P - 1) / 2
    cyl = (yy ** 2 + xx ** 2) <= (P / 2 - 1) ** 2
    fill = float(mu.mean())
    k = _mu_scale(P)
    n_det = P + 4
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    vol_geom = astra.create_vol_geom(P, P)
    proj_geom = astra.create_proj_geom("parallel", 1.0, n_det, angles)
    proj_id = astra.create_projector("cuda" if use_gpu else "linear", proj_geom, vol_geom)
    rec = np.empty_like(mu, dtype=np.float32)
    for z in range(Z):
        big = np.where(cyl, fill, 0.0).astype(np.float32)
        big[oy:oy + Ny, ox:ox + Nx] = mu[z]
        sid, sino = astra.create_sino(big * k, proj_id)
        astra.data2d.delete(sid)
        sino = _noisy(sino, I0, rng)
        if blur_sigma > 0:            # detector point-spread (along detector axis)
            sino = gaussian_filter1d(sino, blur_sigma, axis=1)
        sino_id = astra.data2d.create("-sino", proj_geom, sino)
        rec_id = astra.data2d.create("-vol", vol_geom)
        cfg = astra.astra_dict("FBP_CUDA" if use_gpu else "FBP")
        cfg["ProjectorId"] = proj_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = rec_id
        cfg["FilterType"] = filt
        alg = astra.algorithm.create(cfg)
        astra.algorithm.run(alg)
        rec[z] = astra.data2d.get(rec_id)[oy:oy + Ny, ox:ox + Nx] / k
        astra.algorithm.delete(alg)
        astra.data2d.delete([sino_id, rec_id])
    astra.projector.delete(proj_id)
    return rec


def simulate_cone3d(mu: np.ndarray, n_angles: int = 360, I0: float | None = 2e4,
                    blur_sigma: float = 0.7, seed: int = 0, sod: float = 500.0,
                    odd: float = 250.0):
    """Cone-beam CT + FDK reconstruction (requires CUDA). sod/odd = source-object /
    object-detector distance in voxel units (magnification = (sod+odd)/sod)."""
    if not astra.use_cuda():
        raise RuntimeError("cone3d needs ASTRA with CUDA (a GPU runtime). Use 'parallel2d'.")
    rng = np.random.default_rng(seed)
    Z, Ny, Nx = mu.shape
    k = _mu_scale(Nx)
    mag = (sod + odd) / sod
    det_cols = int(np.ceil(np.sqrt(2) * Nx * mag)) + 4
    det_rows = int(np.ceil(Z * mag)) + 4
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    vol_geom = astra.create_vol_geom(Ny, Nx, Z)
    proj_geom = astra.create_proj_geom("cone", 1.0, 1.0, det_rows, det_cols, angles, sod, odd)
    pid, proj = astra.create_sino3d_gpu(mu * k, proj_geom, vol_geom)  # (rows, angles, cols)
    astra.data3d.delete(pid)
    proj = _noisy(proj, I0, rng)
    if blur_sigma > 0:
        proj = gaussian_filter1d(gaussian_filter1d(proj, blur_sigma, axis=2), blur_sigma, axis=0)
    proj_id = astra.data3d.create("-proj3d", proj_geom, proj)
    rec_id = astra.data3d.create("-vol", vol_geom)
    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ProjectionDataId"] = proj_id
    cfg["ReconstructionDataId"] = rec_id
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    rec = astra.data3d.get(rec_id) / k
    astra.algorithm.delete(alg)
    astra.data3d.delete([proj_id, rec_id])
    return rec.astype(np.float32)


def simulate(mu, backend: str = "parallel2d", **kw):
    if backend == "parallel2d":
        return simulate_parallel2d(mu, **kw)
    if backend == "cone3d":
        return simulate_cone3d(mu, **kw)
    raise ValueError(backend)


# three dose levels for the SNR experiment (photons per detector pixel). Calibrated so the
# fibre-matrix contrast-to-noise ratio (CNR) is roughly 6 / 3 / 1.5 with the default MU.
# Report the MEASURED CNR (see measure_cnr) in the paper, not just I0.
DOSE = {"high": 1e4, "medium": 2e3, "low": 4e2}


def measure_cnr(rec: np.ndarray, labels: np.ndarray) -> float:
    """Fibre-matrix contrast-to-noise ratio, using interior voxels only (eroded masks) so
    partial-volume edges do not inflate the 'noise'."""
    from scipy.ndimage import binary_erosion
    f = binary_erosion(labels == 1, iterations=1)
    m = binary_erosion(labels == 0, iterations=1)
    return float((rec[f].mean() - rec[m].mean()) / np.sqrt(0.5 * (rec[f].var() + rec[m].var())))
