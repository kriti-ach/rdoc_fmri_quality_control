from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass
class SdMetrics:
    nifti_path: str
    n_timepoints: int
    n_brain_voxels: int
    sd_mean: float
    sd_median: float
    sd_p95: float
    sd_p99: float
    sd_max: float
    mad: float
    robust_z99: float
    robust_zmax: float
    n_z_ge_6: int
    n_z_ge_8: int
    frac_z_ge_6: float
    frac_z_ge_8: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _brain_mask(data_4d: np.ndarray, min_mean: float = 1e-6) -> np.ndarray:
    mean_img = data_4d.mean(axis=3)
    return mean_img > min_mean


def temporal_sd_map(data_4d: np.ndarray) -> np.ndarray:
    return np.std(data_4d, axis=3, dtype=np.float64).astype(np.float32)


def robust_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    # Consistent with normal-scale robust z-score.
    scale = 1.4826 * mad if mad > 0 else 1.0
    return (values - med) / scale, med, mad


def compute_sd_metrics(nifti_path: str | Path, min_mean: float = 1e-6) -> tuple[SdMetrics, np.ndarray, np.ndarray]:
    path = Path(nifti_path)
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got shape={data.shape}")

    sd_map = temporal_sd_map(data)
    mask = _brain_mask(data, min_mean=min_mean)
    vals = sd_map[mask]
    if vals.size == 0:
        raise ValueError("Mask is empty. Check intensity scaling or min_mean.")

    z, _, mad = robust_z(vals)
    n6 = int(np.sum(z >= 6.0))
    n8 = int(np.sum(z >= 8.0))

    metrics = SdMetrics(
        nifti_path=str(path),
        n_timepoints=int(data.shape[3]),
        n_brain_voxels=int(vals.size),
        sd_mean=float(np.mean(vals)),
        sd_median=float(np.median(vals)),
        sd_p95=float(np.percentile(vals, 95)),
        sd_p99=float(np.percentile(vals, 99)),
        sd_max=float(np.max(vals)),
        mad=float(mad),
        robust_z99=float(np.percentile(z, 99)),
        robust_zmax=float(np.max(z)),
        n_z_ge_6=n6,
        n_z_ge_8=n8,
        frac_z_ge_6=float(n6 / vals.size),
        frac_z_ge_8=float(n8 / vals.size),
    )

    return metrics, sd_map, mask


def save_sd_nifti(sd_map: np.ndarray, source_nifti_path: str | Path, output_path: str | Path) -> None:
    src = nib.load(str(source_nifti_path))
    out = nib.Nifti1Image(sd_map.astype(np.float32), src.affine, src.header)
    out.header.set_data_shape(sd_map.shape)
    nib.save(out, str(output_path))


def save_mid_sagittal_png(sd_map: np.ndarray, title: str, output_png: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid_x = sd_map.shape[0] // 2
    plane = sd_map[mid_x, :, :]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.rot90(plane), cmap="viridis", aspect="auto")
    ax.set_title(f"Temporal SD (mid-sag x={mid_x})\\n{title}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)
