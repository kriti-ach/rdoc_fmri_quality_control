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
    sd_p999: float
    sd_max: float
    mad: float
    robust_z99: float
    robust_zmax: float
    n_z_ge_3: int
    n_z_ge_5: int
    n_z_ge_6: int
    n_z_ge_8: int
    n_z_ge_10: int
    frac_z_ge_3: float
    frac_z_ge_5: float
    frac_z_ge_6: float
    frac_z_ge_8: float
    frac_z_ge_10: float

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
    
    # Multiple thresholds for better characterization
    n3 = int(np.sum(z >= 3.0))
    n5 = int(np.sum(z >= 5.0))
    n6 = int(np.sum(z >= 6.0))
    n8 = int(np.sum(z >= 8.0))
    n10 = int(np.sum(z >= 10.0))

    metrics = SdMetrics(
        nifti_path=str(path),
        n_timepoints=int(data.shape[3]),
        n_brain_voxels=int(vals.size),
        sd_mean=float(np.mean(vals)),
        sd_median=float(np.median(vals)),
        sd_p95=float(np.percentile(vals, 95)),
        sd_p99=float(np.percentile(vals, 99)),
        sd_p999=float(np.percentile(vals, 99.9)),  # Add this
        sd_max=float(np.max(vals)),
        mad=float(mad),
        robust_z99=float(np.percentile(z, 99)),
        robust_zmax=float(np.max(z)),
        n_z_ge_3=n3,  # Add
        n_z_ge_5=n5,  # Add
        n_z_ge_6=n6,
        n_z_ge_8=n8,
        n_z_ge_10=n10,  # Add
        frac_z_ge_3=float(n3 / vals.size),  # Add
        frac_z_ge_5=float(n5 / vals.size),  # Add
        frac_z_ge_6=float(n6 / vals.size),
        frac_z_ge_8=float(n8 / vals.size),
        frac_z_ge_10=float(n10 / vals.size),  # Add
    )

    return metrics, sd_map, mask


def save_sd_nifti(sd_map: np.ndarray, source_nifti_path: str | Path, output_path: str | Path) -> None:
    src = nib.load(str(source_nifti_path))
    out = nib.Nifti1Image(sd_map.astype(np.float32), src.affine, src.header)
    out.header.set_data_shape(sd_map.shape)
    nib.save(out, str(output_path))

def detect_central_line_artifact(
    sd_map: np.ndarray,
    mask: np.ndarray,
    z_threshold: float = 8.0,
    center_width: int = 5,
) -> dict[str, float | bool]:
    """
    Detect if high-SD outliers are concentrated in central sagittal slices.
    
    Parameters
    ----------
    sd_map : np.ndarray
        3D temporal SD map
    mask : np.ndarray
        Brain mask
    z_threshold : float
        Z-score threshold for defining outliers
    center_width : int
        Number of slices around center to check (total width)
        
    Returns
    -------
    dict with central artifact metrics
    """
    vals = sd_map[mask]
    median_sd = np.median(vals)
    mad = np.median(np.abs(vals - median_sd))
    robust_std = 1.4826 * mad
    
    # Compute z-scores
    z_map = (sd_map - median_sd) / (robust_std + 1e-10)
    
    # Outlier mask
    outlier_mask = (z_map >= z_threshold) & mask
    total_outliers = int(np.sum(outlier_mask))
    
    if total_outliers == 0:
        return {
            "central_artifact_flag": False,
            "central_concentration": 0.0,
            "outliers_in_center": 0,
            "outliers_total": 0,
        }
    
    # Check sagittal center slices (assuming standard orientation)
    # Adjust axis if your data has different orientation
    nx = sd_map.shape[0]
    center_idx = nx // 2
    half_width = center_width // 2
    
    center_slice = slice(
        max(0, center_idx - half_width),
        min(nx, center_idx + half_width + 1)
    )
    
    outliers_in_center = int(np.sum(outlier_mask[center_slice, :, :]))
    concentration = outliers_in_center / total_outliers
    
    # Flag if >50% of outliers are in center slices
    has_artifact = concentration > 0.5
    
    return {
        "central_artifact_flag": bool(has_artifact),
        "central_concentration": float(concentration),
        "outliers_in_center": outliers_in_center,
        "outliers_total": total_outliers,
    }

import matplotlib.pyplot as plt

def visualize_outliers(tsd_map, mask, z_threshold=5, save_path=None):
    """
    Create visualization similar to MRIQC report.
    """
    median_tsd = np.median(tsd_map[mask])
    mad = np.median(np.abs(tsd_map[mask] - median_tsd))
    robust_std = 1.4826 * mad
    z_scores = (tsd_map - median_tsd) / robust_std
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sagittal views
    slice_idx = tsd_map.shape[0] // 2
    axes[0, 0].imshow(tsd_map[slice_idx, :, :].T, cmap='hot', origin='lower')
    axes[0, 0].set_title('TSD - Sagittal')
    
    axes[1, 0].imshow((z_scores[slice_idx, :, :] > z_threshold).T, 
                      cmap='Reds', origin='lower')
    axes[1, 0].set_title(f'Outliers (>{z_threshold}SD) - Sagittal')
    
    # Coronal views
    slice_idx = tsd_map.shape[1] // 2
    axes[0, 1].imshow(tsd_map[:, slice_idx, :].T, cmap='hot', origin='lower')
    axes[0, 1].set_title('TSD - Coronal')
    
    axes[1, 1].imshow((z_scores[:, slice_idx, :] > z_threshold).T, 
                      cmap='Reds', origin='lower')
    axes[1, 1].set_title(f'Outliers - Coronal')
    
    # Axial views
    slice_idx = tsd_map.shape[2] // 2
    axes[0, 2].imshow(tsd_map[:, :, slice_idx].T, cmap='hot', origin='lower')
    axes[0, 2].set_title('TSD - Axial')
    
    axes[1, 2].imshow((z_scores[:, :, slice_idx] > z_threshold).T, 
                      cmap='Reds', origin='lower')
    axes[1, 2].set_title(f'Outliers - Axial')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig

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
