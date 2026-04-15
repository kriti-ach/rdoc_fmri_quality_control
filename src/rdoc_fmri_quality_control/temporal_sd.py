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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_outliers(tsd_map, mask, z_threshold=5, save_path=None, center_width=5, artifact_info=None):
    """
    Create visualization showing outliers and center detection region.
    
    Parameters
    ----------
    artifact_info : dict, optional
        Output from detect_central_line_artifact() to display on plot
    """
    median_tsd = np.median(tsd_map[mask])
    mad = np.median(np.abs(tsd_map[mask] - median_tsd))
    robust_std = 1.4826 * mad
    z_scores = (tsd_map - median_tsd) / robust_std
    
    outlier_mask = (z_scores > z_threshold) & mask
    
    # Center region for sagittal (x-axis)
    nx = tsd_map.shape[0]
    center_idx = nx // 2
    half_width = center_width // 2
    center_start = max(0, center_idx - half_width)
    center_end = min(nx, center_idx + half_width + 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Row 1: TSD maps with center region marked
    # Sagittal (center slice - this IS the center region being checked)
    slice_idx = center_idx
    im0 = axes[0, 0].imshow(tsd_map[slice_idx, :, :].T, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(tsd_map[mask], 99))
    axes[0, 0].set_title(f'TSD - Sagittal (slice {slice_idx})\n**THIS IS CENTER SLICE**\nDetection checks slices {center_start}-{center_end}')
    axes[0, 0].set_xlabel('Y (posterior ← → anterior)')
    axes[0, 0].set_ylabel('Z (inferior ← → superior)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label='TSD')
    
    # Coronal - mark center region as vertical band
    slice_idx = tsd_map.shape[1] // 2
    im1 = axes[0, 1].imshow(tsd_map[:, slice_idx, :].T, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(tsd_map[mask], 99))
    # Add shaded region showing which X slices are "center"
    axes[0, 1].axvspan(center_start, center_end, alpha=0.2, color='red', label=f'Center region\n(X slices {center_start}-{center_end})')
    axes[0, 1].set_title(f'TSD - Coronal (slice {slice_idx})')
    axes[0, 1].set_xlabel('X (left ← → right)')
    axes[0, 1].set_ylabel('Z (inferior ← → superior)')
    axes[0, 1].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='TSD')
    
    # Axial - mark center region as vertical band
    slice_idx = tsd_map.shape[2] // 2
    im2 = axes[0, 2].imshow(tsd_map[:, :, slice_idx].T, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(tsd_map[mask], 99))
    axes[0, 2].axvspan(center_start, center_end, alpha=0.2, color='red', label=f'Center region\n(X slices {center_start}-{center_end})')
    axes[0, 2].set_title(f'TSD - Axial (slice {slice_idx})')
    axes[0, 2].set_xlabel('X (left ← → right)')
    axes[0, 2].set_ylabel('Y (posterior ← → anterior)')
    axes[0, 2].legend(loc='upper right')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='TSD')
    
    # Row 2: Outlier masks overlaid on anatomy
    # Sagittal - show outliers on this center slice
    slice_idx = center_idx
    axes[1, 0].imshow(tsd_map[slice_idx, :, :].T, cmap='gray', origin='lower', alpha=0.5)
    outlier_overlay = np.ma.masked_where(~outlier_mask[slice_idx, :, :].T, outlier_mask[slice_idx, :, :].T)
    axes[1, 0].imshow(outlier_overlay, cmap='Reds', origin='lower', alpha=0.8)
    axes[1, 0].set_title(f'Outliers (z>{z_threshold}) - Sagittal\n**CENTER SLICE** - outliers HERE count as "in center"')
    axes[1, 0].set_xlabel('Y (posterior ← → anterior)')
    axes[1, 0].set_ylabel('Z (inferior ← → superior)')
    
    # Coronal - mark center region
    slice_idx = tsd_map.shape[1] // 2
    axes[1, 1].imshow(tsd_map[:, slice_idx, :].T, cmap='gray', origin='lower', alpha=0.5)
    outlier_overlay = np.ma.masked_where(~outlier_mask[:, slice_idx, :].T, outlier_mask[:, slice_idx, :].T)
    axes[1, 1].imshow(outlier_overlay, cmap='Reds', origin='lower', alpha=0.8)
    axes[1, 1].axvspan(center_start, center_end, alpha=0.2, color='red')
    axes[1, 1].set_title(f'Outliers - Coronal\nRed band = center region')
    axes[1, 1].set_xlabel('X (left ← → right)')
    axes[1, 1].set_ylabel('Z (inferior ← → superior)')
    
    # Axial - mark center region
    slice_idx = tsd_map.shape[2] // 2
    axes[1, 2].imshow(tsd_map[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.5)
    outlier_overlay = np.ma.masked_where(~outlier_mask[:, :, slice_idx].T, outlier_mask[:, :, slice_idx].T)
    axes[1, 2].imshow(outlier_overlay, cmap='Reds', origin='lower', alpha=0.8)
    axes[1, 2].axvspan(center_start, center_end, alpha=0.2, color='red')
    axes[1, 2].set_title(f'Outliers - Axial\nRed band = center region')
    axes[1, 2].set_xlabel('X (left ← → right)')
    axes[1, 2].set_ylabel('Y (posterior ← → anterior)')
    
    # Row 3: Outlier counts per slice across each axis
    # X-axis (sagittal): count outliers in each x-slice
    outlier_counts_x = np.sum(outlier_mask, axis=(1, 2))
    axes[2, 0].bar(range(len(outlier_counts_x)), outlier_counts_x, color='steelblue', alpha=0.7)
    axes[2, 0].axvspan(center_start, center_end, alpha=0.3, color='red', 
                       label=f'Center region\n(slices {center_start}-{center_end})')
    axes[2, 0].axvline(center_idx, color='red', linestyle='--', linewidth=2, label=f'Center slice ({center_idx})')
    axes[2, 0].set_xlabel('X slice index (left ← → right)')
    axes[2, 0].set_ylabel('Outlier count')
    axes[2, 0].set_title('Outliers per sagittal slice\n**This shows if outliers cluster in center X slices**')
    axes[2, 0].legend(fontsize=8)
    
    # Y-axis (coronal): count outliers in each y-slice
    outlier_counts_y = np.sum(outlier_mask, axis=(0, 2))
    axes[2, 1].bar(range(len(outlier_counts_y)), outlier_counts_y, color='steelblue', alpha=0.7)
    axes[2, 1].set_xlabel('Y slice index (posterior ← → anterior)')
    axes[2, 1].set_ylabel('Outlier count')
    axes[2, 1].set_title('Outliers per coronal slice\n(would show front-back clustering)')
    
    # Z-axis (axial): count outliers in each z-slice
    outlier_counts_z = np.sum(outlier_mask, axis=(0, 1))
    axes[2, 2].bar(range(len(outlier_counts_z)), outlier_counts_z, color='steelblue', alpha=0.7)
    axes[2, 2].set_xlabel('Z slice index (inferior ← → superior)')
    axes[2, 2].set_ylabel('Outlier count')
    axes[2, 2].set_title('Outliers per axial slice\n(would show up-down clustering)')
    
    # Add artifact detection info if provided
    if artifact_info:
        flag = artifact_info.get('central_artifact_flag', False)
        conc = artifact_info.get('central_concentration', 0)
        info_text = (
            f"Central Sagittal Artifact Detection:\n"
            f"Checking X slices {center_start}-{center_end} (red band in coronal/axial)\n"
            f"FLAG: {'YES - ARTIFACT DETECTED' if flag else 'NO'}\n"
            f"Concentration in center: {conc:.1%}\n"
            f"Center outliers: {artifact_info.get('outliers_in_center', 0)}\n"
            f"Total outliers: {artifact_info.get('outliers_total', 0)}\n\n"
            f"Interpretation: {conc:.0%} of extreme outliers are\n"
            f"in the center sagittal slices (midline region)"
        )
        fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', 
                         facecolor='yellow' if flag else 'lightgreen', 
                         alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
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
