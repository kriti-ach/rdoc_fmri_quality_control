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

def detect_horizontal_line_artifact(
    sd_map: np.ndarray,
    mask: np.ndarray,
    min_line_width: int = 3,
    gradient_threshold: float = 2.0,
) -> dict[str, float | bool | int]:
    """
    Detect horizontal line artifact by finding sharp SD transition across Z slices.
    
    Looks for a Z coordinate where SD values suddenly increase, indicating
    a horizontal line/band of high variability.
    
    Parameters
    ----------
    sd_map : np.ndarray
        3D temporal SD map
    mask : np.ndarray
        Brain mask
    min_line_width : int
        Minimum width in voxels for elevated band
    gradient_threshold : float
        Minimum SD jump ratio to flag as artifact
        
    Returns
    -------
    dict with line artifact detection results
    """
    nz = sd_map.shape[2]
    
    # Compute mean SD per Z slice (only in masked brain regions)
    mean_sd_per_z = np.zeros(nz)
    median_sd_per_z = np.zeros(nz)
    p95_sd_per_z = np.zeros(nz)
    
    for z in range(nz):
        slice_mask = mask[:, :, z]
        if np.sum(slice_mask) > 0:
            slice_vals = sd_map[:, :, z][slice_mask]
            mean_sd_per_z[z] = np.mean(slice_vals)
            median_sd_per_z[z] = np.median(slice_vals)
            p95_sd_per_z[z] = np.percentile(slice_vals, 95)
    
    # Find the largest positive gradient (SD jump) in the p95 values
    # Use p95 instead of mean to be more sensitive to extreme values
    gradients = np.diff(p95_sd_per_z)
    
    # Find peaks in gradient (where SD suddenly increases)
    if len(gradients) == 0:
        return {
            "line_artifact_detected": False,
            "line_z_coordinate": -1,
            "line_sd_ratio": 0.0,
            "baseline_sd": 0.0,
            "line_sd": 0.0,
        }
    
    max_grad_idx = int(np.argmax(gradients))
    max_gradient = float(gradients[max_grad_idx])
    
    # Check if there's a sustained elevated region after the jump
    z_line = max_grad_idx  # This is where the jump happens
    
    # Compute baseline (mean of slices well below the line)
    baseline_range = slice(max(0, z_line - 10), max(1, z_line - 2))
    baseline_sd = float(np.mean(p95_sd_per_z[baseline_range])) if z_line > 2 else float(p95_sd_per_z[0])
    
    # Compute line region SD (slices at and just above the line)
    line_range = slice(z_line, min(nz, z_line + min_line_width + 2))
    line_sd = float(np.mean(p95_sd_per_z[line_range]))
    
    # Ratio: how much higher is the line region vs baseline
    sd_ratio = line_sd / baseline_sd if baseline_sd > 0 else 0.0
    
    # Flag as artifact if the ratio exceeds threshold
    has_artifact = sd_ratio >= gradient_threshold and baseline_sd > 0
    
    return {
        "line_artifact_detected": bool(has_artifact),
        "line_z_coordinate": int(z_line),
        "line_sd_ratio": float(sd_ratio),
        "baseline_sd": float(baseline_sd),
        "line_sd": float(line_sd),
        "mean_sd_per_z": mean_sd_per_z.tolist(),
        "p95_sd_per_z": p95_sd_per_z.tolist(),
    }

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_outliers(tsd_map, mask, z_threshold=5, save_path=None, center_width=5, artifact_info=None):
    """
    Create visualization showing SD map and detected horizontal line artifact.
    
    Parameters
    ----------
    artifact_info : dict, optional
        Output from detect_horizontal_line_artifact() to display on plot
    """
    median_tsd = np.median(tsd_map[mask])
    mad = np.median(np.abs(tsd_map[mask] - median_tsd))
    robust_std = 1.4826 * mad
    z_scores = (tsd_map - median_tsd) / robust_std
    
    outlier_mask = (z_scores > z_threshold) & mask
    
    # Get detected line location from artifact_info
    line_z = None
    has_line = False
    if artifact_info:
        has_line = artifact_info.get('line_artifact_detected', False)
        line_z = artifact_info.get('line_z_coordinate', None)
        if line_z is not None and line_z < 0:
            line_z = None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: TSD maps with DETECTED LINE marked
    # Sagittal - mark the detected horizontal line
    slice_idx = tsd_map.shape[0] // 2
    im0 = axes[0, 0].imshow(tsd_map[slice_idx, :, :].T, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(tsd_map[mask], 99))
    if line_z is not None:
        axes[0, 0].axhline(line_z, color='red', linewidth=3, label=f'Detected line at Z={line_z}')
        axes[0, 0].legend(loc='upper right', fontsize=10)
    axes[0, 0].set_title(f'TSD - Sagittal (X slice {slice_idx})\n{"RED LINE = DETECTED ARTIFACT" if has_line else "No line detected"}', 
                        fontweight='bold' if has_line else 'normal')
    axes[0, 0].set_xlabel('Y (posterior ← → anterior)')
    axes[0, 0].set_ylabel('Z (inferior ← → superior)')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, label='TSD')
    
    # Coronal - mark the detected horizontal line
    slice_idx = tsd_map.shape[1] // 2
    im1 = axes[0, 1].imshow(tsd_map[:, slice_idx, :].T, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(tsd_map[mask], 99))
    if line_z is not None:
        axes[0, 1].axhline(line_z, color='red', linewidth=3, label=f'Detected line at Z={line_z}')
        axes[0, 1].legend(loc='upper right', fontsize=10)
    axes[0, 1].set_title(f'TSD - Coronal (Y slice {slice_idx})\n{"RED LINE = DETECTED ARTIFACT" if has_line else "No line detected"}',
                        fontweight='bold' if has_line else 'normal')
    axes[0, 1].set_xlabel('X (left ← → right)')
    axes[0, 1].set_ylabel('Z (inferior ← → superior)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='TSD')
    
    # Axial - show the slice at the detected line
    if line_z is not None and 0 <= line_z < tsd_map.shape[2]:
        slice_idx = line_z
    else:
        slice_idx = tsd_map.shape[2] // 2
    im2 = axes[0, 2].imshow(tsd_map[:, :, slice_idx].T, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(tsd_map[mask], 99))
    axes[0, 2].set_title(f'TSD - Axial (Z slice {slice_idx})\n{"THIS IS THE LINE SLICE" if line_z == slice_idx else ""}',
                        fontweight='bold' if line_z == slice_idx else 'normal')
    axes[0, 2].set_xlabel('X (left ← → right)')
    axes[0, 2].set_ylabel('Y (posterior ← → anterior)')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='TSD')
    
    # Row 2: SD profiles showing line detection
    # Plot mean and p95 SD per Z slice
    if artifact_info and 'mean_sd_per_z' in artifact_info:
        mean_sd_per_z = np.array(artifact_info['mean_sd_per_z'])
        p95_sd_per_z = np.array(artifact_info['p95_sd_per_z'])
        z_indices = np.arange(len(mean_sd_per_z))
        
        axes[1, 0].plot(z_indices, mean_sd_per_z, 'b-', label='Mean SD', linewidth=2)
        axes[1, 0].plot(z_indices, p95_sd_per_z, 'g-', label='95th percentile SD', linewidth=2)
        if line_z is not None:
            axes[1, 0].axvline(line_z, color='red', linewidth=3, linestyle='--', label=f'Detected line (Z={line_z})')
        axes[1, 0].set_xlabel('Z slice (inferior → superior)')
        axes[1, 0].set_ylabel('SD value')
        axes[1, 0].set_title('SD Profile Across Z Slices\n**Sharp jump = horizontal line artifact**')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Show gradient (rate of change)
        gradients = np.diff(p95_sd_per_z)
        axes[1, 1].bar(z_indices[:-1], gradients, color='steelblue', alpha=0.7, width=1.0)
        if line_z is not None:
            axes[1, 1].axvline(line_z, color='red', linewidth=3, linestyle='--', label=f'Detected line')
        axes[1, 1].set_xlabel('Z slice (inferior → superior)')
        axes[1, 1].set_ylabel('SD gradient (change between slices)')
        axes[1, 1].set_title('SD Gradient\n**Peak at detected line shows sharp transition**')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No artifact info available', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'No artifact info available', ha='center', va='center')
    
    # Show detection metrics
    if artifact_info:
        metrics_text = (
            f"Line Detection Results:\n\n"
            f"Artifact detected: {'YES' if has_line else 'NO'}\n"
            f"Line Z coordinate: {line_z if line_z is not None else 'N/A'}\n"
            f"SD ratio (line/baseline): {artifact_info.get('line_sd_ratio', 0):.2f}\n"
            f"Baseline SD (below line): {artifact_info.get('baseline_sd', 0):.1f}\n"
            f"Line SD (at/above line): {artifact_info.get('line_sd', 0):.1f}\n\n"
            f"Interpretation:\n"
            f"The line is at Z={line_z if line_z is not None else 'N/A'}\n"
            f"SD is {artifact_info.get('line_sd_ratio', 0):.1f}x higher at the line"
        )
        axes[1, 2].text(0.05, 0.95, metrics_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=10, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='yellow' if has_line else 'lightgreen', 
                                alpha=0.9, edgecolor='black', linewidth=2))
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'No artifact info available', ha='center', va='center')
        axes[1, 2].axis('off')
    
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
