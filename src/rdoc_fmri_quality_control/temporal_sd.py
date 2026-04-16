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
    
    # Compute statistics per Z slice (only in masked brain regions)
    mean_sd_per_z = np.zeros(nz)
    median_sd_per_z = np.zeros(nz)
    p95_sd_per_z = np.zeros(nz)
    sum_sd_per_z = np.zeros(nz)
    
    for z in range(nz):
        slice_mask = mask[:, :, z]
        if np.sum(slice_mask) > 0:
            slice_vals = sd_map[:, :, z][slice_mask]
            mean_sd_per_z[z] = np.mean(slice_vals)
            median_sd_per_z[z] = np.median(slice_vals)
            p95_sd_per_z[z] = np.percentile(slice_vals, 95)
            sum_sd_per_z[z] = np.sum(slice_vals)
    
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
        "sum_sd_per_z": sum_sd_per_z.tolist(),
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
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Large sagittal view on left spanning 2 rows
    ax_sag = fig.add_subplot(gs[:2, 0])
    slice_idx = tsd_map.shape[0] // 2
    im_sag = ax_sag.imshow(tsd_map[slice_idx, :, :].T, cmap='viridis', origin='lower', 
                           vmin=0, vmax=np.percentile(tsd_map[mask], 99))
    if line_z is not None:
        ax_sag.axhline(line_z, color='red', linewidth=4, label=f'Detected line at Z={line_z}')
        # Add a few reference lines above and below
        ax_sag.axhline(line_z - 5, color='cyan', linewidth=1, linestyle='--', alpha=0.5, label='Z-5')
        ax_sag.axhline(line_z + 5, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Z+5')
        ax_sag.legend(loc='upper right', fontsize=10)
    ax_sag.set_title(f'Temporal SD - Sagittal (X slice {slice_idx})\n{"RED LINE = DETECTED ARTIFACT" if has_line else "No line detected"}', 
                    fontweight='bold' if has_line else 'normal', fontsize=14)
    ax_sag.set_xlabel('Y (posterior ← → anterior)', fontsize=11)
    ax_sag.set_ylabel('Z (inferior ← → superior)', fontsize=11)
    plt.colorbar(im_sag, ax=ax_sag, fraction=0.046, label='Temporal SD')
    
    # Show sum of temporal SD per Z slice (bar chart)
    ax_sum = fig.add_subplot(gs[2, 0])
    if artifact_info and 'sum_sd_per_z' in artifact_info:
        sum_sd_per_z = np.array(artifact_info['sum_sd_per_z'])
        z_indices = np.arange(len(sum_sd_per_z))
        
        # Color bars based on position relative to detected line
        if line_z is not None:
            colors = ['red' if z == line_z else 'orange' if abs(z - line_z) <= 2 else 'steelblue' 
                     for z in z_indices]
        else:
            colors = 'steelblue'
        
        ax_sum.bar(z_indices, sum_sd_per_z, color=colors, alpha=0.7, width=1.0)
        if line_z is not None:
            ax_sum.axvline(line_z, color='red', linewidth=3, linestyle='--', label=f'Detected line (Z={line_z})')
            ax_sum.legend(fontsize=9)
        
        ax_sum.set_xlabel('Z slice (inferior → superior)', fontsize=11)
        ax_sum.set_ylabel('Sum of all voxel temporal SDs', fontsize=11)
        ax_sum.set_title('**Total SD per Z Slice**\nHigher bars = more temporal variability in that slice', 
                        fontsize=11, fontweight='bold')
        ax_sum.grid(True, alpha=0.3, axis='y')
    else:
        ax_sum.text(0.5, 0.5, 'No detection data', ha='center', va='center', transform=ax_sum.transAxes)
    
    # Top right: Full SD profile across all Z slices
    ax_profile = fig.add_subplot(gs[0, 1:])
    if artifact_info and 'mean_sd_per_z' in artifact_info:
        mean_sd_per_z = np.array(artifact_info['mean_sd_per_z'])
        p95_sd_per_z = np.array(artifact_info['p95_sd_per_z'])
        z_indices = np.arange(len(mean_sd_per_z))
        
        ax_profile.plot(z_indices, mean_sd_per_z, 'b-', label='Mean SD per slice', linewidth=2, alpha=0.7)
        ax_profile.plot(z_indices, p95_sd_per_z, 'g-', label='95th %ile SD per slice', linewidth=3)
        if line_z is not None:
            ax_profile.axvline(line_z, color='red', linewidth=4, linestyle='--', label=f'DETECTED LINE (Z={line_z})', zorder=10)
            # Shade baseline region
            baseline_end = max(0, line_z - 2)
            ax_profile.axvspan(max(0, line_z - 10), baseline_end, alpha=0.15, color='blue', label='Baseline region')
            # Shade line region
            ax_profile.axvspan(line_z, min(len(z_indices), line_z + 5), alpha=0.15, color='red', label='Line region')
        ax_profile.set_xlabel('Z slice index (inferior → superior)', fontsize=12)
        ax_profile.set_ylabel('SD value', fontsize=12)
        ax_profile.set_title('**KEY DIAGNOSTIC: SD Profile Across Z Slices**\nGreen line should jump sharply at red line', 
                            fontsize=13, fontweight='bold')
        ax_profile.legend(loc='best', fontsize=10)
        ax_profile.grid(True, alpha=0.3)
    else:
        ax_profile.text(0.5, 0.5, 'No detection data', ha='center', va='center', transform=ax_profile.transAxes)
    
    # Middle right: Gradient showing the jump
    ax_gradient = fig.add_subplot(gs[1, 1:])
    if artifact_info and 'p95_sd_per_z' in artifact_info:
        p95_sd_per_z = np.array(artifact_info['p95_sd_per_z'])
        gradients = np.diff(p95_sd_per_z)
        z_indices = np.arange(len(gradients))
        
        colors = ['red' if i == line_z else 'steelblue' for i in z_indices]
        ax_gradient.bar(z_indices, gradients, color=colors, alpha=0.7, width=1.0)
        if line_z is not None:
            ax_gradient.axvline(line_z, color='red', linewidth=3, linestyle='--', alpha=0.5)
        ax_gradient.set_xlabel('Z slice index', fontsize=12)
        ax_gradient.set_ylabel('SD change (slice[i+1] - slice[i])', fontsize=12)
        ax_gradient.set_title('SD Gradient: Where Does SD Jump?\nRed bar = largest positive jump (the detected line)', 
                             fontsize=13, fontweight='bold')
        ax_gradient.grid(True, alpha=0.3, axis='y')
        ax_gradient.axhline(0, color='black', linewidth=0.5)
    else:
        ax_gradient.text(0.5, 0.5, 'No detection data', ha='center', va='center', transform=ax_gradient.transAxes)
    
    # Bottom right: Detection summary
    ax_summary = fig.add_subplot(gs[2, 1:])
    if artifact_info:
        metrics_text = (
            f"═══ HORIZONTAL LINE ARTIFACT DETECTION ═══\n\n"
            f"Artifact detected: {'✓ YES' if has_line else '✗ NO'}\n"
            f"Detected at Z slice: {line_z if line_z is not None else 'N/A'}\n\n"
            f"SD at baseline (below line): {artifact_info.get('baseline_sd', 0):.2f}\n"
            f"SD at line (at/above line): {artifact_info.get('line_sd', 0):.2f}\n"
            f"Ratio (line/baseline): {artifact_info.get('line_sd_ratio', 0):.2f}x\n\n"
            f"═══ INTERPRETATION ═══\n"
            f"The algorithm found the largest SD jump at Z={line_z if line_z is not None else 'N/A'}.\n"
            f"SD is {artifact_info.get('line_sd_ratio', 0):.1f}x higher at/above the line\n"
            f"compared to slices well below it.\n\n"
            f"If the RED LINE on the left image doesn't match\n"
            f"where YOU see the artifact, the detection needs tuning."
        )
        ax_summary.text(0.05, 0.95, metrics_text, transform=ax_summary.transAxes,
                       verticalalignment='top', fontsize=11, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='yellow' if has_line else 'lightgreen', 
                                alpha=0.95, edgecolor='black', linewidth=3))
        ax_summary.axis('off')
    else:
        ax_summary.text(0.5, 0.5, 'No detection data', ha='center', va='center', transform=ax_summary.transAxes)
        ax_summary.axis('off')
    
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
