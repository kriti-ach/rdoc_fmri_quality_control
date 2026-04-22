from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Union

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


def compute_sd_metrics(nifti_path: Union[str, Path], min_mean: float = 1e-6) -> tuple[SdMetrics, np.ndarray, np.ndarray]:
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


def save_sd_nifti(sd_map: np.ndarray, source_nifti_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    src = nib.load(str(source_nifti_path))
    out = nib.Nifti1Image(sd_map.astype(np.float32), src.affine, src.header)
    out.header.set_data_shape(sd_map.shape)
    nib.save(out, str(output_path))

def detect_horizontal_line_artifact(
    sd_map: np.ndarray,
    mask: np.ndarray,
    min_line_width: int = 3,
    gradient_threshold: float = 2.0,
) -> Dict[str, Union[float, bool, int]]:
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
from matplotlib import cm

def visualize_outliers(tsd_map, mask, z_threshold=5, save_path=None, center_width=5, artifact_info=None):
    """
    Create visualization showing SD map and detected horizontal line artifact.
    
    Parameters
    ----------
    artifact_info : dict, optional
        Output from detect_horizontal_line_artifact() to display on plot
    """
    # Get detected line location from artifact_info (for summary text only)
    line_z = None
    has_line = False
    if artifact_info:
        has_line = artifact_info.get('line_artifact_detected', False)
        line_z = artifact_info.get('line_z_coordinate', None)
        if line_z is not None and line_z < 0:
            line_z = None
    
    threshold = 80.0
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, height_ratios=[2.2, 1.0])

    # Sagittal: plane shape (nz, ny) after transpose of (ny, nz)
    ax_sag = fig.add_subplot(gs[0, 0])
    slice_idx = tsd_map.shape[0] // 2
    plane = tsd_map[slice_idx, :, :].T
    nz_p, ny_p = plane.shape
    plane_mask = mask[slice_idx, :, :].T.astype(bool)
    left_vals = plane[plane_mask] if np.any(plane_mask) else plane.ravel()
    # Robust "natural" scaling to avoid single-voxel spikes dominating the colorbar.
    left_vmax = float(np.percentile(left_vals, 99))
    im_sag = ax_sag.imshow(
        plane,
        cmap='viridis',
        origin='lower',
        vmin=0,
        vmax=left_vmax,
        extent=(0, ny_p - 1, 0, nz_p - 1),
        aspect='auto',
    )
    ax_sag.set_title(f'Temporal SD — Sagittal (X slice {slice_idx})', fontsize=14)
    ax_sag.set_xlabel('Y index (posterior ← → anterior)', fontsize=11)
    ax_sag.set_ylabel('Z index (inferior ← → superior)', fontsize=11)
    # Tick for every column (Y) and every row (Z)
    ax_sag.set_xticks(np.arange(ny_p))
    ax_sag.set_xticklabels([str(i) for i in range(ny_p)], fontsize=6, rotation=45, ha='right')
    ax_sag.set_yticks(np.arange(nz_p))
    ax_sag.set_yticklabels([str(i) for i in range(nz_p)], fontsize=6)
    plt.colorbar(im_sag, ax=ax_sag, fraction=0.046, label='Temporal SD')

    # Same sagittal view, but black out voxels above threshold.
    ax_sag_clip = fig.add_subplot(gs[0, 1])
    plane_masked = np.ma.masked_where(plane > threshold, plane)
    cmap_masked = cm.get_cmap("viridis").copy()
    cmap_masked.set_bad(color="black")
    im_sag_clip = ax_sag_clip.imshow(
        plane_masked,
        cmap=cmap_masked,
        origin='lower',
        vmin=0,
        vmax=left_vmax,
        extent=(0, ny_p - 1, 0, nz_p - 1),
        aspect='auto',
    )
    ax_sag_clip.set_title(
        f'Temporal SD — Sagittal (X slice {slice_idx}) [values > {int(threshold)} blacked out]',
        fontsize=14,
    )
    ax_sag_clip.set_xlabel('Y index (posterior ← → anterior)', fontsize=11)
    ax_sag_clip.set_ylabel('Z index (inferior ← → superior)', fontsize=11)
    ax_sag_clip.set_xticks(np.arange(ny_p))
    ax_sag_clip.set_xticklabels([str(i) for i in range(ny_p)], fontsize=6, rotation=45, ha='right')
    ax_sag_clip.set_yticks(np.arange(nz_p))
    ax_sag_clip.set_yticklabels([str(i) for i in range(nz_p)], fontsize=6)
    plt.colorbar(im_sag_clip, ax=ax_sag_clip, fraction=0.046, label='Temporal SD (same scale, blacked > 80)')

    # Sum of temporal SD per Z slice (bar chart, no reference lines)
    ax_sum = fig.add_subplot(gs[1, 0])
    if artifact_info and 'sum_sd_per_z' in artifact_info:
        sum_sd_per_z = np.array(artifact_info['sum_sd_per_z'])
        z_indices = np.arange(len(sum_sd_per_z))

        ax_sum.bar(
            z_indices,
            sum_sd_per_z,
            color='steelblue',
            alpha=0.85,
            width=1.0,
            edgecolor='black',
            linewidth=0.5,
        )

        n_z = len(sum_sd_per_z)
        ax_sum.set_xticks(np.arange(n_z))
        fs = 6 if n_z > 40 else 7
        ax_sum.set_xticklabels([str(i) for i in range(n_z)], fontsize=fs, rotation=45, ha='right')
        ax_sum.set_xlim(-0.5, n_z - 0.5)

        ax_sum.set_xlabel('Z slice (inferior → superior)', fontsize=11)
        ax_sum.set_ylabel('Sum of all voxel temporal SDs', fontsize=11)
        ax_sum.set_title(
            'Total SD per Z slice (sum over voxels)',
            fontsize=11,
            fontweight='bold',
        )
        ax_sum.grid(True, alpha=0.3, axis='y')
    else:
        ax_sum.text(0.5, 0.5, 'No detection data', ha='center', va='center', transform=ax_sum.transAxes)

    # Matching bar chart for clipped SD values.
    ax_sum_clip = fig.add_subplot(gs[1, 1])
    nz = tsd_map.shape[2]
    sum_sd_clip_per_z = np.zeros(nz, dtype=np.float64)
    for z in range(nz):
        slice_mask = mask[:, :, z]
        if np.any(slice_mask):
            vals = tsd_map[:, :, z][slice_mask]
            sum_sd_clip_per_z[z] = np.sum(np.clip(vals, 0.0, threshold))

    z_indices_clip = np.arange(len(sum_sd_clip_per_z))
    ax_sum_clip.bar(
        z_indices_clip,
        sum_sd_clip_per_z,
        color='steelblue',
        alpha=0.85,
        width=1.0,
        edgecolor='black',
        linewidth=0.5,
    )
    n_z_clip = len(sum_sd_clip_per_z)
    ax_sum_clip.set_xticks(np.arange(n_z_clip))
    fs_clip = 6 if n_z_clip > 40 else 7
    ax_sum_clip.set_xticklabels([str(i) for i in range(n_z_clip)], fontsize=fs_clip, rotation=45, ha='right')
    ax_sum_clip.set_xlim(-0.5, n_z_clip - 0.5)
    ax_sum_clip.set_xlabel('Z slice (inferior → superior)', fontsize=11)
    ax_sum_clip.set_ylabel(f'Sum of voxel temporal SDs (<= {int(threshold)})', fontsize=11)
    ax_sum_clip.set_title(
        f'Total SD per Z slice (sum over voxels, values > {int(threshold)} removed)',
        fontsize=11,
        fontweight='bold',
    )
    ax_sum_clip.grid(True, alpha=0.3, axis='y')
    # Keep the same y-axis range as the left bar chart for direct comparison.
    ax_sum_clip.set_ylim(ax_sum.get_ylim())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    
    return fig

def save_mid_sagittal_png(sd_map: np.ndarray, title: str, output_png: Union[str, Path]) -> None:
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
