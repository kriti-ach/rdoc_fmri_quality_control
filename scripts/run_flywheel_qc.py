#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path
from typing import Iterable

import flywheel
import nibabel as nib
import numpy as np
import yaml

from rdoc_fmri_quality_control.temporal_sd import compute_sd_metrics, robust_z


def resolve_config_path(config_arg: str | None) -> Path:
    if config_arg:
        path = Path(config_arg).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    candidates = [Path("config/config.yaml"), Path("config/config.example.yaml")]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No config provided and none found at config/config.yaml or config/config.example.yaml. "
        "Pass --config /path/to/config.yaml."
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_candidate_niftis(acq) -> Iterable[str]:
    for f in acq.files:
        name = f.name
        low = name.lower()
        if low.endswith(".nii") or low.endswith(".nii.gz"):
            yield name


def download_acquisition_file(fw: flywheel.Client, acq_id: str, file_name: str, out_path: Path) -> None:
    if hasattr(fw, "download_file_from_acquisition"):
        fw.download_file_from_acquisition(acq_id, file_name, str(out_path))
        return
    acq = fw.get_acquisition(acq_id)
    acq.download_file(file_name, str(out_path))


def _obj_get(obj, key: str, default=None):
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def subject_label_from_session(ses) -> str:
    subj = _obj_get(ses, "subject", {})
    for key in ("label", "code", "firstname", "id"):
        val = _obj_get(subj, key, "")
        if val:
            return str(val)
    return ""


def session_time_key(ses) -> str:
    # We only need a stable chronological-ish sort key for within-subject ranking.
    for key in ("timestamp", "created", "modified", "label"):
        val = _obj_get(ses, key, "")
        if val:
            return str(val)
    return ""


def infer_task_label(file_name: str, acq_label: str, session_label: str) -> str:
    text_candidates = [file_name, acq_label, session_label]
    patterns = [
        re.compile(r"task-([A-Za-z0-9]+)", re.IGNORECASE),
        re.compile(r"_([A-Za-z][A-Za-z0-9]+)_bold", re.IGNORECASE),
    ]
    for text in text_candidates:
        if not text:
            continue
        for pat in patterns:
            m = pat.search(text)
            if m:
                return m.group(1)
    return ""


def compute_roi_sd_metrics(sd_map: np.ndarray, brain_mask: np.ndarray, roi_mask: np.ndarray) -> dict[str, float | int]:
    roi = np.logical_and(brain_mask, roi_mask > 0)
    vals = sd_map[roi]
    if vals.size == 0:
        raise ValueError("ROI mask has no overlap with brain mask.")

    z, _, _ = robust_z(vals)
    n6 = int(np.sum(z >= 6.0))
    n8 = int(np.sum(z >= 8.0))
    return {
        "roi_n_voxels": int(vals.size),
        "roi_sd_mean": float(np.mean(vals)),
        "roi_sd_p95": float(np.percentile(vals, 95)),
        "roi_sd_p99": float(np.percentile(vals, 99)),
        "roi_sd_max": float(np.max(vals)),
        "roi_robust_zmax": float(np.max(z)),
        "roi_frac_z_ge_6": float(n6 / vals.size),
        "roi_frac_z_ge_8": float(n8 / vals.size),
    }


def compute_outlier_location_metrics(
    sd_map: np.ndarray,
    brain_mask: np.ndarray,
    z_threshold: float = 8.0,
    midline_half_width_frac_x: float = 0.10,
) -> dict[str, float | int | str]:
    vals = sd_map[brain_mask]
    z, _, _ = robust_z(vals)

    brain_flat_idx = np.flatnonzero(brain_mask)
    outlier_flat_idx = brain_flat_idx[z >= z_threshold]
    n_outliers = int(outlier_flat_idx.size)

    if n_outliers == 0:
        return {
            "extreme_z_threshold": float(z_threshold),
            "n_extreme_outlier_voxels": 0,
            "frac_extreme_outlier_voxels": 0.0,
            "midline_half_width_vox_x": int(max(1, round(sd_map.shape[0] * midline_half_width_frac_x))),
            "n_extreme_midline_voxels": 0,
            "frac_extreme_midline": 0.0,
            "extreme_com_x": np.nan,
            "extreme_com_y": np.nan,
            "extreme_com_z": np.nan,
            "extreme_lr_bias": "none",
        }

    xyz = np.column_stack(np.unravel_index(outlier_flat_idx, sd_map.shape))
    x = xyz[:, 0].astype(np.float64)
    mid_x = (sd_map.shape[0] - 1) / 2.0
    midline_half_width_vox = int(max(1, round(sd_map.shape[0] * midline_half_width_frac_x)))
    is_midline = np.abs(x - mid_x) <= midline_half_width_vox
    n_midline = int(np.sum(is_midline))

    com_x = float(np.mean(x))
    com_y = float(np.mean(xyz[:, 1]))
    com_z = float(np.mean(xyz[:, 2]))
    if abs(com_x - mid_x) <= midline_half_width_vox:
        lr_bias = "midline"
    elif com_x < mid_x:
        lr_bias = "left"
    else:
        lr_bias = "right"

    return {
        "extreme_z_threshold": float(z_threshold),
        "n_extreme_outlier_voxels": n_outliers,
        "frac_extreme_outlier_voxels": float(n_outliers / np.count_nonzero(brain_mask)),
        "midline_half_width_vox_x": midline_half_width_vox,
        "n_extreme_midline_voxels": n_midline,
        "frac_extreme_midline": float(n_midline / n_outliers),
        "extreme_com_x": com_x,
        "extreme_com_y": com_y,
        "extreme_com_z": com_z,
        "extreme_lr_bias": lr_bias,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal SD QC across Flywheel acquisitions")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config (optional; defaults to config/config.yaml if present).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick testing")
    parser.add_argument(
        "--task-name-contains",
        default="",
        help="Optional substring filter for tasks/scans. Leave empty to include all tasks.",
    )
    parser.add_argument("--subject-label", default=None, help="Only include sessions for this subject label/code (e.g., s23)")
    parser.add_argument(
        "--session-rank",
        type=int,
        default=None,
        help="Within-subject session rank (1-based chronological order): 1=prelim, 2=session0, 3=session1, ...",
    )
    parser.add_argument(
        "--file-regex",
        default=None,
        help="Optional regex to filter scan filenames (e.g., '_e2\\.nii(\\.gz)?$')",
    )
    parser.add_argument(
        "--line-roi-mask",
        default=None,
        help="Optional ROI mask NIfTI for line-artifact area; must match BOLD volume shape.",
    )
    parser.add_argument(
        "--extreme-z-threshold",
        type=float,
        default=8.0,
        help="Robust-z threshold used to define extreme temporal-SD outlier voxels.",
    )
    parser.add_argument(
        "--midline-half-width-frac-x",
        type=float,
        default=0.10,
        help="Midline half-width in x dimension as fraction of image width (for outlier localization).",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    fw_cfg = cfg["flywheel"]
    qc = cfg["qc"]

    fw = flywheel.Client(fw_cfg["api_key"])
    group_id = fw_cfg["group_id"]
    project_label = fw_cfg["project_label"]
    task_filter = args.task_name_contains.strip()

    projects = fw.projects.find(f"group={group_id},label={project_label}")
    if not projects:
        raise RuntimeError(f"No project found for group={group_id}, label={project_label}")
    project = projects[0]

    out_dir = Path(qc["all_scans_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "flywheel_temporal_sd_metrics.csv"

    file_regex = re.compile(args.file_regex) if args.file_regex else None
    roi_mask_data = None
    if args.line_roi_mask:
        roi_mask_data = np.asanyarray(nib.load(args.line_roi_mask).dataobj)
        if roi_mask_data.ndim != 3:
            raise ValueError(f"ROI mask must be 3D; got shape={roi_mask_data.shape}")

    rows = []
    n_processed = 0
    sessions = fw.get_project_sessions(project.id)
    if args.subject_label:
        sessions = [
            ses
            for ses in sessions
            if subject_label_from_session(ses).lower() == args.subject_label.lower()
        ]
    if args.session_rank is not None:
        if args.session_rank < 1:
            raise ValueError("--session-rank must be >= 1")
        sessions = sorted(sessions, key=session_time_key)
        rank_idx = args.session_rank - 1
        if rank_idx >= len(sessions):
            raise RuntimeError(
                f"Subject/session filter found only {len(sessions)} sessions; rank {args.session_rank} is out of range."
            )
        sessions = [sessions[rank_idx]]

    for ses in sessions:
        for acq in fw.get_session_acquisitions(ses.id):
            for fname in iter_candidate_niftis(acq):
                if task_filter and task_filter.lower() not in fname.lower():
                    continue
                if file_regex and not file_regex.search(fname):
                    continue
                with tempfile.TemporaryDirectory(prefix="fw_qc_") as tmpd:
                    local_path = Path(tmpd) / Path(fname).name
                    try:
                        download_acquisition_file(fw, acq.id, fname, local_path)
                        metrics, sd_map, brain_mask = compute_sd_metrics(
                            local_path,
                            min_mean=float(qc.get("mask_min_mean", 1e-6)),
                        )
                        loc_metrics = compute_outlier_location_metrics(
                            sd_map=sd_map,
                            brain_mask=brain_mask,
                            z_threshold=float(args.extreme_z_threshold),
                            midline_half_width_frac_x=float(args.midline_half_width_frac_x),
                        )
                        roi_metrics = {}
                        if roi_mask_data is not None:
                            if tuple(roi_mask_data.shape) != tuple(sd_map.shape):
                                raise ValueError(
                                    f"ROI shape {roi_mask_data.shape} != BOLD shape {sd_map.shape} for {fname}"
                                )
                            roi_metrics = compute_roi_sd_metrics(sd_map, brain_mask, roi_mask_data)
                    except Exception as e:  # noqa: BLE001
                        rows.append(
                            {
                                "project": project.label,
                                "session_label": ses.label,
                                "acquisition_label": acq.label,
                                "file_name": fname,
                                "status": "failed",
                                "error": str(e),
                            }
                        )
                        continue

                row = metrics.to_dict()
                row.update(
                    {
                        "project": project.label,
                        "session_label": ses.label,
                        "subject_label": subject_label_from_session(ses),
                        "task_label": infer_task_label(fname, str(_obj_get(acq, "label", "")), str(_obj_get(ses, "label", ""))),
                        "acquisition_label": acq.label,
                        "file_name": fname,
                        "status": "ok",
                        "error": "",
                    }
                )
                if roi_mask_data is not None:
                    row.update(roi_metrics)
                row.update(loc_metrics)
                rows.append(row)
                n_processed += 1
                print(f"processed {n_processed}: {ses.label} | {acq.label} | {fname}")

                if args.limit and n_processed >= args.limit:
                    break
            if args.limit and n_processed >= args.limit:
                break
        if args.limit and n_processed >= args.limit:
            break

    # Stable header across success/failure rows.
    base_cols = [
        "project",
        "subject_label",
        "task_label",
        "session_label",
        "acquisition_label",
        "file_name",
        "status",
        "error",
    ]
    metric_cols = [
        "nifti_path",
        "n_timepoints",
        "n_brain_voxels",
        "sd_mean",
        "sd_median",
        "sd_p95",
        "sd_p99",
        "sd_max",
        "mad",
        "robust_z99",
        "robust_zmax",
        "n_z_ge_6",
        "n_z_ge_8",
        "frac_z_ge_6",
        "frac_z_ge_8",
    ]
    roi_cols = [
        "roi_n_voxels",
        "roi_sd_mean",
        "roi_sd_p95",
        "roi_sd_p99",
        "roi_sd_max",
        "roi_robust_zmax",
        "roi_frac_z_ge_6",
        "roi_frac_z_ge_8",
    ]
    loc_cols = [
        "extreme_z_threshold",
        "n_extreme_outlier_voxels",
        "frac_extreme_outlier_voxels",
        "midline_half_width_vox_x",
        "n_extreme_midline_voxels",
        "frac_extreme_midline",
        "extreme_com_x",
        "extreme_com_y",
        "extreme_com_z",
        "extreme_lr_bias",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + metric_cols + roi_cols + loc_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_csv}")
    print(f"Total rows: {len(rows)}; successful scans: {sum(r['status'] == 'ok' for r in rows)}")


if __name__ == "__main__":
    main()
