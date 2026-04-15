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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal SD QC across Flywheel acquisitions")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick testing")
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
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    fw_cfg = cfg["flywheel"]
    qc = cfg["qc"]

    fw = flywheel.Client(fw_cfg["api_key"])
    group_id = fw_cfg["group_id"]
    project_label = fw_cfg["project_label"]
    task_filter = fw_cfg.get("task_name_contains", "")

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
                        "acquisition_label": acq.label,
                        "file_name": fname,
                        "status": "ok",
                        "error": "",
                    }
                )
                if roi_mask_data is not None:
                    row.update(roi_metrics)
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

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + metric_cols + roi_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_csv}")
    print(f"Total rows: {len(rows)}; successful scans: {sum(r['status'] == 'ok' for r in rows)}")


if __name__ == "__main__":
    main()
