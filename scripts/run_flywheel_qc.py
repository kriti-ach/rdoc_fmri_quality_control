#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path
from typing import Iterable

import flywheel
import yaml

from rdoc_fmri_quality_control.temporal_sd import (
    compute_sd_metrics,
    detect_horizontal_line_artifact,
    visualize_outliers,
)


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
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save outlier visualization PNGs for each scan.",
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
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    fw_cfg = cfg["flywheel"]
    qc = cfg["qc"]

    # Match other pipelines: rely on Flywheel auth already present in environment/session.
    fw = flywheel.Client()
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
    
    viz_dir = None
    if args.save_visualizations:
        viz_dir = out_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    file_regex = re.compile(args.file_regex) if args.file_regex else None

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
                        artifact_metrics = detect_horizontal_line_artifact(
                            sd_map=sd_map,
                            mask=brain_mask,
                            min_line_width=3,
                            gradient_threshold=2.0,
                        )
                        
                        if viz_dir is not None:
                            viz_name = f"{subject_label_from_session(ses)}_{ses.label}_{Path(fname).stem}_outliers.png"
                            viz_path = viz_dir / viz_name
                            visualize_outliers(
                                tsd_map=sd_map,
                                mask=brain_mask,
                                z_threshold=8,
                                save_path=str(viz_path),
                                center_width=5,
                                artifact_info=artifact_metrics,
                            )
                            print(f"  saved visualization: {viz_path.name}")
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
                row.update(artifact_metrics)
                rows.append(row)
                n_processed += 1
                print(f"processed {n_processed}: {ses.label} | {acq.label} | {fname}")

                if args.limit and n_processed >= args.limit:
                    break
            if args.limit and n_processed >= args.limit:
                break
        if args.limit and n_processed >= args.limit:
            break

    # Essential extent, prevalence, and horizontal line artifact detection metrics.
    fieldnames = [
        "project",
        "subject_label",
        "task_label",
        "session_label",
        "file_name",
        "status",
        "error",
        "n_timepoints",
        "n_brain_voxels",
        "sd_p99",
        "sd_p999",
        "sd_max",
        "robust_zmax",
        "frac_z_ge_5",
        "frac_z_ge_8",
        "frac_z_ge_10",
        "line_artifact_detected",
        "line_z_coordinate",
        "line_sd_ratio",
        "baseline_sd",
        "line_sd",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_csv}")
    print(f"Total rows: {len(rows)}; successful scans: {sum(r['status'] == 'ok' for r in rows)}")


if __name__ == "__main__":
    main()
