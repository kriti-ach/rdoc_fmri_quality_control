#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import flywheel
import yaml

from rdoc_fmri_quality_control.temporal_sd import (
    compute_sd_metrics,
    detect_horizontal_line_artifact,
    visualize_outliers,
)


def resolve_config_path(config_arg: Optional[str]) -> Path:
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


def parse_echo_index(file_name: str) -> Optional[int]:
    m = re.search(r"_e(\d+)\.nii(?:\.gz)?$", file_name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def strip_echo_suffix(file_name: str) -> str:
    return re.sub(r"_e\d+\.nii(?:\.gz)?$", "", file_name, flags=re.IGNORECASE)


def parse_tes(raw: Optional[str]) -> list[float]:
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_tedana_optcom(echo_paths: list[Path], tes_ms: list[float], out_dir: Path, prefix: str) -> Path:
    # tedana CLI expects TE values in milliseconds.
    cmd = [
        "tedana",
        "-d",
        *[str(p) for p in echo_paths],
        "-e",
        *[str(te) for te in tes_ms],
        "--out-dir",
        str(out_dir),
        "--prefix",
        prefix,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tedana failed: {proc.stderr.strip() or proc.stdout.strip()}")

    candidates = sorted(out_dir.glob(f"{prefix}*optcom*.nii.gz"))
    if not candidates:
        candidates = sorted(out_dir.glob("*optcom*.nii.gz"))
    if not candidates:
        raise RuntimeError("tedana completed but no optcom NIfTI was found.")
    return candidates[0]



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
        help="Optional regex to filter scan filenames (e.g., '_e2\\.nii(\\.gz)?$'). "
        "If omitted in single-echo mode, defaults to e2 only.",
    )
    parser.add_argument(
        "--combine-echoes",
        action="store_true",
        help="Group echoes per run and run tedana optimal combination before QC.",
    )
    parser.add_argument(
        "--echoes",
        default="1,2,3",
        help="Echo indices to combine when --combine-echoes is used (default: 1,2,3).",
    )
    parser.add_argument(
        "--tedana-tes-ms",
        default=None,
        help="Comma-separated TE values in milliseconds (required for --combine-echoes unless qc.tedana_tes_ms exists).",
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

    combine_echoes = bool(args.combine_echoes)
    if args.file_regex:
        file_regex = re.compile(args.file_regex)
    elif not combine_echoes:
        # Default single-echo behavior: analyze only e2 files.
        file_regex = re.compile(r"_e2\.nii(?:\.gz)?$", flags=re.IGNORECASE)
    else:
        file_regex = None
    requested_echoes = [int(x.strip()) for x in args.echoes.split(",") if x.strip()]
    if combine_echoes and not requested_echoes:
        raise ValueError("--echoes must include at least one echo index")

    tes_ms = parse_tes(args.tedana_tes_ms)
    if combine_echoes and not tes_ms:
        tes_ms = [float(x) for x in qc.get("tedana_tes_ms", [])]
    if combine_echoes and (not tes_ms or len(tes_ms) != len(requested_echoes)):
        raise ValueError(
            "For --combine-echoes, provide --tedana-tes-ms with one TE per echo "
            f"(got echoes={requested_echoes}, tes_ms={tes_ms})."
        )

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
            # Restrict to BOLD-style acquisitions to avoid fieldmap/fmap NIfTIs.
            acq_label = str(_obj_get(acq, "label", "")).lower()
            if "bold" not in acq_label:
                continue

            acq_files = list(iter_candidate_niftis(acq))
            if combine_echoes:
                grouped: dict[str, dict[int, str]] = {}
                for fname in acq_files:
                    if task_filter and task_filter.lower() not in fname.lower():
                        continue
                    if file_regex and not file_regex.search(fname):
                        continue
                    echo_idx = parse_echo_index(fname)
                    if echo_idx is None:
                        continue
                    base = strip_echo_suffix(fname)
                    grouped.setdefault(base, {})[echo_idx] = fname

                work_items: list[tuple[str, list[str], str]] = []
                for base, echo_map in grouped.items():
                    if not all(e in echo_map for e in requested_echoes):
                        continue
                    ordered_files = [echo_map[e] for e in requested_echoes]
                    out_name = f"{base}_optcom.nii.gz"
                    work_items.append((out_name, ordered_files, base))
            else:
                work_items = []
                for fname in acq_files:
                    if task_filter and task_filter.lower() not in fname.lower():
                        continue
                    if file_regex and not file_regex.search(fname):
                        continue
                    work_items.append((fname, [fname], strip_echo_suffix(fname)))

            for out_name, source_files, base in work_items:
                with tempfile.TemporaryDirectory(prefix="fw_qc_") as tmpd:
                    tmp_path = Path(tmpd)
                    try:
                        if combine_echoes:
                            local_echoes: list[Path] = []
                            for sf in source_files:
                                p = tmp_path / Path(sf).name
                                download_acquisition_file(fw, acq.id, sf, p)
                                local_echoes.append(p)
                            optcom_path = run_tedana_optcom(
                                echo_paths=local_echoes,
                                tes_ms=tes_ms,
                                out_dir=tmp_path / "tedana_out",
                                prefix=base,
                            )
                            analysis_path = optcom_path
                        else:
                            analysis_path = tmp_path / Path(source_files[0]).name
                            download_acquisition_file(fw, acq.id, source_files[0], analysis_path)

                        metrics, sd_map, brain_mask = compute_sd_metrics(
                            analysis_path,
                            min_mean=float(qc.get("mask_min_mean", 1e-6)),
                        )
                        artifact_metrics = detect_horizontal_line_artifact(
                            sd_map=sd_map,
                            mask=brain_mask,
                            min_line_width=3,
                            gradient_threshold=2.0,
                        )

                        if viz_dir is not None:
                            viz_name = f"{subject_label_from_session(ses)}_{ses.label}_{Path(out_name).stem}_outliers.png"
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
                                "file_name": out_name,
                                "source_echo_files": "|".join(source_files),
                                "analysis_type": "tedana_optcom" if combine_echoes else "single_echo",
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
                        "task_label": infer_task_label(source_files[0], str(_obj_get(acq, "label", "")), str(_obj_get(ses, "label", ""))),
                        "acquisition_label": acq.label,
                        "file_name": out_name,
                        "source_echo_files": "|".join(source_files),
                        "analysis_type": "tedana_optcom" if combine_echoes else "single_echo",
                        "status": "ok",
                        "error": "",
                    }
                )
                row.update(artifact_metrics)
                rows.append(row)
                n_processed += 1
                print(f"processed {n_processed}: {ses.label} | {acq.label} | {out_name}")

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
        "acquisition_label",
        "file_name",
        "source_echo_files",
        "analysis_type",
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
