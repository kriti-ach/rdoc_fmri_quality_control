#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

import flywheel


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


def download_acquisition_file(fw: flywheel.Client, acq_id: str, file_name: str, out_path: Path) -> None:
    if hasattr(fw, "download_file_from_acquisition"):
        fw.download_file_from_acquisition(acq_id, file_name, str(out_path))
        return
    acq = fw.get_acquisition(acq_id)
    acq.download_file(file_name, str(out_path))


def find_t1_file(
    fw: flywheel.Client,
    group_id: str,
    project_label: str,
    subject_label: str,
    acq_label: str,
) -> tuple[str, str, str]:
    projects = fw.projects.find(f"group={group_id},label={project_label}")
    if not projects:
        raise RuntimeError(f"No project found for group={group_id}, label={project_label}")
    project = projects[0]

    sessions = fw.get_project_sessions(project.id)
    sessions = [s for s in sessions if subject_label_from_session(s).lower() == subject_label.lower()]
    if not sessions:
        raise RuntimeError(f"No sessions found for subject {subject_label}")

    for ses in sessions:
        for acq in fw.get_session_acquisitions(ses.id):
            if str(_obj_get(acq, "label", "")) != acq_label:
                continue
            for f in acq.files:
                name = f.name
                low = name.lower()
                if (low.endswith(".nii") or low.endswith(".nii.gz")) and "t1" in low:
                    return acq.id, name, ses.label
            for f in acq.files:
                name = f.name
                low = name.lower()
                if low.endswith(".nii") or low.endswith(".nii.gz"):
                    return acq.id, name, ses.label

    raise RuntimeError(
        f"No NIfTI found for subject={subject_label} in acquisition label '{acq_label}'."
    )


def run_flirt(t1_path: Path, out_dir: Path, mni_ref: Path) -> tuple[Path, Path]:
    out_nii = out_dir / f"{t1_path.stem.replace('.nii', '')}_in_MNI_affine.nii.gz"
    out_mat = out_dir / f"{t1_path.stem.replace('.nii', '')}_to_MNI_affine.mat"
    cmd = [
        "flirt",
        "-in",
        str(t1_path),
        "-ref",
        str(mni_ref),
        "-out",
        str(out_nii),
        "-omat",
        str(out_mat),
        "-dof",
        "12",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "flirt failed")
    return out_nii, out_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download subject T1w from Flywheel and run quick MNI coreg check.")
    parser.add_argument("--group-id", default="russpold", help="Flywheel group id")
    parser.add_argument("--project-label", default="rdoc_fmri", help="Flywheel project label")
    parser.add_argument("--subject-label", required=True, help="Subject label/code (e.g., s43)")
    parser.add_argument("--acq-label", default="NEW Sag_MPRAGE_T1", help="Acquisition label containing T1w")
    parser.add_argument(
        "--out-dir",
        default="outputs/t1w_coreg_check",
        help="Directory to store downloaded T1 and registration outputs",
    )
    parser.add_argument(
        "--mni-ref",
        default=None,
        help="Path to MNI reference NIfTI (default: $FSLDIR/data/standard/MNI152_T1_2mm.nii.gz)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mni_ref:
        mni_ref = Path(args.mni_ref).expanduser()
    else:
        fsldir = Path.cwd().anchor  # placeholder to satisfy type checker
        import os

        fsldir_env = os.environ.get("FSLDIR", "")
        if not fsldir_env:
            raise RuntimeError("FSLDIR not set. Pass --mni-ref explicitly or load FSL first.")
        mni_ref = Path(fsldir_env) / "data" / "standard" / "MNI152_T1_2mm.nii.gz"

    if not mni_ref.exists():
        raise RuntimeError(f"MNI reference not found: {mni_ref}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fw = flywheel.Client()
    acq_id, t1_name, session_label = find_t1_file(
        fw=fw,
        group_id=args.group_id,
        project_label=args.project_label,
        subject_label=args.subject_label,
        acq_label=args.acq_label,
    )

    local_t1 = out_dir / t1_name
    print(f"Downloading T1: {t1_name} (session {session_label})")
    download_acquisition_file(fw, acq_id, t1_name, local_t1)

    print("Running FLIRT (T1 -> MNI, 12 dof)...")
    out_nii, out_mat = run_flirt(local_t1, out_dir, mni_ref)

    print("\nDone.")
    print(f"Downloaded T1: {local_t1}")
    print(f"Registered T1: {out_nii}")
    print(f"Affine matrix:  {out_mat}")
    print("\nVisual QC command:")
    print(f'fsleyes "{mni_ref}" "{out_nii}"')


if __name__ == "__main__":
    main()

