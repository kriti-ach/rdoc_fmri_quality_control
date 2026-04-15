#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rdoc_fmri_quality_control.temporal_sd import (
    compute_sd_metrics,
    save_mid_sagittal_png,
    save_sd_nifti,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-scan temporal SD QC")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Optional override NIfTI path")
    parser.add_argument("--out-prefix", default=None, help="Optional override output prefix")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    qc = cfg["qc"]

    input_nifti = Path(args.input or qc["single_scan_nifti"])
    out_dir = Path(qc["single_scan_output_dir"])
    out_prefix = args.out_prefix or input_nifti.name.replace(".nii.gz", "")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, sd_map, _ = compute_sd_metrics(input_nifti, min_mean=float(qc.get("mask_min_mean", 1e-6)))

    sd_path = out_dir / f"{out_prefix}_temporal_sd.nii.gz"
    png_path = out_dir / f"{out_prefix}_temporal_sd_mid_sag.png"
    json_path = out_dir / f"{out_prefix}_metrics.json"

    save_sd_nifti(sd_map, input_nifti, sd_path)
    save_mid_sagittal_png(sd_map, input_nifti.name, png_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    print(f"input:   {input_nifti}")
    print(f"sd map:  {sd_path}")
    print(f"figure:  {png_path}")
    print(f"metrics: {json_path}")
    print("key prevalence stats:")
    print(f"  frac_z_ge_6={metrics.frac_z_ge_6:.6f}, frac_z_ge_8={metrics.frac_z_ge_8:.6f}")


if __name__ == "__main__":
    main()
