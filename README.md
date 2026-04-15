# rdoc_fmri_quality_control

Config-driven QC pipeline to quantify temporal-standard-deviation (temporal SD) outliers in fMRI scans.

## What it quantifies

For each 4D NIfTI it computes a temporal SD map and summary metrics that can be used to measure:

- **Extent**: how extreme values are (`sd_p99`, `sd_max`, `robust_zmax`)
- **Prevalence in a scan**: fraction of brain voxels above robust thresholds (`frac_z_ge_6`, `frac_z_ge_8`)
- **Prevalence across subjects/scans**: aggregated CSV rows from all scans on Flywheel

## Repository layout

- `config/config.example.yaml` - all paths, Flywheel project/group, and run settings
- `scripts/run_single_scan_qc.py` - run QC for one local NIfTI first
- `scripts/run_flywheel_qc.py` - iterate Flywheel files and write one metrics CSV
- `src/rdoc_fmri_quality_control/temporal_sd.py` - core metrics implementation
- `batch/run_mriqc.sbatch` - MRIQC participant run for one scan (no trimming)

## Setup

```bash
cd /path/to/rdoc_fmri_quality_control
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy and edit config:

```bash
cp config/config.example.yaml config/config.yaml
```

## 1) Test on one scan

```bash
python scripts/run_single_scan_qc.py --config config/config.yaml
```

Outputs:

- `<prefix>_temporal_sd.nii.gz`
- `<prefix>_temporal_sd_mid_sag.png`
- `<prefix>_metrics.json`

## 2) Run across Flywheel project

Set API key in config (or template it with your environment tooling), then:

```bash
python scripts/run_flywheel_qc.py --config config/config.yaml
```

### Filtered Flywheel runs (subject/session/file pattern)

`run_flywheel_qc.py` can now target a specific subject/session rank and filename pattern directly from Flywheel:

```bash
python scripts/run_flywheel_qc.py \
  --config config/config.yaml \
  --subject-label s23 \
  --session-rank 6 \
  --file-regex '_e2\.nii(\.gz)?$'
```

Notes:
- Session rank is 1-based chronological order within the selected subject:
  - 1 = prelim, 2 = session0, 3 = session1, ...
  - so session4 is rank 6.
- `--file-regex '_e2\.nii(\.gz)?$'` keeps only e2 scans.

### Optional line-area ROI metrics

If you have a 3D ROI mask in the same voxel space as the BOLD data, add:

```bash
python scripts/run_flywheel_qc.py \
  --config config/config.yaml \
  --line-roi-mask /path/to/line_area_mask.nii.gz
```

This appends ROI-specific columns (`roi_sd_*`, `roi_robust_zmax`, `roi_frac_z_ge_*`) to the output CSV.

For a quick smoke test:

```bash
python scripts/run_flywheel_qc.py --config config/config.yaml --limit 5
```

Output:

- `flywheel_temporal_sd_metrics.csv`

Each row contains scanner/file IDs plus SD metrics and robust outlier prevalence.

## Suggested first analyses

- Histogram of `robust_zmax` across scans
- Proportion of scans with `frac_z_ge_8` above a threshold (e.g., 0.005)
- Compare `sd_p99` by sequence/protocol metadata

## MRIQC note

The provided `batch/run_mriqc.sbatch` now runs only the original scan (no TR trimming).
