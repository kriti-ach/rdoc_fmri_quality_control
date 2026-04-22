#!/bin/bash
#SBATCH --job-name=tedana_qc_s23_ses4
#SBATCH --output=/scratch/users/kritiach/rdoc_fmri_quality_control/scripts/logs/tedana_qc_%j.out
#SBATCH --error=/scratch/users/kritiach/rdoc_fmri_quality_control/scripts/logs/tedana_qc_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=russpold,normal

set -euo pipefail

REPO_ROOT=/scratch/users/kritiach/rdoc_fmri_quality_control
LOG_DIR="${REPO_ROOT}/scripts/logs"

# Override these with `sbatch --export=...` if needed.
SUBJECT_LABEL="${SUBJECT_LABEL:-s23}"
SESSION_RANK="${SESSION_RANK:-6}"
FILE_REGEX="${FILE_REGEX:-_e[123]\\.nii(\\.gz)?$}"
ECHOES="${ECHOES:-1,2,3}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

# Ensure uv is available and dependencies are synced in this environment.
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found in PATH. Load/install uv before running this job." >&2
  exit 1
fi

uv sync

echo "Running tedana-optcom QC:"
echo "  subject=${SUBJECT_LABEL} session_rank=${SESSION_RANK} echoes=${ECHOES}"
echo "  file_regex=${FILE_REGEX}"

uv run python scripts/run_flywheel_qc.py \
  --subject-label "${SUBJECT_LABEL}" \
  --session-rank "${SESSION_RANK}" \
  --combine-echoes \
  --echoes "${ECHOES}" \
  --file-regex "${FILE_REGEX}" \
  --save-visualizations

echo "Done."
