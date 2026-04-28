#!/bin/bash
#SBATCH --job-name=fmriprep_s43
#SBATCH --output=/scratch/users/kritiach/rdoc_fmri_quality_control/scripts/logs/fmriprep_%j.out
#SBATCH --error=/scratch/users/kritiach/rdoc_fmri_quality_control/scripts/logs/fmriprep_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=russpold,normal

set -euo pipefail

REPO_ROOT=/scratch/users/kritiach/rdoc_fmri_quality_control
BIDS_ROOT="${REPO_ROOT}/bids"
DERIV_ROOT="${REPO_ROOT}/derivatives/fmriprep"
WORK_DIR="${REPO_ROOT}/work/fmriprep"
LOG_DIR="${REPO_ROOT}/scripts/logs"
SIF=/home/groups/russpold/singularity_images/fmriprep_25.2.4.sif

mkdir -p "${DERIV_ROOT}" "${WORK_DIR}" "${LOG_DIR}"

if [[ ! -d "${BIDS_ROOT}/sub-s43/ses-prelim" ]]; then
  echo "ERROR: expected BIDS directory not found: ${BIDS_ROOT}/sub-s43/ses-prelim" >&2
  exit 1
fi

if [[ ! -f "${SIF}" ]]; then
  echo "ERROR: fMRIPrep image not found: ${SIF}" >&2
  exit 1
fi

# Optional: use TemplateFlow cache on OAK if available.
export TEMPLATEFLOW_HOME=/oak/stanford/groups/russpold/data/templateflow

singularity exec --cleanenv \
  -B "${REPO_ROOT}:${REPO_ROOT}" \
  -B "/oak/stanford/groups/russpold:/oak/stanford/groups/russpold" \
  "${SIF}" \
  fmriprep "${BIDS_ROOT}" "${DERIV_ROOT}" participant \
    --participant-label s43 \
    --session-label prelim \
    --nthreads "${SLURM_CPUS_PER_TASK:-8}" \
    --omp-nthreads "${SLURM_CPUS_PER_TASK:-8}" \
    --mem-mb 48000 \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --stop-on-first-crash \
    -w "${WORK_DIR}"

echo "Done."
echo "Report: ${DERIV_ROOT}/fmriprep/sub-s43.html"
