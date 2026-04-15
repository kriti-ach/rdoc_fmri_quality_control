#!/bin/bash
#SBATCH --job-name=mriqc_s23_gonogo
#SBATCH --output=/scratch/users/kritiach/rdoc_fmri_quality_control/scripts/logs/mriqc_%j.out
#SBATCH --error=/scratch/users/kritiach/rdoc_fmri_quality_control/scripts/logs/mriqc_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=russpold,normal

set -euo pipefail

OAK_BASE=/oak/stanford/groups/russpold/data/rdoc_mriqc_test
SIF=/home/groups/russpold/singularity_images/mriqc_24.0.2.sif

DATA_DIR="${OAK_BASE}/data"
DERIV_ROOT="${OAK_BASE}/derivatives"
WORK_DIR="${OAK_BASE}/work/mriqc"

NII=32621_6_1_ses-04_RDoC_fmri_20260202123240_6_e2.nii.gz
SUB_LABEL=23
SES_TAG=ses-04
TASK=goNogo
RUN=1

BIDS_ROOT="${OAK_BASE}/bids"
OUT_DIR="${DERIV_ROOT}/mriqc"
BIDS_NAME="sub-${SUB_LABEL}_${SES_TAG}_task-${TASK}_run-${RUN}_bold.nii.gz"
SUB_DIR="sub-${SUB_LABEL}/${SES_TAG}/func"

mkdir -p "${BIDS_ROOT}/${SUB_DIR}" "${OUT_DIR}" "${WORK_DIR}"

if [[ ! -f "${DATA_DIR}/${NII}" ]]; then
  echo "ERROR: missing ${DATA_DIR}/${NII}" >&2
  exit 1
fi

ln -sfn "${DATA_DIR}/${NII}" "${BIDS_ROOT}/${SUB_DIR}/${BIDS_NAME}"

if [[ ! -f "${BIDS_ROOT}/dataset_description.json" ]]; then
  cat > "${BIDS_ROOT}/dataset_description.json" << 'EOF'
{
  "Name": "rdoc_fmri",
  "BIDSVersion": "1.9.0",
  "DatasetType": "raw"
}
EOF
fi

singularity exec --cleanenv \
  -B "${OAK_BASE}:${OAK_BASE}" \
  "${SIF}" \
  mriqc "${BIDS_ROOT}" "${OUT_DIR}" participant \
    --no-sub \
    --n_procs "${SLURM_CPUS_PER_TASK:-8}" \
    -w "${WORK_DIR}"

echo "Done. Outputs in ${OUT_DIR}"
find "${OUT_DIR}" -name '*.html' -print 2>/dev/null | sort || true
