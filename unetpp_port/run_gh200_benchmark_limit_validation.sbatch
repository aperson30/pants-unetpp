#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --job-name=pants_cudnn_limit
#SBATCH --output=/projects/bdyo/asanjeev/gh200_cudnn_limit_%j.log

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
source /projects/bdyo/asanjeev/pants_venv/bin/activate

repo=/projects/bdyo/asanjeev/pants-unetpp-fork
run_root=/projects/bdyo/asanjeev/gh200_cudnn_limit_${SLURM_JOB_ID}
mkdir -p "${run_root}/nnUNet_results"
export PYTHONPATH="${repo}:${PYTHONPATH:-}"
export nnUNet_raw=/projects/bdyo/asanjeev/nnUNet_raw
export nnUNet_preprocessed=/projects/bdyo/asanjeev/nnUNet_preprocessed
export nnUNet_results="${run_root}/nnUNet_results"
export nnUNet_n_proc_DA=12
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

probe=${repo}/unetpp_port/gh200_real_pipeline_probe.py
plans=${nnUNet_preprocessed}/Dataset001_PanTSCalib/nnUNetPlansBS4.json
dataset=${nnUNet_raw}/Dataset001_PanTSCalib/dataset.json

run_arm() {
    label=$1
    shift
    set +e
    timeout 8m srun --ntasks=1 --cpu-bind=cores --mem-bind=local "$@" \
        > "${run_root}/${label}.log" 2>&1
    rc=$?
    set -e
    printf 'ARM_RESULT label=%s exit_code=%s log=%s\n' \
        "${label}" "${rc}" "${run_root}/${label}.log"
}

for trainer in unetpp_ds_on plain_ds_on; do
    for limit in 10 0; do
        label=${trainer}_limit_${limit}
        run_arm "${label}" python3 "${probe}" \
            --trainer "${trainer}" --plans "${plans}" --dataset-json "${dataset}" \
            --cudnn-benchmark --cudnn-benchmark-limit "${limit}" \
            --warmup 8 --steps 30 --output "${run_root}/${label}.json"
    done
done

printf 'LIMIT_VALIDATION_COMPLETE run_root=%s\n' "${run_root}"
