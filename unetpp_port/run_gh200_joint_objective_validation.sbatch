#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --job-name=pants_joint_objective
#SBATCH --output=/projects/bdyo/asanjeev/gh200_joint_objective_%j.log

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
source /projects/bdyo/asanjeev/pants_venv/bin/activate

repo=/projects/bdyo/asanjeev/pants-unetpp-fork
run_root=/projects/bdyo/asanjeev/gh200_joint_objective_${SLURM_JOB_ID}
mkdir -p "${run_root}/nnUNet_results"

export PYTHONPATH="${repo}:${PYTHONPATH:-}"
export nnUNet_raw=/projects/bdyo/asanjeev/nnUNet_raw
export nnUNet_preprocessed=/projects/bdyo/asanjeev/nnUNet_preprocessed
export nnUNet_results="${run_root}/nnUNet_results"
export nnUNet_n_proc_DA=12
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

hostname
nvidia-smi -L
taskset -pc $$
numactl --show

kernel_probe=${repo}/unetpp_port/gh200_kernel_screen.py
real_probe=${repo}/unetpp_port/gh200_real_pipeline_probe.py
comparator=${repo}/unetpp_port/compare_training_states.py
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

# Controlled synthetic inputs produce complete parameter/gradient/momentum state comparisons.
for arch in unetpp plainunet; do
    for boundary in separate joint; do
        label=state_${arch}_${boundary}
        run_arm "${label}" python3 "${kernel_probe}" \
            --arch "${arch}" --compile-mode reduce-overhead \
            --compile-boundary "${boundary}" --cudnn-benchmark \
            --warmup 8 --steps 20 --state-output "${run_root}/${label}.pt"
    done
    run_arm "compare_${arch}" python3 "${comparator}" \
        "${run_root}/state_${arch}_separate.pt" "${run_root}/state_${arch}_joint.pt"
done

# Real nnU-Net trainers and augmented PanTS calibration patches establish throughput in every cell.
for trainer in unetpp_ds_on unetpp_ds_off plain_ds_on plain_ds_off; do
    for boundary in separate joint; do
        label=real_${trainer}_${boundary}
        run_arm "${label}" python3 "${real_probe}" \
            --trainer "${trainer}" --plans "${plans}" --dataset-json "${dataset}" \
            --compile-boundary "${boundary}" --cudnn-benchmark \
            --warmup 8 --steps 30 --output "${run_root}/${label}.json"
    done
done

printf 'VALIDATION_COMPLETE run_root=%s\n' "${run_root}"
