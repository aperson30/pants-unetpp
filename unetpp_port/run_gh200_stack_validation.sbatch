#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --job-name=pants_stack_validation
#SBATCH --output=/projects/bdyo/asanjeev/gh200_stack_validation_%j.log

set -euo pipefail

repo=/projects/bdyo/asanjeev/pants-unetpp-fork
run_root=/projects/bdyo/asanjeev/gh200_stack_validation_${SLURM_JOB_ID}
mkdir -p "${run_root}/nnUNet_results"

export nnUNet_raw=/projects/bdyo/asanjeev/nnUNet_raw
export nnUNet_preprocessed=/projects/bdyo/asanjeev/nnUNet_preprocessed
export nnUNet_results="${run_root}/nnUNet_results"
export nnUNet_n_proc_DA=12
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

real_probe=${repo}/unetpp_port/gh200_real_pipeline_probe.py
kernel_probe=${repo}/unetpp_port/gh200_kernel_screen.py
comparator=${repo}/unetpp_port/compare_training_states.py
plans=${nnUNet_preprocessed}/Dataset001_PanTSCalib/nnUNetPlansBS4.json
dataset=${nnUNet_raw}/Dataset001_PanTSCalib/dataset.json

hostname
nvidia-smi -L
taskset -pc $$
numactl --show

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

run_current_stack() {
    stack=$1
    python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())'
    for arch in unetpp plainunet; do
        run_arm "state_${arch}_${stack}" python "${kernel_probe}" \
            --arch "${arch}" --compile-mode reduce-overhead --cudnn-benchmark \
            --warmup 5 --steps 3 --state-output "${run_root}/state_${arch}_${stack}.pt"
    done
    for trainer in unetpp_ds_on unetpp_ds_off plain_ds_on plain_ds_off; do
        run_arm "real_${trainer}_${stack}" python "${real_probe}" \
            --trainer "${trainer}" --plans "${plans}" --dataset-json "${dataset}" \
            --cudnn-benchmark --warmup 8 --steps 20 \
            --output "${run_root}/real_${trainer}_${stack}.json"
    done
}

# Validated production stack.
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
source /projects/bdyo/asanjeev/pants_venv/bin/activate
export PYTHONPATH="${repo}:${PYTHONPATH:-}"
run_current_stack old
deactivate
module unload python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1

# Isolated candidate stack; no install and no production-venv mutation.
module load python/miniforge3_pytorch/2.12.0 >/dev/null 2>&1
conda activate base
export PYTHONPATH="/u/asanjeev/.local/lib/python3.12/site-packages:${repo}"
run_current_stack new

# Compare controlled states after eight identical updates. Comparator failures are recorded but do
# not prevent the other architecture from being checked.
for arch in unetpp plainunet; do
    run_arm "compare_${arch}" python "${comparator}" \
        "${run_root}/state_${arch}_old.pt" "${run_root}/state_${arch}_new.pt"
done

printf 'STACK_VALIDATION_COMPLETE run_root=%s\n' "${run_root}"
