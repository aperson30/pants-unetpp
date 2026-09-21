#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --job-name=pants_stack_ab
#SBATCH --output=/projects/bdyo/asanjeev/gh200_stack_ab_%j.log

set -euo pipefail

repo=/projects/bdyo/asanjeev/pants-unetpp-fork
run_root=/projects/bdyo/asanjeev/gh200_stack_ab_${SLURM_JOB_ID}
mkdir -p "${run_root}/nnUNet_results"

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

hostname
nvidia-smi -L
taskset -pc $$
numactl --show

run_probe() {
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

# Validated production stack. The venv inherits torch from the matching system module.
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
source /projects/bdyo/asanjeev/pants_venv/bin/activate
export PYTHONPATH="${repo}:${PYTHONPATH:-}"
python -c 'import torch; print("OLD_STACK", torch.__version__, torch.version.cuda, torch.backends.cudnn.version())'
run_probe old_stack python "${probe}" \
    --trainer unetpp_ds_on --plans "${plans}" --dataset-json "${dataset}" \
    --cudnn-benchmark --warmup 8 --steps 30 --output "${run_root}/old_stack.json"
deactivate
module unload python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1

# Isolated system module: no package install and no mutation of the validated production venv.
module load python/miniforge3_pytorch/2.12.0 >/dev/null 2>&1
conda activate base
export PYTHONPATH="/u/asanjeev/.local/lib/python3.12/site-packages:${repo}"
python -c 'import torch, nnunetv2; print("NEW_STACK", torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), nnunetv2.__path__)'
run_probe new_stack python "${probe}" \
    --trainer unetpp_ds_on --plans "${plans}" --dataset-json "${dataset}" \
    --cudnn-benchmark --warmup 8 --steps 30 --output "${run_root}/new_stack.json"

printf 'STACK_AB_COMPLETE run_root=%s\n' "${run_root}"
