#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --job-name=pants_layout_new
#SBATCH --output=/projects/bdyo/asanjeev/gh200_layout_new_%j.log

set -euo pipefail

module load python/miniforge3_pytorch/2.12.0 >/dev/null 2>&1
conda activate base

repo=/projects/bdyo/asanjeev/pants-unetpp-fork
run_root=/projects/bdyo/asanjeev/gh200_layout_new_${SLURM_JOB_ID}
mkdir -p "${run_root}"

export PYTHONPATH="/u/asanjeev/.local/lib/python3.12/site-packages:${repo}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

probe=${repo}/unetpp_port/gh200_kernel_screen.py
comparator=${repo}/unetpp_port/compare_training_states.py

hostname
nvidia-smi -L
taskset -pc $$
numactl --show
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())'

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

for arch in unetpp plainunet; do
    for layout in contiguous channels_last_3d; do
        extra=()
        if [[ ${layout} == channels_last_3d ]]; then
            extra+=(--channels-last-3d)
        fi
        run_arm "${arch}_${layout}" python "${probe}" \
            --arch "${arch}" --compile-mode reduce-overhead --cudnn-benchmark \
            --warmup 5 --steps 15 --state-output "${run_root}/${arch}_${layout}.pt" \
            "${extra[@]}"
    done
    run_arm "compare_${arch}" python "${comparator}" \
        "${run_root}/${arch}_contiguous.pt" "${run_root}/${arch}_channels_last_3d.pt"
done

printf 'LAYOUT_SCREEN_COMPLETE run_root=%s\n' "${run_root}"
