#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --job-name=pants_opt_screen
#SBATCH --output=/projects/bdyo/asanjeev/gh200_opt_screen_%j.log

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
source /projects/bdyo/asanjeev/pants_venv/bin/activate

repo=/projects/bdyo/asanjeev/pants-unetpp-fork
run_root=/projects/bdyo/asanjeev/gh200_opt_screen_${SLURM_JOB_ID}
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
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

probe=${repo}/unetpp_port/gh200_real_pipeline_probe.py
kernel=${repo}/unetpp_port/gh200_kernel_screen.py
plans=${nnUNet_preprocessed}/Dataset001_PanTSCalib/nnUNetPlansBS4.json
dataset=${nnUNet_raw}/Dataset001_PanTSCalib/dataset.json

run_arm() {
    label=$1
    output_file=$2
    shift 2
    set +e
    timeout 8m srun --ntasks=1 --cpu-bind=cores --mem-bind=local "$@" \
        > "${output_file}" 2>&1
    rc=$?
    set -e
    printf 'ARM_RESULT label=%s exit_code=%s log=%s\n' "${label}" "${rc}" "${output_file}"
}

run_arm pipeline_unetpp "${run_root}/pipeline_unetpp.log" \
    python3 "${probe}" \
    --trainer unetpp_ds_on --plans "${plans}" --dataset-json "${dataset}" \
    --warmup 8 --steps 25 --output "${run_root}/pipeline_unetpp.json"

for mode in reduce-overhead max-autotune max-autotune-no-cudagraphs; do
    run_arm "kernel_unetpp_${mode}" "${run_root}/kernel_unetpp_${mode}.log" \
        python3 "${kernel}" --arch unetpp --compile-mode "${mode}" \
        --warmup 5 --steps 12
done

run_arm kernel_unetpp_reduce-overhead_cudnn-benchmark \
    "${run_root}/kernel_unetpp_reduce-overhead_cudnn-benchmark.log" \
    python3 "${kernel}" --arch unetpp --compile-mode reduce-overhead \
    --cudnn-benchmark --warmup 5 --steps 12

printf 'SCREEN_COMPLETE run_root=%s\n' "${run_root}"
