#!/bin/bash
#SBATCH --account=bdyo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --constraint=nvperf
#SBATCH --gpus=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --job-name=pants_nsys
#SBATCH --output=/projects/bdyo/asanjeev/gh200_nsys_%j.log

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
source /projects/bdyo/asanjeev/pants_venv/bin/activate

repo=/projects/bdyo/asanjeev/pants-unetpp-fork
run_root=/projects/bdyo/asanjeev/gh200_nsys_${SLURM_JOB_ID}
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
report=${run_root}/unetpp_ds_on

hostname
nvidia-smi -L
taskset -pc $$
numactl --show

timeout 12m srun --ntasks=1 --cpu-bind=cores --mem-bind=local \
    nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,osrt --cuda-memory-usage=true \
    --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=10000 \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="${report}" \
    python3 "${probe}" --trainer unetpp_ds_on --plans "${plans}" \
    --dataset-json "${dataset}" --cudnn-benchmark --warmup 8 --steps 5 \
    --cuda-profiler --output "${run_root}/timing.json"

nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
    --format csv --output . "${report}.nsys-rep"

printf 'PROFILE_COMPLETE run_root=%s report=%s.nsys-rep\n' "${run_root}" "${report}"
