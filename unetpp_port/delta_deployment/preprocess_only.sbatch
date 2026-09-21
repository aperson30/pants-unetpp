#!/bin/bash
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --job-name=pants_preprocess
#SBATCH --output=/projects/bdyo/asanjeev/delta_work/preprocess_%j.log

set -e
module load pytorch-conda/2.12 >/dev/null 2>&1
cd /projects/bdyo/asanjeev/delta_work
source pants_venv/bin/activate

export nnUNet_raw=/projects/bdyo/asanjeev/delta_work/nnUNet_raw
export nnUNet_preprocessed=/projects/bdyo/asanjeev/delta_work/nnUNet_preprocessed
export nnUNet_results=/projects/bdyo/asanjeev/delta_work/nnUNet_results

echo "=== plan + preprocess (16 workers) ==="
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity -npfp 16 -np 16

echo "=== create BS4 plan variant ==="
python3 -c "
import json
p='$nnUNet_preprocessed/Dataset001_PanTS/nnUNetPlans.json'
plans=json.load(open(p))
plans['plans_name']='nnUNetPlansBS4'
plans['configurations']['3d_fullres']['batch_size']=4
out='$nnUNet_preprocessed/Dataset001_PanTS/nnUNetPlansBS4.json'
json.dump(plans, open(out,'w'), indent=2)
print('wrote', out)
print('patch_size:', plans['configurations']['3d_fullres']['patch_size'])
"
echo "ALL_PHASES_DONE"
