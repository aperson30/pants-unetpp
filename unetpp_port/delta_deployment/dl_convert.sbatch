#!/bin/bash
#SBATCH --account=bdyo-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --job-name=pants_dl_convert
#SBATCH --output=/projects/bdyo/asanjeev/delta_work/dl_convert_%j.log

set -e
cd /projects/bdyo/asanjeev/delta_work
mkdir -p PanTS/data/ImageTr PanTS/data/ImageTe
cd PanTS/data

echo "=== metadata ==="
wget -q -O metadata.xlsx "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true" || true

echo "=== downloading all 9 image chunks in parallel ==="
for i in 1 2 3 4 5 6 7 8 9; do
  start=$(printf "%08d" $(( (i - 1) * 1000 + 1 )))
  end=$(printf "%08d" $(( i * 1000 )))
  file="PanTSMini_ImageTr_${start}_${end}.tar.gz"
  url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${file}?download=true"
  (
    wget -q -O "$file" "$url" && \
    tar -xzf "$file" -C ImageTr && \
    rm "$file" && \
    echo "[$i/9] image chunk done"
  ) &
done

echo "=== downloading labels in parallel with images ==="
(
  curl -sL -f "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz" -o PanTSMini_Label.tar.gz && \
  mkdir -p LabelAll && \
  tar -xzf PanTSMini_Label.tar.gz -C LabelAll && \
  rm PanTSMini_Label.tar.gz && \
  echo "labels done"
) &

wait
echo "=== all downloads/extracts finished ==="

echo "=== organizing label splits ==="
mkdir -p LabelTr LabelTe
cd LabelAll
find . -maxdepth 1 -type d \( -name 'PanTS_0000[0-8]*' -o -name 'PanTS_00009000' \) -print0 | xargs -0 -r mv -t ../LabelTr/
find . -maxdepth 1 -type d -name 'PanTS_00009*' -print0 | xargs -0 -r mv -t ../LabelTe/
cd ..
rmdir LabelAll 2>/dev/null || true

echo "=== image test set ==="
file="PanTSMini_ImageTe_00009001_00009901.tar.gz"
url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/$file?download=true"
wget -q -O "$file" "$url" && tar -xzf "$file" -C ImageTe && rm "$file"

echo "=== counts ==="
ls ImageTr | wc -l
ls ImageTe | wc -l
ls LabelTr | wc -l
ls LabelTe | wc -l
du -sh /projects/bdyo/asanjeev/delta_work/PanTS
echo "DOWNLOAD_PHASE_DONE"

echo "=== convert to nnU-Net format ==="
module load pytorch-conda/2.12 >/dev/null 2>&1
cd /projects/bdyo/asanjeev/delta_work
source pants_venv/bin/activate

export nnUNet_raw=/projects/bdyo/asanjeev/delta_work/nnUNet_raw
export nnUNet_preprocessed=/projects/bdyo/asanjeev/delta_work/nnUNet_preprocessed
export nnUNet_results=/projects/bdyo/asanjeev/delta_work/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

python3 pants-unetpp-fork/data_conversion/convert_pants_to_nnunet.py \
  --pants-root PanTS/data \
  --nnunet-dataset-dir "$nnUNet_raw/Dataset001_PanTS" \
  --test-answer-key-dir /projects/bdyo/asanjeev/delta_work/PanTS_test_answer_key

echo "CONVERT_PHASE_DONE"
echo "DL_CONVERT_JOB_DONE"
