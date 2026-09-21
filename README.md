# UNet++ in nnU-Net v2, benchmarked on PanTS

Porting the UNet++ architecture — which only ever existed for nnU-Net **v1** — into nnU-Net **v2**,
then training and evaluating it on the official [PanTS](https://github.com/MrGiovanni/PanTS)
pancreatic tumour dataset. The goal is the five benchmark metrics (P-Sen, T-Sen, Spe, AUC, DSC) for
the currently-blank "UNet++" row in PanTS's public benchmark table.

Assigned by Prof. Zongwei Zhou (JHU) to Engho Lam (CUHK medical student) during work in Prof. Tu's lab.

> **Project is paused.** GPU access resumes February 2027. Read **[HANDOFF.md](HANDOFF.md)** first —
> it records exactly where things stood, what is finished, what is not, and what to do on day one.
> **[FINDINGS.md](FINDINGS.md)** records the two training defects diagnosed here and a fidelity audit
> of the port against both the UNet++ paper and the official v1 implementation.

---

## What is actually being compared

A 2×2 grid — architecture against deep supervision — so the effect of each can be separated:

|                  | Deep supervision ON | Deep supervision OFF |
|------------------|---------------------|----------------------|
| **Plain U-Net** (nnU-Net architecture) | `nnUNetTrainerBF16` | `nnUNetTrainerBF16NoDeepSupervision` |
| **UNet++** (this port)            | `nnUNetTrainerUNetPlusPlus` | `nnUNetTrainerUNetPlusPlusNoDeepSupervision` |

The two plain U-Net runs keep the stock nnU-Net architecture and training mathematics and act as
the control group. All grid arms use the same BF16 autocast and complete-loss compilation execution
optimizations, so precision/runtime plumbing does not become an architecture confound.

### A fifth, optional run: `nnUNetTrainerUNetPlusPlusPaper`

The four runs above use nnU-Net's deep supervision conventions. `nnUNetTrainerUNetPlusPlusPaper`
instead follows the **UNet++ paper's** design, which differs in two coupled ways:

| | Loss weighting | Inference |
|---|---|---|
| `nnUNetTrainerUNetPlusPlus` (the grid) | nnU-Net's exponential decay | deepest branch only |
| `nnUNetTrainerUNetPlusPlusPaper` | equal (η ≡ 1), every branch | **average of all branches** |

It is not part of the 2×2 grid — it answers a separate question, namely whether the paper's own
scheme beats nnU-Net's on this dataset. Run it against `nnUNetTrainerUNetPlusPlus` on the same fold
and plans, and compare **tumour-class** metrics.

> **Do not apply the equal weighting without the branch averaging.** That exact half-measure was
> tried and failed badly — 0% of epochs detected any tumour through epoch 500, versus 87% for the
> otherwise identical default run. The two rules are a matched pair; see [FINDINGS.md](FINDINGS.md).

---

## Repository layout

| Directory | Contents |
|---|---|
| `unetpp_port/` | The port itself — network, trainers, smoke test. **Copy these into nnU-Net v2 on the server.** |
| `data_conversion/` | PanTS → nnU-Net format. Merges 28 per-organ masks into one labelled volume per case. |
| `training/` | Launch scripts and a progress checker for the four training runs. |
| `evaluation/` | Post-training pipeline: validation → test prediction → the five metrics. |
| `archive/` | Superseded scripts kept for provenance. Do not use; see `archive/README.md`. |
| `docs/` | Full chronological project log and results. |
| `scripts/` | Utility for capturing server state into this repo. |

---

## Setup from scratch

Assumes a Linux GPU server with NVIDIA GPUs and conda. The reference machine was
`wenyuan.ucsd.edu` with 8 × RTX A5000 (24 GB each), 503 GB RAM, 64 CPU cores.

### 1. Copy this repository to the server

```bash
scp -r pants-unetpp <user>@<server>:~/
```

### 2. Environment and nnU-Net v2

```bash
conda create -n pants_unetpp python=3.10 -y
conda activate pants_unetpp

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet && pip install -e . && cd ..
pip install nibabel scipy scikit-learn blosc2
```

### 3. Install the port into nnU-Net

nnU-Net discovers trainers by class name, so these files must live inside its own tree:

```bash
cp ~/pants-unetpp/unetpp_port/unet_plusplus.py                            nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerBF16Mixin.py                   nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerFullLossCompileMixin.py        nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerQualityNeutralOptimizationMixin.py nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerSparseValidationMixin.py       nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerBF16.py                        nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerBF16NoDeepSupervision.py       nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerUNetPlusPlus.py                nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerUNetPlusPlusNoDeepSupervision.py nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainerUNetPlusPlusPaper.py           nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/pants-unetpp/unetpp_port/nnUNetTrainer*SparseValidation.py           nnUNet/nnunetv2/training/nnUNetTrainer/
```

For final fixed-epoch comparisons, the `...SparseValidation` variants run the same training steps
but perform online validation every five epochs (and always on the final epoch). Use the final
checkpoint for comparisons. The variants exist for all four grid cells and the paper configuration.

### 4. Verify the port before touching real data

```bash
cd ~/pants-unetpp/unetpp_port && python smoke_test.py
```

Must print `All smoke tests passed.` — it runs dummy tensors through the network at several stage
counts, with and without deep supervision, in a few seconds on CPU. If this fails, stop and fix it;
everything downstream is wasted otherwise.

### 5. Required patch to nnU-Net itself

Multi-GPU training of the **deep-supervision-off** trainers fails without this. The network always
builds all deep-supervision output heads, but with DS off only the final head contributes to the
loss — so PyTorch's distributed wrapper aborts over "unused parameters".

```bash
sed -i 's/self.network = DDP(self.network, device_ids=\[self.local_rank\])/self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)/' \
  nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
grep -n "DDP(" nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py   # confirm the flag is there
```

This affects gradient bucketing and a little speed — not the training mathematics.

### 6. nnU-Net environment variables

Point these at a disk with **1 TB+ free** (raw ≈ 330 GB, preprocessed ≈ 800 GB):

```bash
export nnUNet_raw="/Scratch/<user>/nnUNet_raw"
export nnUNet_preprocessed="/Scratch/<user>/nnUNet_preprocessed"
export nnUNet_results="/Scratch/<user>/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
# append the three export lines to ~/.bashrc so they survive logout
```

### 7. Download PanTS (~300 GB, slow)

```bash
cd /Scratch/<user>
git clone https://github.com/MrGiovanni/PanTS.git && cd PanTS
nohup bash download_PanTS_data.sh  > download_data.log  2>&1 &
# only after that finishes:
nohup bash download_PanTS_label.sh > download_label.log 2>&1 &
```

### 8. Convert to nnU-Net format

```bash
nohup python ~/pants-unetpp/data_conversion/convert_pants_to_nnunet.py \
  --pants-root /Scratch/<user>/PanTS/data \
  --nnunet-dataset-dir "$nnUNet_raw/Dataset001_PanTS" \
  --test-answer-key-dir /Scratch/<user>/PanTS_test_answer_key \
  > convert.log 2>&1 &
```

Test-set labels deliberately go to a **separate** directory. nnU-Net never needs them, but scoring
does. There is no `labelsTs` — a mistake that cost real time once already.

### 9. Preprocess (~30 hours)

```bash
nohup nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity \
  > preprocess.log 2>&1 &
```

### 10. Create the batch-size-4 plans file

**This step is not optional** — see [FINDINGS.md](FINDINGS.md). At the planner's default batch size
of 2 the models never learn the tumour class at all.

```bash
python3 -c "
import json
p='$nnUNet_preprocessed/Dataset001_PanTS/nnUNetPlans.json'
plans=json.load(open(p))
plans['plans_name']='nnUNetPlansBS4'
plans['configurations']['3d_fullres']['batch_size']=4
out='$nnUNet_preprocessed/Dataset001_PanTS/nnUNetPlansBS4.json'
json.dump(plans, open(out,'w'), indent=2)
print('wrote', out)
"
```

### 11. If you are not `enl014` on `wenyuan.ucsd.edu`

The scripts were written against one specific server and carry hardcoded paths — `/Scratch/enl014/…`
and `~/pants_phase7_scoring`. They were deliberately **not** parameterised before the project paused,
because rewriting working-but-untested scripts is how things quietly break.

Adapt them before first use:

```bash
grep -rln "enl014\|pants_phase7_scoring" .          # every file that needs attention
```

Most occurrences are in docstring examples and are harmless. The ones that actually execute:

| File | What to change |
|---|---|
| `training/retrain_wave1.sh` | `cd ~/pants_phase7_scoring` — the working directory |
| `training/queue_default_nods.sh` | working directory and `RESULTS=` |
| `training/check_tumor_progress.py` | `RESULTS` constant at the top |
| `evaluation/post_train_scheduler.sh` | `RESULTS`, working directory |
| `evaluation/post_train_one_run.sh` | `RAW`, `RESULTS`, `TE_GT`, working directory |
| `scripts/capture_server_state.sh` | honours `$nnUNet_*` env vars; only the answer-key path is fixed |

The Python scripts in `evaluation/` and `data_conversion/` take all paths as command-line arguments
and need no editing.

---

## Usage

### Train

UNet++ needs ~17 GB per patch on a 24 GB card, so batch size 4 means **4 GPUs, one patch each**.
The plain U-Net needs ~5.7 GB per patch. With 8 GPUs the four runs go in two waves.

```bash
cd ~/pants-unetpp/training
./retrain_wave1.sh                                    # both UNet++ runs, 4 GPUs each
nohup ./queue_default_nods.sh > queue.log 2>&1 &      # auto-launches the last run when a slot frees
```

Each run is 1000 epochs; budget **roughly 40–60 hours per run**, and more if the machine is shared.

### Check progress

```bash
python3 ~/pants-unetpp/training/check_tumor_progress.py
```

Prints tumour Dice per 100-epoch window against a reference trajectory from a known-good run.

> **Do not judge a run before epoch ~500.** Tumour learning has a delayed onset — a healthy run shows
> exactly 0% tumour detection for its first 300 epochs and is indistinguishable from total failure.
> This makes short pilot runs useless for validating any configuration change.

### Evaluate

```bash
cd ~/pants-unetpp/evaluation
nohup ./post_train_scheduler.sh > scheduler.log 2>&1 &
```

Watches for finished models and dispatches each to a free GPU for the full pipeline: cross-validation
validation → prediction on the 901 official test cases → all five metrics. Results land in
`metrics_PanTSte_<run>_bs4.json`.

Validation runs **single-GPU on purpose**. Under multi-GPU it repeatedly died on NCCL timeouts,
because validation cases vary wildly in duration and the GPUs drift out of sync past the watchdog
limit.

---

## Gotchas that cost real time

1. **Always use `nohup`.** An SSH disconnect once killed a foreground script mid-write and destroyed
   probability data for 346 cases. Every server command, no matter how short.
2. **Batch size 2 silently fails.** Overall Dice looks healthy (~0.67) because 27 large organs
   dominate the average, while tumour Dice is ~0.000. Always check the tumour class specifically.
3. **Memory is RAM, not VRAM.** Running several evaluation pipelines at once exhausts *system* RAM —
   each case's 29-channel probability array is several GB, held by CPU export workers. Keep
   `nnUNet_def_n_proc=2` and avoid running more than two pipelines concurrently.
4. **`labelsTs` does not exist.** Score the test set against the separate answer-key directory.
5. **Post-training validation crashing is expected** under multi-GPU and costs nothing — checkpoints
   are written first. Re-run it single-GPU with `--val`.
6. **Match torch and torchvision versions.** A mismatch produced a confusing
   `operator torchvision::nms does not exist` failure. torch 2.6.0+cu124 pairs with torchvision 0.21.0.

---

## Documents

| File | What it covers |
|---|---|
| [HANDOFF.md](HANDOFF.md) | Where things stood, what to do first when resuming |
| [FINDINGS.md](FINDINGS.md) | The two defects, the fidelity audit, open questions for the professor |
| [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) | Full chronological lab notebook |
| [unetpp_port/README.md](unetpp_port/README.md) | Technical explanation of the port |
