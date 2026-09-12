# Handoff

**Paused:** September 2026. **Resuming:** February 2027, when JHU GPU access reopens.
**Last verified server state:** 3 September 2026 (see *Before you lose access* — this needs a final
capture).

Written for whoever picks this up — possibly a different student, possibly the same one five months
later. Read [README.md](README.md) for what the project is and how to run it; this document is about
*where things stood* and *what to do first*.

---

## One-paragraph summary

The port works. All four models in the 2×2 comparison have been trained to completion (1000 epochs
each) and all four demonstrably learn the tumour class. What is **not** finished is evaluation: the
901-case official test predictions are partially complete for two runs and not started for the other
two. An earlier UNet++ run already produced a full set of publishable metrics (DSC 0.421), so there
is a real result in hand even if nothing further happens.

---

## Status by phase

| Phase | Status | Notes |
|---|---|---|
| 0–1 Server + nnU-Net v2 setup | Done | Conda env `pants_unetpp`; may not survive to February |
| 2 PanTS download + conversion | Done | ~330 GB raw. **At risk of scratch purge — see below** |
| 3 UNet++ port to v2 | Done | Smoke-tested; two real bugs found and fixed during porting |
| 4 Preprocessing | Done | ~30 hours; ~800 GB. **Also at risk of purge** |
| 5 Train the 2×2 grid | **Done** | All four at 1000 epochs, batch size 4, tumour learning confirmed |
| 6 Predict on official test set | **Partial** | 2 of 4 runs partially predicted; 2 not started |
| 7 Report the five metrics | **Partial** | Preliminary numbers only; see FINDINGS.md |

---

## Where things live on the server

`ssh enl014@wenyuan.ucsd.edu`, conda env `pants_unetpp`.

| Path | What |
|---|---|
| `/Scratch/enl014/nnUNet_raw/Dataset001_PanTS` | Converted data (~330 GB) |
| `/Scratch/enl014/nnUNet_preprocessed/Dataset001_PanTS` | Preprocessed (~800 GB), plus `nnUNetPlansBS4.json` |
| `/Scratch/enl014/nnUNet_results/Dataset001_PanTS` | All four trained models + logs |
| `/Scratch/enl014/PanTS_test_answer_key/` | **Test-set ground truth** (901 files). There is no `labelsTs` |
| `/Scratch/enl014/PanTS_te_predictions_*_bs4/` | Test predictions, partial |
| `/Scratch/enl014/BACKUP_original_unetpp_bs4_run/` | Checkpoint + log of the earlier successful run |
| `~/pants_phase7_scoring/` | Working directory where scripts were run |
| `~/unetpp_reference/` | Downloaded copies of the official v1 UNet++ files, for the fidelity audit |

Trainer files live inside nnU-Net's own tree at
`~/nnUNet/nnunetv2/training/nnUNetTrainer/` — and note that `nnUNetTrainer.py` there carries the
`find_unused_parameters=True` patch described in the README.

---

## ⚠ Before you lose access — do this now, not in February

Five months is long enough for a scratch filesystem to be purged, a conda environment to be
rebuilt, or an account to be cleaned up. **The trained models are the expensive, irreplaceable
part** — roughly a week of GPU time across four runs. Everything else can be regenerated.

Priority order:

1. **The four checkpoints** (~366 MB each, ~1.5 GB total). Copy somewhere durable — not `/Scratch`.
   ```bash
   mkdir -p ~/pants_models_backup
   for t in nnUNetTrainer nnUNetTrainerNoDeepSupervision nnUNetTrainerUNetPlusPlus nnUNetTrainerUNetPlusPlusNoDeepSupervision; do
     d=/Scratch/enl014/nnUNet_results/Dataset001_PanTS/${t}__nnUNetPlansBS4__3d_fullres/fold_0
     mkdir -p ~/pants_models_backup/$t
     cp -a "$d"/checkpoint_final.pth "$d"/training_log_*.txt "$d"/*.json ~/pants_models_backup/$t/ 2>/dev/null
   done
   du -sh ~/pants_models_backup
   ```
   Plus the earlier successful run, already at `/Scratch/enl014/BACKUP_original_unetpp_bs4_run/` —
   move that off `/Scratch` too.

2. **The small text artefacts** into this repository, so they survive in git:
   ```bash
   bash ~/pants-unetpp/scripts/capture_server_state.sh
   ```
   Then `scp` the generated `docs/results/` back to your laptop and commit it.

3. **Ask about the scratch retention policy.** If `/Scratch` is purged, the 330 GB download and the
   30-hour preprocessing both have to be redone before anything can run. Knowing this in advance
   changes what February looks like.

---

## Day one, February

### 1. Check what survived

```bash
ssh enl014@wenyuan.ucsd.edu
conda env list | grep pants_unetpp
ls /Scratch/enl014/nnUNet_raw/Dataset001_PanTS/imagesTr | wc -l      # expect 9000
ls /Scratch/enl014/PanTS_test_answer_key | wc -l                     # expect 901
ls /Scratch/enl014/nnUNet_results/Dataset001_PanTS/*/fold_0/checkpoint_final.pth
nvidia-smi
```

Branch on the result:

- **Everything intact** → go to step 2.
- **Checkpoints gone but backed up** → restore them into the results tree; skip retraining.
- **Scratch purged** → re-run README steps 6–10 first (download, convert, preprocess). Budget
  **roughly 3–4 days** before training can start, and more if the download is slow.

### 2. Finish the test-set predictions

This is the actual outstanding work. Two runs are partially predicted and two have not started.

```bash
cd ~/pants_phase7_scoring
export nnUNet_def_n_proc=2
export nnUNet_n_proc_DA=4
```

Then for each run — **one or two at a time, never four**, which is what exhausted RAM before:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python3 -u ~/pants-unetpp/evaluation/predict_and_shrink.py \
  --images-dir /Scratch/enl014/nnUNet_raw/Dataset001_PanTS/imagesTs \
  --output-dir /Scratch/enl014/PanTS_te_predictions_<run>_bs4 \
  --dataset 1 --config 3d_fullres -tr <TrainerClass> -p nnUNetPlansBS4 -f 0 \
  --max-probs-csv max_tumor_probs_te_<run>_bs4.csv --batch-size 10 \
  > predict_te_<run>_bs4.log 2>&1 &
disown
```

`--continue_prediction` is built in, so this resumes rather than restarting. Run/trainer pairs:

| `<run>` | `<TrainerClass>` |
|---|---|
| `default_ds` | `nnUNetTrainer` |
| `default_nods` | `nnUNetTrainerNoDeepSupervision` |
| `unetpp_ds` | `nnUNetTrainerUNetPlusPlus` |
| `unetpp_nods` | `nnUNetTrainerUNetPlusPlusNoDeepSupervision` |

Watch `free -g`; if available RAM drops below ~50 GB, drop to one job at a time.

### 3. Compute the five metrics

Once a run reaches 901/901:

```bash
python3 ~/pants-unetpp/evaluation/compute_tumor_metrics.py \
  --pred-dir /Scratch/enl014/PanTS_te_predictions_<run>_bs4 \
  --labels-dir /Scratch/enl014/PanTS_test_answer_key \
  --tumor-class 28 \
  --probs-csv max_tumor_probs_te_<run>_bs4.csv \
  --out-json metrics_PanTSte_<run>_bs4.json
```

Confirm `n_cases_evaluated` is **901**. Anything less means predictions are still incomplete — the
partial numbers already in FINDINGS.md came from exactly this trap.

### 4. Then what

With four complete metric sets you have the 2×2 comparison and can fill in the PanTS benchmark row.
Before reporting, get answers to the four open questions in [FINDINGS.md](FINDINGS.md) — particularly
**2-class versus 28-class**, which determines whether the numbers are comparable to the other
leaderboard rows at all. A 2-class answer means one more training round.

---

## Things that will waste your time if you don't know them

1. **A healthy run looks like a failed one for 300 epochs.** Tumour Dice is exactly 0.0 early on. Do
   not kill a run or judge a configuration change before epoch ~500. Use
   `training/check_tumor_progress.py`, which prints against a known-good reference trajectory.
2. **Never run four evaluation pipelines at once.** Each case's 29-channel probability array is
   several GB in *system* RAM, held by CPU export workers. Two at a time with
   `nnUNet_def_n_proc=2`.
3. **Validation must be single-GPU.** Multi-GPU validation repeatedly died on NCCL timeouts because
   cases vary wildly in duration. Use `--val` with no `-num_gpus`.
4. **Post-training validation crashing is normal** and costs nothing — checkpoints are written first.
5. **`nohup` everything.** A dropped SSH connection once destroyed 346 cases of probability data.
6. **The `find_unused_parameters=True` patch** to nnU-Net's own `nnUNetTrainer.py` is required for
   the DS-off trainers under multi-GPU. If nnU-Net is reinstalled, reapply it.
7. **Batch size must be 4.** At the planner's default of 2 the models silently never learn tumours.

---

## Open threads

- **Test predictions incomplete** — the main outstanding task (step 2 above).
- **Four questions for Prof. Zhou** unanswered — see FINDINGS.md. Worth emailing before February so
  the answers are waiting.
- **The official v1 fork appears to train one branch and predict with another** — documented with
  line numbers in FINDINGS.md, not yet raised with the professor.
- **Branch-averaging at inference** (what the UNet++ paper specifies) has never been tested here.
