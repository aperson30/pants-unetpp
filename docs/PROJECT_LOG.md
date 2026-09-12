# Project log: UNet++ in nnU-Net v2, benchmarked on PanTS

This is a running record of the professor's Task 2 assignment (see also Task 1 / ShapeKit, parked for
now). Keep this updated as things change — treat it like a lab notebook, not a one-time document.

## STANDING RULE: always use `nohup`, no matter how short or trivial the command seems
Learned this the hard way: an SSH disconnect killed a foreground `python extract_max_probs.py` run
partway through (no `nohup`), which combined with a bug in that script (deleting each original file
before its extracted value was saved to disk, rather than after) **permanently destroyed the
probability data for 346 test cases**, requiring a full re-prediction to recover. From now on: every
command run on the server — no exceptions, including ones that "should" finish in seconds — gets
wrapped in `nohup ... > file.log 2>&1 &`. A dropped connection must never be able to kill work in
progress or leave data half-written.

## The assignment, in one sentence
Port the UNet++ architecture (only ever built for the old nnU-Net v1) into nnU-Net v2, train it on the
official PanTS training set, test it on the official PanTS test set, and report P-Sen / T-Sen / Spe /
AUC / DSC — filling in the currently-blank "UNet++" row in PanTS's own public benchmark table
(github.com/MrGiovanni/PanTS README).

## Where the code lives
- `~/Downloads/unetplusplus_nnunetv2_port/` — the ported architecture.
  - `unet_plusplus.py` — the network itself.
  - `nnUNetTrainerUNetPlusPlus.py` — tells nnU-Net v2 to use it instead of its default network.
  - `smoke_test.py` — fast sanity check, run this before trusting anything else.
  - `README.md` — technical explanation of the port + the OOM risk to watch for.
- `~/Downloads/pants_nnunet_conversion/` — the PanTS → nnU-Net data conversion script.
  - `convert_pants_to_nnunet.py` — merges PanTS's 28 per-organ label files into one combined label
    per case, and sorts everything into nnU-Net's expected folder layout.

## Key facts/decisions established so far (don't re-derive these from scratch later)
- **PanTS official split** is baked into folder/case-ID structure, not a separate split file:
  `ImageTr/PanTS_00000001`…`00009000` = the 9,000 training cases; `ImageTe/PanTS_00009001`…`00009901`
  = the 901 official test cases.
- **Confirmed required metrics** (matches PanTS's own benchmark table, confirmed against the
  professor's original message): P-Sen (patient-wise sensitivity), T-Sen (tumor-wise sensitivity),
  Spe (specificity), AUC, DSC (Dice).
- **Two real bugs the port had to fix**, not just a mechanical copy-paste:
  1. The official v1 UNet++ code is silently hardcoded for exactly 5 downsampling stages. Our
     version generalizes this with loops so it works for whatever stage count nnU-Net v2's planner
     picks for PanTS.
  2. UNet++'s own deep-supervision outputs are all full-resolution (not progressively downsampled
     like nnU-Net's default scheme) — fixed via a `_get_deep_supervision_scales` override rather than
     writing a new loss function.
- **Tumor/pancreas label overlap:** when merging PanTS's 28 organ files into one, `pancreatic_lesion`
  (the tumor) is deliberately merged *last* so tumor voxels always win over the surrounding
  `pancreas` label on any overlap.
- **Test-set labels** are kept in a separate folder (`--test-answer-key-dir`), NOT inside nnU-Net's
  own `nnUNet_raw` dataset folder — nnU-Net itself never needs/uses test labels, but we need them
  later (Phase 7) to actually score the model's predictions.

## Verification done so far (all local, no GPU/real data yet)
Ran `unet_plusplus.py` through dummy-tensor forward passes on CPU with 4, 6, and 7 downsampling
stages, both with and without deep supervision — all passed with correct output shapes. This is
NOT the same as verifying it trains correctly on real data; that can only happen on the GPU server.

## Server info
`ssh enl014@wenyuan.ucsd.edu`

## Status as of last session (update this section as you go)
Setup runbook in progress on the server, using conda environment **`pants_unetpp`** (created fresh,
NOT the pre-existing `nnunet` environment that turned out to already be on this server from earlier
work — renamed to avoid ambiguity with whatever that one is used for).

**Resolved:** a pip dependency-conflict warning (mentioned `mlflow`, `pytorch-ignite`,
`torchaudio 0.11.0+cu113`) turned out to be harmless — those packages live in the user-level
`~/.local/lib/python3.10/site-packages` folder (shared across every environment on this account,
from some earlier `pip install --user ...`, unrelated to this project), not inside `pants_unetpp`
itself. Confirmed nnU-Net v2 installed correctly: `import nnunetv2` resolves to
`/home/enl014/nnUNet/nnunetv2/__init__.py` as expected.

**Confirmed:** `smoke_test.py` passed on the actual server (real installed nnU-Net v2 + PyTorch, not
just the local CPU sandbox test from earlier) — 5-stage and 6-stage configs both build and run
correctly, with and without deep supervision. Phase 3 (the architecture port) is now verified in the
real target environment, not just in isolation.

**Storage decision:** using `/Scratch/enl014/` for everything data-related (raw PanTS download,
nnU-Net's raw/preprocessed/results folders). Reasons: (1) home directory (`~`, on `/`) is at 95% full
with only 22G free — nowhere near enough room and risky to fill further; (2) `/Scratch` is a local
disk (no hostname prefix in `df -h`, unlike `/mnt/cube`, `/mnt/pentagon` etc. which are network
mounts) — local disk avoids network I/O becoming a training bottleneck, and it has 2.7T free.

**Home-directory cleanup done:** removed ~4.2G of confirmed-safe items from `~` (duplicate
`AbdomenAtlasDemo.tar.gz` copies, an unrelated `ollama` download, `abdomenatlas_minimal_upload`, and
the SuPreM vertebrae checkpoint since that work is fully done). Left `AbdomenAtlasDemo/`,
`AbdomenAtlasDemoPredict/`, and `SuPreM/` untouched since they're tied to the already-approved
postprocessing script. Also ran `conda clean --all -y` to clear conda's package cache.

**PanTS download: DONE.** Data (~362G) and labels both downloaded successfully to
`/Scratch/enl014/PanTS/data`.

**Phase 2 (data conversion): DONE on real data.** Ran a 3-case sanity check first (in a throwaway
`_sanitycheck` output folder) and manually verified image/label shapes matched and label values were
anatomically sensible before committing to the full run. Full conversion then completed cleanly:
**9,000 training cases + 901 test cases converted**, exactly matching the official split. Output is at
`/Scratch/enl014/nnUNet_raw/Dataset001_PanTS` (images/labels) and
`/Scratch/enl014/PanTS_test_answer_key` (test-set ground truth, kept separate per the earlier
"answer key" decision).

**Cleanup still to do:** delete the two throwaway sanity-check folders
(`/Scratch/enl014/nnUNet_raw/Dataset001_PanTS_sanitycheck` and
`/Scratch/enl014/PanTS_test_answer_key_sanitycheck`) — no longer needed now the real run succeeded.

**Phase 4 (preprocessing) — first attempt failed, root-caused and fixed:**
`nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity` crashed with
`ITK ERROR: ITK only supports orthonormal direction cosines`. Diagnosed with a new script,
`check_affine_orthonormality.py` (in `~/pants_nnunet_conversion/`), which found **~24 out of 9,901
cases (~0.24%)** had orientation metadata that wasn't perfectly "square" — all deviations were tiny
(~0.0001-0.0004), consistent with floating-point rounding noise rather than a real physically-tilted
scan (which would show deviations ~0.01-0.3). nnU-Net's SimpleITK-based reader enforces this strictly;
`nibabel` (used in our earlier sanity check) does not, which is why the 3-case check didn't catch it.
Decision: **fix rather than exclude** these files (nudge to the nearest perfectly-square orientation
via `fix_affine_orthonormality.py`, same folder) since the deviation is negligible and this preserves
the full official 9,901-case dataset rather than shrinking it.

**Affine fix confirmed working** — preprocessing re-ran successfully after the fix.
**Phase 4 (preprocessing): DONE.** 3D-fullres configuration (the only one we need) fully preprocessed
(~29 hours). Killed the still-running, unneeded `3d_lowres` stage afterward (that's for a cascade
training approach we're not using).

**Phase 5 (training) — hit and resolved two real issues before it actually started working:**
1. `ModuleNotFoundError: No module named 'unet_plusplus'` — the trainer file's import
   (`from unet_plusplus import ...`) worked when run standalone (`smoke_test.py`) but not when
   imported as part of nnU-Net's own package system. Fixed with a relative import
   (`from .unet_plusplus import ...`) in `nnUNetTrainerUNetPlusPlus.py`.
2. `RuntimeError: The NVIDIA driver on your system is too old` — `pip install`'d `torch` had
   defaulted to a build expecting a newer CUDA version than this server's driver (550.163.01, CUDA
   12.4) supports. Fixed by reinstalling with `pip install torch --index-url
   https://download.pytorch.org/whl/cu124` inside the `pants_unetpp` environment.
3. **The anticipated OOM risk (flagged in the port's README from the start) materialized exactly as
   expected**: `CUDA out of memory` at ~23.5/23.67GB used, batch_size=2. Fixed by creating a modified
   copy of the plans file (`nnUNetPlansBS1.json`, same folder as the original) with `batch_size`
   dropped to 1 — did NOT require redoing preprocessing, since batch size only affects training-time
   data grouping, not the prepared data itself.

**Confirmed: training is now running successfully** — GPU memory stable at ~15.8/24.5GB, GPU
utilization 98-100%, no crash, past the point of the previous failure. Launched via:
```
nohup nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerUNetPlusPlus -p nnUNetPlansBS1 > train.log 2>&1 &
```

**Switched to multi-GPU training** (4 GPUs instead of 1) to speed things up, since 7 of 8 GPUs on
this shared server were sitting idle. One correction along the way: nnU-Net's DDP mode treats
`batch_size` in the plans file as the *global* batch size split evenly across GPUs, not a per-GPU
number — first attempt (`nnUNetPlansBS1.json`, batch_size=1, `-num_gpus 4`) failed with
`AssertionError: Cannot run DDP if the batch size is smaller than the number of GPUs`. Fixed by
creating `nnUNetPlansBS4.json` (batch_size=4, dividing to exactly 1 per GPU — the same per-GPU amount
already proven to fit in memory). **Confirmed working:** all 4 GPUs at 98-99% utilization, ~16.5GB
each, no crash. Launch command:
```
nohup nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerUNetPlusPlus -p nnUNetPlansBS4 -num_gpus 4 > train.log 2>&1 &
```

**Phase 5 (training): DONE.** All 1,000 epochs completed successfully, ~140s/epoch steady state on 4
GPUs (~39 hours total). `checkpoint_final.pth` and `checkpoint_best.pth` both saved correctly at
`/Scratch/enl014/nnUNet_results/Dataset001_PanTS/nnUNetTrainerUNetPlusPlus__nnUNetPlansBS4__3d_fullres/fold_0/`.

Note: nnU-Net's automatic *internal* self-validation step (testing itself against the 1,800 cases
held back from the 9,000 training cases — NOT the official PanTS-te test set) crashed partway through
with a `BrokenPipeError` in a background worker process, after writing partial output. **This does not
affect the trained model** (already fully saved before this step ran) and this internal check isn't
part of the actual deliverable anyway (that's the official 901-case test set, handled separately in
Phase 6/7) — decided not to debug/rerun it, not worth the time given it's not required.

**Phase 6 (inference on official test set): DONE.** First attempt (single GPU, no probabilities
saved) completed but turned out to be insufficient — realized AUC (one of the 5 required metrics)
needs per-case confidence scores, not just final segmentation masks, and probabilities can't be
recovered after the fact. Redid prediction with `--save_probabilities` added, this time split across
4 GPUs manually (prediction doesn't have training's automatic `-num_gpus`; instead uses `-num_parts`/
`-part_id` + `CUDA_VISIBLE_DEVICES` per process — confirmed this doesn't affect results at all, unlike
training's multi-GPU, since each test case is predicted completely independently of every other case).
Hit the same "wrong conda environment" mistake as earlier (ended up in `nnunet` instead of
`pants_unetpp` in a new terminal session, resurfacing the old CUDA-driver-mismatch error) — same fix
as before. All 901 cases completed successfully across all 4 parts (226+225+225+225=901), processes
exited cleanly.

**Correction:** the "all 901 done cleanly" read was wrong — all 4 parts actually crashed with
`OSError: No space left on device` (`/Scratch` hit 100% full, 0 bytes free). 653/901 cases had
already saved their probability file before the crash. Root cause: over 1TB of unused preprocessed
data (`nnUNetPlans_2d` 879G + `nnUNetPlans_3d_lowres` 141G — leftover from stages we never use, only
`3d_fullres` matters) had been silently accumulating on the drive. Deleted both to free the space,
then resumed prediction using nnU-Net's `--continue_prediction` flag (skips already-completed cases,
only redoes the missing ~248) rather than restarting all 901 from scratch.

**Two more issues hit during the resume, both resolved:**
1. First resume attempt (all 4 parts) crashed instantly with the same `No space left on device` —
   turned out the `rm -rf` cleanup commands hadn't actually been run yet before checking `df -h`
   (still showed 100% full). Re-ran the deletions for real, freed ~1019GB, confirmed via `df -h`.
2. After that, parts 2 and 3 crashed with `CUDA out of memory` — but this was **not our bug**: a
   different user (`shc067`, unrelated job called `infer2align`) had started legitimately using GPUs
   2-7 on this shared server in the meantime, colliding with our `-part_id 2`/`3` GPU assignments.
   Did NOT kill their process (not ours to touch). Fixed by re-running parts 2 and 3 on GPUs 0 and 1
   instead (which had spare capacity, ~8GB/24.5GB used), sharing those two GPUs across all 4 of our
   parts rather than displacing another user's work.

**Phase 6: FULLY DONE.** Confirmed via direct file count (901/901 `.npz` probability files present,
not just log line counts, since some log files had been overwritten by mid-run restarts) — all 901
official test cases have complete segmentation + probability output at
`/Scratch/enl014/PanTS_predictions_test`. One more real-world hiccup handled along the way: another
user's unrelated job (`shc067`, `infer2align`) temporarily occupied GPUs 2-7 during the resume,
causing parts 2/3 to fail — worked around by sharing GPUs 0/1 rather than displacing their work, then
moved back to dedicated GPUs 2/3 once their job finished, restoring full speed.

**Major correction to the "Phase 6 fully done" claim:** while writing the Phase 7 scoring script
(`~/pants_phase7_scoring/compute_pants_metrics.py`), hit a `BadZipFile`/`No data left in file` error
loading a probability file. Wrote `check_npz_integrity.py` to scan all 901 `.npz` files properly
(actually reading their contents, not just checking existence) — found **585 of 901 (65%) were
corrupted**, mostly empty/truncated. Root cause: the earlier disk-full crash corrupted far more files
than the "901/901 `.npz` files present" count revealed at the time — that check only confirmed files
existed, not that their contents were valid, and `--continue_prediction` only checks existence too,
so these corrupted-but-present files were silently treated as "already done" during the resume.
Fixed with `clean_corrupted_predictions.py`, which finds every genuinely corrupted case and deletes
its full output (segmentation + probabilities + metadata) so a follow-up `--continue_prediction` run
will properly redo exactly those ~585 cases instead of skipping them.

**Next action:** rerun the 4-GPU prediction command with `--continue_prediction` to regenerate the
585 corrupted cases, re-run `check_npz_integrity.py` afterward to confirm zero corruption remains,
THEN run Phase 7's `compute_pants_metrics.py` for real.

**Lesson learned (general, not just this project): after ANY interrupted write — a crash, a
disk-full event, or a job you killed on purpose — verify file *integrity* (actually open/read the
contents), not just *existence*, before trusting it downstream.** `--continue_prediction` (and
similar "resume" flags in other tools) typically only check whether a file exists, not whether it's
valid, so a corrupted-but-present file can silently masquerade as "already done."

Root cause of the 585 corrupted files here was most likely the original disk-full crash (all 4
parallel export queues had files mid-write when the disk hit 0 bytes free, simultaneously) — NOT
primarily the later `pkill` used to move parts 2/3 to different GPUs, though that kill is also a
real, smaller contributing risk (whatever file was mid-write at that exact instant could be
affected). Note for future sessions: killing a job that's actively writing output files always
carries this risk, worth avoiding when avoidable. BUT: the `Ctrl+Z` → `bg` → `disown` trick (detaching
an already-running foreground job into the background without killing it) does NOT help when the
actual goal is reassigning which GPU a process uses — GPU assignment via `CUDA_VISIBLE_DEVICES` is
fixed at process start and can't be changed for an already-running process, so kill-and-relaunch was
genuinely the only option in that situation. That trick IS useful, though, for the narrower case of
"I forgot `nohup` and need to background/detach a job without losing its progress or restarting it."

## Full setup runbook (server: enl014@wenyuan.ucsd.edu)

**On your laptop** (a terminal NOT already SSH'd in):
```
scp -r "/Users/engho/Downloads/unetplusplus_nnunetv2_port" enl014@wenyuan.ucsd.edu:~/
scp -r "/Users/engho/Downloads/pants_nnunet_conversion" enl014@wenyuan.ucsd.edu:~/
```

**On the server** (`ssh enl014@wenyuan.ucsd.edu`), from here on:
```
nvidia-smi                                   # confirm GPU is visible

conda create -n pants_unetpp python=3.10 -y
conda activate pants_unetpp                  # NOTE: an env named "nnunet" already existed on this
                                              # server from earlier work -- deliberately using a
                                              # differently-named fresh env to avoid ambiguity

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
cd ..
pip install nibabel

cp ~/unetplusplus_nnunetv2_port/unet_plusplus.py nnUNet/nnunetv2/training/nnUNetTrainer/
cp ~/unetplusplus_nnunetv2_port/nnUNetTrainerUNetPlusPlus.py nnUNet/nnunetv2/training/nnUNetTrainer/

cd ~/unetplusplus_nnunetv2_port
python smoke_test.py                         # must print "All smoke tests passed." before continuing

mkdir -p /Scratch/enl014/nnUNet_raw /Scratch/enl014/nnUNet_preprocessed /Scratch/enl014/nnUNet_results
export nnUNet_raw="/Scratch/enl014/nnUNet_raw"
export nnUNet_preprocessed="/Scratch/enl014/nnUNet_preprocessed"
export nnUNet_results="/Scratch/enl014/nnUNet_results"
# also add the 3 export lines above to ~/.bashrc so they persist across logins:
echo 'export nnUNet_raw="/Scratch/enl014/nnUNet_raw"' >> ~/.bashrc
echo 'export nnUNet_preprocessed="/Scratch/enl014/nnUNet_preprocessed"' >> ~/.bashrc
echo 'export nnUNet_results="/Scratch/enl014/nnUNet_results"' >> ~/.bashrc

cd /Scratch/enl014
git clone https://github.com/MrGiovanni/PanTS.git
cd PanTS
nohup bash download_PanTS_data.sh > download_data.log 2>&1 &   # ~300GB, slow -- runs in background,
                                                                 # survives closing your laptop
# check progress anytime with: tail -20 /Scratch/enl014/PanTS/download_data.log
# once that's done (confirm the log shows it finished), run the label download the same way:
nohup bash download_PanTS_label.sh > download_label.log 2>&1 &
cd ..

python ~/pants_nnunet_conversion/convert_pants_to_nnunet.py \
    --pants-root /Scratch/enl014/PanTS/data \
    --nnunet-dataset-dir "$nnUNet_raw/Dataset001_PanTS" \
    --test-answer-key-dir /Scratch/enl014/PanTS_test_answer_key
```

(Earlier attempt used `tmux` for this, but switched to the simpler `nohup ... &` approach — same
effect, survives disconnecting, less to learn.)

## Status update (this section was stale for a while — corrected now)
Phases 2-6 are all genuinely, verifiably complete: data converted, preprocessed, trained (1000
epochs), and all 901 official test cases have verified-valid segmentation + probability data (see
`verify_and_repair_predictions.py`, which actually loads and checks every file, not just counts
them — ran clean on all 901).

New reusable tools built along the way, useful for any future prediction run on this or other
datasets:
- `predict_and_shrink.py` — predicts in small batches and immediately shrinks each batch's huge
  probability files down to the one number actually needed (avoids the repeated disk-full crashes
  we hit from letting full-size probability data for hundreds of cases pile up at once).
- `verify_and_repair_predictions.py` — thorough per-case audit (file existence AND actual
  readability) with automatic repair, re-verified after fixing. Catches the "file exists but is
  silently incomplete/corrupted" failure mode we kept hitting in different disguises.
- **Standing rule, learned the hard way:** always wrap every server command in `nohup`, no matter
  how quick it seems — a dropped SSH connection killed a foreground script mid-run and (combined
  with a since-fixed bug) permanently lost 346 cases' worth of data.

**PHASE 7 COMPLETE. Final results, UNet++ with deep supervision, official 901-case PanTS test set:**
- P-Sen: 0.8212
- T-Sen: 0.7205
- Spe: 0.8573
- AUC: 0.9033
- DSC: 0.4213

(Comparison, PanTS's published baselines: MedFormer 80.8/75.2/90.0/0.924/52.9,
R-Super 80.1/80.1/93.2/0.903/53.4 — our P-Sen and AUC are competitive/comparable, T-Sen/Spe/DSC
somewhat lower but plausible, not broken numbers.)

Per-case breakdown saved at `~/pants_phase7_scoring/per_case_results.csv` on the server.

**Next after that:** per the professor's note about deep supervision (see conversation — UNet++'s own
deep supervision is "horizontal"/same-resolution, different from nnU-Net's default "vertical"/
progressively-downsampled scheme), run the remaining legs of what's likely a 2×2 comparison:
nnU-Net default architecture with/without deep supervision, and UNet++ without deep supervision
(UNet++ with deep supervision is the run just completed). Confirm this interpretation with the
professor before committing ~39 hours per training run.

Task 1 (ShapeKit integration) — still parked, not started; see the plan file for the generalization
risk that should be tested before integrating.

## 2×2 comparison — cross-val tumor metrics (NOT the official PanTS-te test set)
After the original UNet++-DS-on run above, the professor flagged that UNet++'s deep supervision
can't just be toggled on/off like nnU-Net's default (different mechanism — see conversation; fixed
a real loss-weighting bug in `nnUNetTrainerUNetPlusPlus.py` as a result). This required retraining
all 4 legs of the 2×2 grid (architecture × deep supervision) from scratch, under a freshly
regenerated preprocessing pass and a new `nnUNetPlansBS2` plan (2-GPU parallel training, later
switched to single-GPU for validation to dodge repeated NCCL/DDP timeout crashes — see conversation
for the full debugging trail: torch/torchvision version mismatch, DDP unused-parameters bug, CPU
worker contention, NCCL collective timeouts).

**Important caveat: these numbers are from nnU-Net's own internal 5-fold cross-validation split
(fold 0's held-out 1,800 cases, carved out of the 9,000 training cases) — NOT the official 901-case
PanTS-te test set.** Not directly comparable to the published leaderboard baselines or to the
DSC 0.4213 result above. Official PanTS-te predictions for all 4 legs are in progress as of this
writing (see conversation) and should replace/supplement this table once done.

Tumor-specific metrics (class 28 = pancreatic_lesion only, not the 28-class aggregate Dice):

| Trainer | DSC (tumor) | P-Sen | T-Sen | Spe | Overall Dice (all 28 classes) |
|---|---|---|---|---|---|
| Default, DS on | 0.00022 | 0.0632 | 0.0352 | 0.9982 | 0.6741 |
| Default, DS off | 0.0 | 0.0 | 0.0 | 1.0 | 0.6716 |
| UNet++, DS on | 0.0 | 0.0 | 0.0 | 0.9994 | 0.6687 |
| UNet++, DS off | 0.0 | 0.0115 | 0.0 | 0.9982 | 0.6752 |

(174 tumor-positive cases / 1,626 tumor-negative cases / 227 individual tumor instances across the
1,800-case fold.)

**Headline finding:** all 4 configurations essentially fail at tumor detection (near-zero or literal
zero sensitivity), despite healthy-looking overall Dice (~0.67-0.68) driven by the other 27
organ/anatomy classes. Spot-checked one case with a real, substantial tumor (1,613 GT voxels,
matching prediction/GT shapes, ruling out a resampling bug) — model predicted zero tumor voxels.
Most likely cause: none of these 4 runs used any tumor-specific oversampling/class-weighting, so a
rare structure (174/1800 cases) got outcompeted by the 27 far more common organ classes during
training. Deep supervision (default arch only) is the only run that learned any tumor signal at all,
though still far too low to be clinically useful (6.3% patient-level sensitivity).

### ROOT-CAUSE INVESTIGATION (why 0.0 here vs DSC 0.4213 in the original run)

**The failure is in TRAINING, not scoring or inference.** nnU-Net logs per-class pseudo-Dice every
epoch; the tumor class (class 28, last entry in the list) at final epoch:

| Run | tumor pseudo-Dice | n classes |
|---|---|---|
| ORIGINAL (BS4, UNet++ DS-on) | **0.3347** | 28 |
| Default DS-on (BS2) | 0.0004 | 28 |
| Default DS-off (BS2) | 0.0 | 28 |
| UNet++ DS-on (BS2) | 0.0 | 28 |
| UNet++ DS-off (BS2) | 0.0 | 28 |

The original learned tumors; none of the 4 new runs ever did, at any point in 1000 epochs.

**Ruled out, with evidence:**
- *Truncated training* — all 5 runs reached epoch 999 (full 1000).
- *Missing tumors in preprocessed data* — `PanTS_00000029_seg.b2nd` contains 3,626 class-28 voxels
  (raw GT has 1,613; differs because preprocessing resamples spacing). All 28 classes present.
- *Broken label config* — pseudo-Dice lists are length 28 in every run.
- *My UNet++ port / loss-weighting fix* — the two DEFAULT-architecture runs use stock, unmodified
  nnU-Net code and failed identically. Not caused by any custom code.
- *My scoring script* — the original run's own `per_case_results.csv` shows real detections
  (e.g. PanTS_00009001: pred_has_tumor=True, max_prob=0.984, dice=0.606), and its columns match the
  quantities `compute_tumor_metrics.py` computes.

**Remaining difference:** diffing `nnUNetPlansBS4.json` vs `nnUNetPlansBS2.json` shows the ONLY
config difference is `batch_size` (4 vs 2) — patch_size, spacing, normalization,
oversample_foreground_percent and data_identifier are all identical. The other uncontrolled variable
is that the preprocessed DATA was regenerated between the two (both plans point at the same
`nnUNetPlans_3d_fullres` folder, so the original's preprocessed data was overwritten).

Additional things ruled out with evidence:
- *`class_locations` missing tumor* — `PanTS_00000029.pkl` has 3,626 sampled locations for class 28,
  so the oversampler could and did have tumor patches available.
- *Raw data changed* — `labelsTr`/`imagesTr` are dated Jul 21, unchanged since before the original
  run trained (Jul 26). Only the preprocessed data was regenerated (Aug 18), from identical raw.
- *Crashes truncating training* — all the NCCL/DDP crashes happened in the POST-training validation
  phase. Training itself completed 1000 clean epochs in every run.

**DIAGNOSIS: delayed-onset rare-class learning, pushed past the epoch budget by batch_size 2.**

Tumor pseudo-Dice, % of epochs in each 100-epoch window where a tumor was seen at all:

| Epoch range | ORIGINAL BS4 | Default DS-on BS2 | UNet++ DS-on BS2 |
|---|---|---|---|
| 0-300 | 0% | 1-2% | 0-3% |
| 300-400 | 7% | 0% | 0% |
| 400-500 | 58% | 0% | 2% |
| 500-600 | 87% | 2% | 0% |
| 600-700 | 97% | 0% | 0% |
| 700-800 | 98% | 1% | 1% |
| 800-900 | 100% | 3% | 2% |
| 900-1000 | 99% (mean 0.38) | 17% | 1% |

The original learned NOTHING about tumors for its first ~300 epochs, then underwent a sharp phase
transition and climbed steadily to 0.38. None of the BS2 runs ever underwent that transition within
the same 1000 epochs. (Default DS-on hitting 17% in its last 100 epochs may be the onset just
beginning.) Note also: when BS2 models DID see a tumor they segmented it well (UNet++ BS2 averaged
0.549 on its nonzero epochs, higher than the original's 0.342) — the models aren't incapable, they
were simply never pushed over the transition.

Practical consequence: **any short pilot run is worthless as a test** — the original showed 0% at
epoch 50 and at epoch 200. Validating a config change requires going at least ~500 epochs.

**FIX: retrain all 4 legs at batch_size 4** (`nnUNetPlansBS4`, which already exists and is identical
to BS2 apart from batch_size). Memory observed during BS2 runs: default arch ~5.7GB per patch,
UNet++ ~17GB per patch, on 24GB A5000s — so UNet++ at BS4 needs either 2 patches/GPU (if it fits)
or 4 GPUs at 1 patch each. Check `/home/enl014/nnUNet/train.log` for how the original was launched
before allocating.

**Do not report any of the 4 new runs' numbers until this is resolved** — they reflect a broken
training setup, not a real architecture/deep-supervision comparison.

**Also discovered:** `labelsTs` does NOT exist (nnU-Net convention leaves imagesTs unlabeled). The
official 901-case test ground truth lives at `/Scratch/enl014/PanTS_test_answer_key/`. Any PanTS-te
scoring command must point there.

**Also unresolved (worth asking the professor):** PanTS's paper (Appendix B.4.1) only gives the
textbook DSC formula and never states whether benchmark DSC is tumor-class-only vs all-structures,
averaged over all 901 cases vs tumor-positive only, or whether tumor predictions come from argmax or
a probability threshold. Any number reported against the leaderboard carries an assumption that
should be stated explicitly.

## Fidelity audit — where the port follows nnU-Net vs UNet++ (2026-09-03)

Three reference points, and they do not always agree:
- **P** = the UNet++ paper (Zhou et al., "Redesigning Skip Connections", PMC7357299)
- **O** = the official UNet++-in-nnU-Net-v1 fork (MrGiovanni/UNetPlusPlus) — the thing we were asked to port
- **N** = nnU-Net v2's own conventions

| Aspect | P (paper) | O (official v1) | Our v2 port | Verdict |
|---|---|---|---|---|
| Nested dense skip grid | yes | yes | yes | faithful |
| Node inputs: X[i][0..j-1] + Up(X[i+1][j-1]) | yes | yes | yes | faithful (verified against paper) |
| Channels constant per row | yes | yes | yes | faithful |
| Number of encoder stages | 5 | **hardcoded 5** | from nnU-Net planner (5-7) | deliberate deviation — the hardcoding would silently break on datasets where v2's planner picks another depth |
| Upsampling | transposed conv | transposed conv | transposed conv | faithful |
| Encoder / features_per_stage | own | from v1 planner | from v2 planner | faithful in spirit (both defer to the framework) |
| DS outputs all at full resolution | yes | yes | yes | faithful |
| DS target downsampling | n/a | n/a | overridden to none | faithful to UNet++ (N would downsample; we don't) |
| **DS loss weighting** | **η_i ≡ 1 (equal)** | **see below — unresolved** | N's exponential decay | **deviation, evidence-based** |
| **Inference output** | **average all branches** | uses output[0] only | final branch only | **deviation from P, matches O** |
| Loss function | BCE + soft Dice | N v1 default (DC_and_CE) | N v2 default (DC_and_CE) | follows framework, as O does |
| Optimiser | Adam | SGD m=0.99 nesterov (N v1) | SGD lr 0.01 poly (N v2) | follows framework, as O does |
| Nonlinearity | ReLU | LeakyReLU (N) | LeakyReLU (N) | follows framework, as O does |
| Normalisation | BatchNorm | InstanceNorm (N) | InstanceNorm (N) | follows framework, as O does |
| Architecture pruning at inference | supported | not used | not implemented | not required for this task |

**RESOLVED (verified against the raw files, downloaded to `~/unetpp_reference/` on the server).**
The official v1 fork does NOT do multi-output deep supervision in its loss, and appears to train a
different branch than it predicts with.

`training/loss_functions/deep_supervision.py`, `MultipleOutputLoss2.forward()` — the summation over
the remaining outputs is commented out, leaving a single-output loss:
```python
l = weights[0] * self.loss(x[-1], y[0])
#for i in range(1, len(x)):
#    if weights[i] != 0:
#        l += weights[i] * self.loss(x[i], y[0])
return l
```
And `nnUNetTrainerV2.initialize()` has the nnU-Net decay-weight line commented out, setting
`self.ds_loss_weights = None` (which `MultipleOutputLoss2` turns into `[1] * len(x)`, though only
`weights[0]` is ever used given the above).

Which branch is `x[-1]`? In `network_architecture/generic_UNetPlusPlus.py` (lines 396-432) the local
`seg_outputs` list is appended shallowest-first — index 0 = x0_1 … index 4 = x0_5 — and the deep
supervision return is:
```python
return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                  zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
```
i.e. the tuple is (x0_5, x0_4, x0_3, x0_2, x0_1). So `x[-1]` is **x0_1, the shallowest branch**, while
the non-deep-supervision return path (line 432) is `seg_outputs[-1]` = **x0_5, the deepest** — which is
what nnU-Net v1 uses at inference. As written, the reference trains x0_1 and predicts with x0_5.

Caveat before repeating this to anyone: there may be intent here that isn't visible from the code —
the UNet++ paper describes *pruning* (deploying a shallow branch for speed), so this could be a
leftover from experiments in that direction. Line references for checking: `deep_supervision.py`
lines 38-42, `generic_UNetPlusPlus.py` lines 396-432.

**Consequence for our port.** Ours is internally consistent where the reference is not: it trains all
branches with nnU-Net's decaying weights and predicts with the deepest, so the branch producing the
answer receives the most supervision. Three distinct designs exist —
P: equal weights + average all branches at inference;
O: single-output loss on the shallowest + predict with the deepest;
ours: decaying weights over all + predict with the deepest.
Only ours has been measured on PanTS.

**What IS settled:** the paper specifies equal weights (η_i ≡ 1) AND averaging all branches at
inference — the two are designed to go together. Our port implements neither: it uses nnU-Net's decay
weighting with final-branch-only inference. The decay weighting is a deliberate, measured choice
(equal weighting gave 0% tumour-detecting epochs through epoch 500 vs 87% for decay, all else equal).
The final-branch-only inference is inherited from nnU-Net and has not been tested against the paper's
branch-averaging — that is Q4 in the status brief.

## Related documents
- Full phase-by-phase plan with plain-language explanations:
  `~/.claude/plans/users-engho-downloads-postprocessing-ve-distributed-cray.md`
