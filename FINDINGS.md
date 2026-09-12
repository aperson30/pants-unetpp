# Findings

Two training defects diagnosed during this work, plus an audit of how faithfully the port follows the
UNet++ paper versus the official v1 implementation. Both defects produced *silent* failure — the runs
completed normally and reported healthy-looking overall Dice while detecting essentially no tumours.

---

## Why the failure was invisible

The models segment 28 structures. Overall foreground Dice came out around **0.67**, which looks
entirely reasonable — but that average is dominated by 27 large, common organs. The tumour is present
in only ~10% of cases and is tiny when present, so a model can score well overall while never
predicting a single tumour voxel.

**Always evaluate the tumour class (label 28) on its own.** Aggregate Dice will not tell you that
anything is wrong.

---

## Defect 1 — batch size 2 never reaches the point where tumour learning begins

Four runs were trained at `batch_size 2` split across two GPUs (one patch per GPU). All four failed:

| Run | Final tumour pseudo-Dice |
|---|---|
| Plain U-Net, DS on | 0.0004 |
| Plain U-Net, DS off | 0.0 |
| UNet++, DS on | 0.0 |
| UNet++, DS off | 0.0 |

An earlier run at `batch_size 4` reached **0.33**. Diffing the two plans files showed **batch size was
the only difference** — patch size, spacing, normalisation, foreground oversampling and data
identifier were all identical.

### The mechanism: delayed onset

Tumour learning here is not gradual. It sits at zero for hundreds of epochs, then rises sharply. The
batch-2 runs never crossed that transition inside the 1000-epoch budget.

Percentage of epochs in which any tumour was detected at all:

| Epochs | Batch 4 (works) | Batch 2 — U-Net | Batch 2 — UNet++ |
|---|---|---|---|
| 0–300 | 0% | 1–2% | 0–3% |
| 300–400 | 7% | 0% | 0% |
| 400–500 | 58% | 0% | 2% |
| 500–600 | 87% | 2% | 0% |
| 600–800 | 97–98% | 0–1% | 0–1% |
| 900–1000 | 99% (Dice 0.38) | 17% | 1% |

Why batch size matters so much: the plans set `batch_dice: True`, so the Dice term is computed across
the whole batch together. A step containing no tumour gives the model **no tumour feedback at all**.
Halving the patches per step roughly doubles how many steps are completely tumour-free, and that was
enough to keep the run below the threshold for its entire budget.

### Consequences

- **Retrain at batch size 4** (`nnUNetPlansBS4`). UNet++ needs ~17 GB per patch on a 24 GB card, so
  that means 4 GPUs at one patch each.
- **Short pilot runs cannot validate configuration changes.** The known-good run showed 0% tumour
  detection at epoch 50, at epoch 200, and at epoch 300. Any test needs ~500 epochs before it means
  anything.

### Ruled out, with evidence

| Hypothesis | Evidence against |
|---|---|
| Training cut short | All five runs reached epoch 999 of 1000 |
| Tumours missing from preprocessed data | `PanTS_00000029` has 3,626 class-28 voxels and 3,626 `class_locations` entries |
| Label misconfiguration | Every run logs exactly 28 classes; `dataset.json` maps `pancreatic_lesion → 28` |
| The custom UNet++ code | The two plain nnU-Net runs use unmodified framework code and failed identically |
| The scoring script | The earlier successful run scores correctly with the same script |
| Source data changed | `labelsTr` / `imagesTr` unchanged since 21 July, predating the successful run |
| Crashes truncating training | All the NCCL/DDP crashes happened in *post-training validation*, after checkpoints were written |

---

## Defect 2 — equal deep-supervision weighting suppressed tumour learning

This one arose from Prof. Zhou's observation that deep supervision in UNet++ and nnU-Net are
different mechanisms and cannot simply be toggled on and off.

**The setup.** nnU-Net assigns zero weight to its *last* auxiliary output, because in its
architecture that output is the lowest-resolution and least informative. UNet++'s outputs are all at
full resolution and differ only in nesting depth, and this port returns them reversed — so nnU-Net's
rule zeroed out the **shallowest branch** instead. The target handling was corrected (via
`_get_deep_supervision_scales`), but the weighting rule had been inherited unchanged.

**The attempted fix.** Weight all outputs equally, which is what the UNet++ paper specifies
(η<sub>i</sub> ≡ 1).

**The result — clearly worse.** Zero tumour-detecting epochs through epoch 500, versus 87% for the
otherwise identical run using nnU-Net's inherited weighting. Same architecture, same deep supervision
setting, same batch size, same GPU configuration; the loss weighting was the only variable. Reverted.

### Why equal weighting backfired here

The paper's weighting comes as a **matched pair** with an inference rule:

| | Training | Inference |
|---|---|---|
| UNet++ paper | All 4 branches, equal weight | **Average all 4** |
| This port | Mostly the 4th, decaying for the rest | **Only the 4th** |
| What was tried (broken) | All 4 equally | Only the 4th |

Both coherent designs make sense on their own. The hybrid does not: weighting equally while using one
output spends roughly three quarters of the gradient on branches that get discarded, leaving the
branch that actually produces the prediction with a fraction of the tumour signal. Common organs have
signal to spare; a class present in 10% of cases does not.

**Note on scope:** only the loss weighting was reverted. The network is UNet++ throughout — nested
decoder, dense skip connections — and deep supervision targets are still handled the UNet++ way (no
downsampling).

### What the port actually does at training time

Worth stating plainly, since "deep supervision" can mean several things:

- The network returns **all** outputs when DS is on (`seg_outputs[::-1]`, deepest first).
- `_get_deep_supervision_scales()` returns all-ones, so **every output is compared against the
  full-resolution ground truth** — no target downsampling. This is the UNet++-specific correction.
- nnU-Net v2's inherited `DeepSupervisionWrapper` **computes the loss on each output and sums them**,
  weighted:
  ```python
  return sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
  ```
- Weights come from nnU-Net's default exponential decay, normalised, with the last entry zeroed
  (`1e-6` under DDP). With 5 outputs that is approximately:

  | Output | Weight |
  |---|---|
  | X[0][5] — deepest, and the one used at inference | 0.533 |
  | X[0][4] | 0.267 |
  | X[0][3] | 0.133 |
  | X[0][2] | 0.067 |
  | X[0][1] — shallowest | ~0 |

So: multi-output deep supervision **is** active. What distinguishes this port from the paper is the
weighting (decaying rather than equal) and the inference rule (deepest branch rather than an average),
not whether each output is supervised.

---

## Fidelity audit

Three reference points, which do **not** all agree:

- **P** — the UNet++ paper (Zhou et al., *Redesigning Skip Connections*, PMC7357299)
- **O** — the official UNet++-in-nnU-Net-v1 fork (`MrGiovanni/UNetPlusPlus`), the code this ports
- **N** — nnU-Net v2's own conventions

| Aspect | P | O | This port | Verdict |
|---|---|---|---|---|
| Nested dense skip grid | yes | yes | yes | faithful |
| Node inputs: X[i][0..j-1] + Up(X[i+1][j-1]) | yes | yes | yes | faithful (verified against paper) |
| Channels constant per row | yes | yes | yes | faithful |
| Number of encoder stages | 5 | **hardcoded 5** | from nnU-Net planner (5–7) | deliberate deviation |
| Upsampling | transposed conv | transposed conv | transposed conv | faithful |
| Encoder / features per stage | own | from v1 planner | from v2 planner | faithful in spirit |
| DS outputs all full-resolution | yes | yes | yes | faithful |
| DS target downsampling | n/a | n/a | overridden to none | faithful to UNet++ |
| **DS loss weighting** | **equal (η ≡ 1)** | **single output only** | N's exponential decay | **deviation, evidence-based** |
| **Inference output** | **average all branches** | output[0] only | final branch only | **deviation from P** |
| Loss function | BCE + soft Dice | N v1 default | N v2 default | follows framework, as O does |
| Optimiser | Adam | SGD (N v1) | SGD lr 0.01 poly (N v2) | follows framework, as O does |
| Nonlinearity | ReLU | LeakyReLU (N) | LeakyReLU (N) | follows framework, as O does |
| Normalisation | BatchNorm | InstanceNorm (N) | InstanceNorm (N) | follows framework, as O does |
| Inference pruning | supported | not used | not implemented | not required |

The hardcoded-5-stages deviation is deliberate: nnU-Net v2's planner picks stage count from dataset
statistics, so a direct copy would break silently on any dataset where it picks something other
than 5.

### The official v1 fork does not do multi-output deep supervision

Verified by reading the raw files. In `training/loss_functions/deep_supervision.py`, the summation
over the remaining outputs is commented out:

```python
l = weights[0] * self.loss(x[-1], y[0])
#for i in range(1, len(x)):
#    if weights[i] != 0:
#        l += weights[i] * self.loss(x[i], y[0])
return l
```

And `nnUNetTrainerV2.initialize()` has nnU-Net's decay-weight line commented out, setting
`self.ds_loss_weights = None`.

Which branch is `x[-1]`? In `network_architecture/generic_UNetPlusPlus.py` (lines 396–432) the local
`seg_outputs` list is appended shallowest-first, and the deep-supervision return is
`tuple([seg_outputs[-1]] + [...seg_outputs[:-1][::-1]])` — i.e. `(x0_5, x0_4, x0_3, x0_2, x0_1)`. So
`x[-1]` is **x0_1, the shallowest branch**, while the non-DS return path used at inference is
`seg_outputs[-1]` = **x0_5, the deepest**.

**As written, the reference trains one branch and predicts with another.**

> **Caveat before repeating this.** There may be intent not visible in the code — the UNet++ paper
> describes *pruning* (deploying a shallow branch for speed), so this could be deliberate. Present it
> as "here is what the code does, line numbers attached", not as a bug report. Line references:
> `deep_supervision.py` 38–42, `generic_UNetPlusPlus.py` 396–432.

This port is at least internally consistent where the reference is not: it trains all branches with
decaying weight and predicts with the deepest, so the branch producing the answer receives the most
supervision. **Only this configuration has actually been measured on PanTS.**

---

## Open questions for Prof. Zhou

Four things that could not be settled from published material:

1. **Do the README benchmark rows use the 2-class or the 28-class setup?** This work trained
   28-class. The PanTS paper describes both as separately trained models. If the leaderboard rows are
   2-class, the tumour competes against one structure rather than 27 and the numbers are not directly
   comparable.
2. **Is there a documented training configuration for those rows?** Appendix B.3.1 covers the paper's
   own nnU-Net experiments (batch size 2, single RTX 4090, 1000 epochs, SGD 0.01). For comparative
   methods the paper cites external leaderboard results, and the README table carries no
   configuration at all.
3. **How exactly is benchmark DSC computed?** Appendix B.4.1 gives only the standard formula — not
   whether DSC is tumour-only or averaged across structures, over all 901 cases or only
   tumour-positive ones, or from argmax or a probability threshold. This work used: tumour-only,
   tumour-positive cases, argmax.
4. **Which deep supervision design should the port follow?** Given the three differ (table above),
   match the paper (equal weights + branch averaging), match the v1 port, or keep the current
   configuration? Testing an alternative means another training round plus an inference change.

---

## Results

### Completed — earlier UNet++ run, full official test set (901 cases)

| Method | P-Sen | T-Sen | Spe | AUC | DSC |
|---|---|---|---|---|---|
| **UNet++ (this work, DS on)** | 0.821 | 0.721 | 0.857 | 0.903 | 0.421 |
| MedFormer *(published)* | 0.808 | 0.752 | 0.900 | 0.924 | 0.529 |
| R-Super *(published)* | 0.801 | 0.801 | 0.932 | 0.903 | 0.534 |

Predates both defects. Checkpoint preserved at
`/Scratch/enl014/BACKUP_original_unetpp_bs4_run/checkpoint_final.pth`.

### Retrained 2×2 grid — training complete

| Run | Architecture | Deep sup. | Final tumour pseudo-Dice* |
|---|---|---|---|
| `default_ds` | Plain U-Net | On | 0.3998 |
| `unetpp_nods` | UNet++ | Off | 0.3923 |
| `unetpp_ds` | UNet++ | On | 0.3706 |
| `default_nods` | Plain U-Net | Off | 0.3665 |

\* Training-time proxy measured on patches — **not** the benchmark DSC. The 0.37–0.40 spread is
narrow and should not be read as an architecture result.

### Test-set metrics — incomplete as of the pause

| Run | Cases | P-Sen | T-Sen | Spe | AUC | DSC |
|---|---|---|---|---|---|---|
| `default_ds` | 753 / 901 | 0.838 | 0.709 | 0.854 | 0.909 | 0.427 |
| `default_nods` | 262 / 901 | 0.807 | 0.688 | 0.756 | 0.894 | 0.374 |

Partial because several evaluation pipelines running concurrently exhausted system RAM and the OS
killed export workers mid-batch. Nothing is lost — `predict_and_shrink.py` uses
`--continue_prediction` and resumes from wherever it stopped. See [HANDOFF.md](HANDOFF.md).
