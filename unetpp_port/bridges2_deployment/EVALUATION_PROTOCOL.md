# Bridges-2 post-training evaluation preparation

## October 1 launch safeguards

All four 1000-epoch training runs and full 1800-case validations are now complete.
The checkpoint manifest is a historical record; the launch guard recomputes current
checkpoint hashes and checks full summary case identities, including class 28.
The continuation architecture source is pinned to `61be5315e8cc5f4ec35637c9b748e5ebc33da1b7`,
the actual successful continuation revision, not the older submission revision.
Earlier training used equivalent custom trainer files; verify frozen file equality
before deployment rather than implying all epochs used this continuation revision.

Prediction uses the reference CLI, not the unverified persistent-predictor candidate.
Default FP16 autocast, full mirroring, tile step 0.5 and all 29 output channels
are retained equally across cells. BF16 training-validation is a different path;
this distinction is deliberate and must be reported, not concealed.
One preprocessing/export worker per model bounds CPU-memory pressure. A single
two-task Slurm step and distinct physical UUID gate mirror the verified training
placement. Five-case chunks are an IO/restart grouping, not a changed model batch.
Resume requires unchanged checkpoint/plans/dataset/source/input hashes and valid
decoded prediction geometry and labels. Images use a pinned HuggingFace revision;
both downloaded archive hashes are recorded.

The GPU job now finishes at `GRID_TEST_PREDICTIONS_DONE_CPU_SCORING_REQUIRED`.
It saves converted GT on Ocean, alongside compact predictions and probability CSVs.
Run `evaluation/score_grid_cpu.py --evaluation-dir <downloaded evaluation directory>`
in the local CPU audit environment after copying these artifacts. It calls the same
tested scorer, checks exact 901 IDs, and writes `grid_metrics.json` only after all
four metric files pass. No GPU is held for CPU-only scoring. The older description
below of in-allocation scoring is superseded by this section.

`evaluate_grid.sbatch` and `submit_evaluation.sh` are prepared only. Do not submit while the
four-cell grid is queued or running. The wrapper requires final checkpoints and exactly 1,800
fold-validation predictions in each of the four cells; the GPU job rechecks all 7,200 NIfTI
headers and validation summaries before test inference. A final checkpoint alone is insufficient
because nnU-Net writes it before the final validation finishes.

The job uses the exact frozen training source commit for model classes and freezes its own clean
evaluation commit at submission. It requires the validated PyTorch 2.10.0+cu126, cuDNN 9.10.2,
nnU-Net v2 2.8.1 stack. It records hashes of the evaluation source, preventing a resumed run from
mixing scorer versions. It downloads/converts only the
901-case test split into job-local `$LOCAL`, asserts the exact test case IDs, then predicts two
cells concurrently on two H100s followed by the other two. Each case's tumor-only segmentation
and maximum class-28 probability are saved directly to persistent Ocean storage after inference;
completed case/score pairs can be reused if a later job must resume. Each model is scored only
after its complete 901-case prediction audit passes.

The five metrics use the same **project protocol** as `evaluation/DELTA_EVALUATION_PROTOCOL.md`:
class-28 Dice on ground-truth-positive cases, patient sensitivity, six-connected tumor-component
sensitivity, specificity on ground-truth-negative cases, and AUC from maximum class-28 softmax
probability. The public PanTS materials do not fully specify these scorer conventions; obtain PI
confirmation before claiming bit-for-bit agreement with an official leaderboard.

The preparation has no GPU submission. Before launch, review the final training checkpoint and
validation state, confirm Ocean headroom for persistent test outputs, run `bash -n` and Slurm
`--test-only`, and submit through `submit_evaluation.sh`. The final run produces
`results/evaluation_grid/<training-commit>/grid_metrics.json` only after all four 901-case
metric files pass their audits.
