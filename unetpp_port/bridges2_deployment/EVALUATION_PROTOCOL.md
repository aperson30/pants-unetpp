# Bridges-2 post-training evaluation preparation

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
