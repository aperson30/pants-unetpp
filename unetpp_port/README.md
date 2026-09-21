# UNet++ ported to nnU-Net v2

The architecture and `nnUNetTrainer*.py` files are meant to be copied into the nnU-Net v2 trainer
directory on the GPU server. The important pieces are:

- `unet_plusplus.py` — the network architecture itself (encoder + UNet++'s nested decoder).
- `nnUNetTrainerUNetPlusPlus.py` — the custom trainer that tells nnU-Net v2 to use this network
  instead of its default one.
- `nnUNetTrainerQualityNeutralOptimizationMixin.py` — defers per-step scalar synchronization,
  avoids dense one-hot validation tensors for exclusive labels, and safely reuses verified-identical
  full-resolution UNet++ targets.
- `nnUNetTrainerSparseValidationMixin.py` and the `*SparseValidation.py` trainers — validate every
  five epochs without changing optimizer steps; variants cover all four grid cells plus the paper
  configuration.
- `smoke_test.py` — a fast (~seconds) sanity check you should run before starting any real
  preprocessing/training, to catch shape bugs early.

## Why this isn't just a copy-paste of the original UNet++ code

The official UNet++ implementation (`MrGiovanni/UNetPlusPlus`) only works for networks with exactly
5 downsampling stages — it's hardcoded, not something the original authors documented. nnU-Net v2's
automatic planner won't necessarily choose 5 stages for a new dataset (it depends on median image
size/spacing), so a direct copy-paste would work by luck on some datasets and break on others. This
port builds the same nested-skip-connection architecture generically, for any number of stages.
See the docstring at the top of `unet_plusplus.py` for the full explanation, and the docstring at the
top of `nnUNetTrainerUNetPlusPlus.py` for how deep supervision is handled (UNet++'s own
deep-supervision scheme doesn't match nnU-Net's default one, since all of UNet++'s auxiliary outputs
are full-resolution rather than progressively downsampled).

## Setup on your GPU server

1. Copy both `.py` files into `nnUNet/nnunetv2/training/nnUNetTrainer/` in your nnU-Net v2 checkout.
2. `python smoke_test.py` — should print "All smoke tests passed." in a few seconds on CPU, faster on
   GPU. If this fails, something is wrong with the environment or the port — fix it here before
   touching real data.
3. Once your dataset is converted and preprocessed (see the plan doc's Phase 2), train with:
   ```
   nnUNetv2_train <DATASET_ID> 3d_fullres 0 -tr nnUNetTrainerUNetPlusPlus
   ```

## Known risk to watch for: out-of-memory on your first real training attempt

This port reuses whatever `n_stages` / `features_per_stage` / patch size nnU-Net's planner picked for
its *default* architecture. UNet++'s decoder concatenates far more feature maps than a plain U-Net
decoder does, so it uses more GPU memory at the same settings. If your first `nnUNetv2_train` run
crashes with a CUDA out-of-memory error, that's the expected failure mode of this approach — not a
bug — and the fix is to manually reduce `batch_size` (or patch size) in a copy of the dataset's
`plans.json` for the `3d_fullres` configuration, then re-run preprocessing/training with that edited
plans file (`nnUNetv2_train ... -p <your_edited_plans_identifier>`).

## What's verified vs. what isn't

Verified locally (CPU, small dummy tensors, see `smoke_test.py`): the architecture builds and runs a
forward pass correctly for several different stage counts and both with/without deep supervision, and
output shapes are correct in each case.

NOT yet verified (needs your GPU server, since it can't be tested from here): actual training
convergence, real memory usage on PanTS-sized 3D patches, and the deep-supervision loss wiring
end-to-end with real nnU-Net data loaders. Treat the first training run as a debugging pass — watch
the first few iterations' loss and GPU memory before walking away from it.
