# Input parity and independent-judge audit

## Verified CPU input-transform parity

Inspected official MAISI code at revision
5cb04e82fed71f2fe64a2617ab695be1f5a37fea. `scripts/transforms.py` defines the
VAE validation path with RAS orientation, CT[-1000,1000]->[0,1] clipping,
default original spacing, whole-volume validation when val_patch_size=None,
and divisibility padding k=4. Reproduced these explicit MONAI operations
under our isolatedMONAI1.5.1 runtime; this does not execute a full official GPU
inference pipeline or establish bit parity of model execution.

Real-case array comparisons with the probe's input preprocessing:

| Case | Native spacing mm | Prepared shape | Max normalized difference |
|---|---|---|---:|
|120|0.683594,0.683594,4|512x512x104|0, exactly equal|
|165|0.976562,0.976562,2.5|512x512x88|0, exactly equal|

Therefore input intensity/orientation/padding differences do not explain the
observed changes relative to this validation path. No fixed-spacing resampling
is required by this DEFAULT VAE path. The official diffusion embedding-creation
script is DIFFERENT: it resizes dimensions to multiples of128 before encoding.
Do not silently import that geometry-changing step into the native-grid test.

Sources (pinned code reviewed directly):
https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/blob/5cb04e82fed71f2fe64a2617ab695be1f5a37fea/scripts/transforms.py
https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/blob/5cb04e82fed71f2fe64a2617ab695be1f5a37fea/scripts/diff_model_create_training_data.py

## NOT full official execution parity

Official config_network_rflow.json uses norm_float16=true, num_splits=4.
Our diagnostic uses norm_float16=false, num_splits=1, whole-volume FP32,
posterior mean rather than posterior sampling. Checkpoint strict-load parity
does not eliminate these differences. A future comparison should treat
posterior sampling, precision and chunk/tile choices as distinct controlled
factors, not attribute any discrepancy solely to VAE compression.
https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/blob/5cb04e82fed71f2fe64a2617ab695be1f5a37fea/configs/config_network_rflow.json

## Independent automatic judge: candidate, not ready

MONAI pancreas_ct_dints_segmentation targets background0, pancreas1, tumor2;
compatible with MSD masks. Official metadata says trained on Task07Pancreas.
The model is independently implemented, NOT established independent of these
patients. Inference resamples to1mm, clips[-87,199]HU, and uses96cubed sliding
windows with .625 overlap, sw_batch_size4. Keep its own preprocessing identical
on both inputs and map predictions back to original GT geometry before scoring.
Do not confuse its intensity range with MAISI's input range.

Public HF package revision1b4b04a0de2cf6236860891bf1c2e9494e1afaf7 has a
~554MBmodel.pt and scripts/prepare_datalist.py but NO configs/dataset_0.json.
The split generator uses seed123, train_size196; reconstructing that intended
split would not prove this checkpoint was trained on exactly that split or
dataset version. Need the actual released checkpoint training split, author
confirmation, or patient-disjoint data before a held-out detection claim.

No weights downloaded and no judge inference run. Dependencies also differ:
bundle metadata specifiesMONAI1.4/torch2.4; do not install into the shared
trainer environment or assume compatibility. Exact runtime/cost unmeasured.

Sources:
https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/models/pancreas_ct_dints_segmentation/configs/metadata.json
https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/models/pancreas_ct_dints_segmentation/configs/inference.yaml
https://raw.githubusercontent.com/Project-MONAI/model-zoo/dev/models/pancreas_ct_dints_segmentation/scripts/prepare_datalist.py
https://huggingface.co/MONAI/pancreas_ct_dints_segmentation/tree/1b4b04a0de2cf6236860891bf1c2e9494e1afaf7

Another candidate, the official PANORAMA baseline, publishes weights and folds
but is a TWO-STAGE pancreas-localization/PDAC detector. It uses its own ROI
pipeline and targets PDAC rather than all MSD tumor histologies. Need actual
case histology/domain compatibility, cohort overlap checks, and matched ROI
handling before using it as a valid judge. Its documented pipeline should not
be replaced by GT-centered crops to manufacture apparent sensitivity.
https://github.com/DIAGNijmegen/PANORAMA_baseline
https://zenodo.org/records/11160381

For any judge, prespecify thresholds/lesion matching from independent validation
or the released protocol, not these two reconstructions. Run native/control/
reconstruction identically, retain full probability maps, map to GT geometry,
score each lesion and false positives. If the control lesion is already missed,
a reconstruction miss is not evidence of compression-induced loss. Additional
GPU cost must be measured and capped separately; current remainder~.08889
allocation-hours is NOT assumed sufficient for detector setup or inference.

## Ready next gate

Blinded A/B packet from the existing two reconstructions includes ALL tumor-
containing axial slices. Private condition key retained remote, not committed
or sent in reviewer packet. Need a qualified reader and their actual ratings;
no clinical/independent reader result exists yet. No new GPU work for this audit.
