# Independent judge: a concrete lead, not launch-ready

Verified primary-source lead: DiffTumor authors release pancreatic tumor
segmentation checkpoints for U-Net, nnU-Net and Swin UNETR, trained on real and
synthetic tumors. U-Net weight URL is explicit in their README:
https://github.com/MrGiovanni/DiffTumor
https://huggingface.co/MrGiovanni/DiffTumor/resolve/main/SegmentationModel/unet_synt_pancreas_tumors.pt

Their actual validation code defines a MONAI3D residual U-Net with channels
(16,32,64,128,256),3outputs, strides2x4,num_res_units2. It applies RAS,
1mmresampling, HU[-175,250]->[0,1], minimum96cubepadding,96cube slidingwindows,
batch1,75%overlap,Gaussian weighting. It then replaces the organ channel with
an external organ_pseudo mask and suppresses tumor outside that mask.
https://github.com/MrGiovanni/DiffTumor/blob/main/STEP3.SegmentationModel/validation.py

Why this matters: a published checkpoint alone is NOT a verified end-to-end
judge. Need identify/pin external organ-pseudo producer and label map, validate
preprocessing/native-grid restoration and strictstate loading. Do not treat
their label2tumor as PANORAMAlabel1 or PanTSlabel28. Avoid their optional
strict=False load path for our compatibility preflight. Legacy transforms
AddChanneld/metadatainversion require checked MONAI1.5.1 equivalents, not a
blind modernport. Runtime needs real-shape calibration before GPUhours claim.

The checkpoint is a separatelytrained model, not necessarily patient-independent
or clinicallyindependent. MSD and PANORAMA datasetnames differ but patient
disjointness, checkpointfold and healthy/synthetic-source overlap are not
verified. Another CNNjudge can still share the same domainshift; agreeing
frozenjudges would strengthen task-response evidence, not prove irreversible
loss or replace a readerstudy. No known clinicalthreshold from our probe.

Cheapest safe nextpreflight: pin source/weights, verify license/provenance,
restrictedweights-onlyCPUload, exactarchitecture/keyshape match, tinyCPUforward
and transform/nativegrid regression. No trainable adaptation or new GPUjob
authorized by this document. Only then propose a bounded full-context inference
with both native/reconstructed controls and frozen tumor-specific metrics.
Do not silently substitute GT organs, a reducedcrop or unverified mask to make
the pipeline easier. OneGT-assisted score would not be end-to-end detection.

No weights downloaded, no GPU spent, no independentjudge score obtained here.
PANORAMA fold4 currently remains the ONLY actually executed tumorjudge.
