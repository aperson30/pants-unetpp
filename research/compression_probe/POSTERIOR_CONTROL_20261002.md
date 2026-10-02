# Locked posterior mean versus sampled-latent control

Question: do the observed frozen-detector score drops depend on using the
posterior mean instead of a sampled latent? Pinned MONAI1.5.1 encode returns
mu and sigma=exp(logvar/2), i.e. standard deviation. Its sampling implementation
is mu+randn_like(sigma)*sigma. encode_stage_2_inputs samples; reconstruct uses
the mean. Verified against pinned installed wheel, not inferred from API names:
https://github.com/Project-MONAI/MONAI/blob/1.5.1/monai/networks/nets/autoencoderkl.py
The MAISI embedding tutorial calls encode_stage_2_inputs under autocast and
resizes to rounded dimensions. This screen stays full native-grid FP32 and
therefore is NOT an exact reproduction of that full embedding pipeline:
https://github.com/Project-MONAI/tutorials/blob/main/generation/maisi/scripts/diff_model_create_training_data.py

Reuse completed pilot100226: smallest already-processed full scan, NOT a new
case selected by fresh outcome. It has a historical native GT-overlapping
candidate; selection/feasibility bias must be reported. One patient only.
Encode whole native-spacing RAS-reoriented CT ONCE, same clipping/normalization
and multiple4 padding, original VAEweights/FP32/GroupNorm/num_splits1/decoder.
Decode fresh mean and samples at seeds0,1,2 fixed before outcomes. Use pinned
model.sampling, never sigma squared or a hand-picked seed. No output clipping,
tiling, anatomical crop in VAE, new training or diffusion-generated sample.

Six detector arms: native, clippingcontrol, freshmean and ALL three seeds.
One saved publisher-native window reused for all arms; GTlabel1 scoring-only.
Official fold4 stage2 frozen preprocessing/mirroring/postprocessing retained.
Each arm predicted twice under measured strictdeterministic policy; all7class
max repeatdrift<=1e-4 and identicalsegmentation required, otherwise failclosed.
Nativecontrol is fresh in SAME process; oldhistoricalreplay gate remainsfailed.
Record rawtumormean/max, candidateoverlap/confidence, mask/cropinclusion,
posterior sigmastats, seeds, actualresource/time/backends and retainprivate
maps/images for audit. Report all arms, not bestseed or patientmax as detection.

Decision: if sampled latents consistently preserve/recover native detector
response while mean does not, narrow/revise the mean-based bottleneck story.
If all samples remain low, mean-only choice is insufficient explanation on
thispatient; frozen-judge domainshift/clinicaladequacy still unresolved.
Heterogeneous samples reported as such, not averaged into a success claim.
Candidatepresence/confidence is NOT validated clinicalrecall; no independent
judge, informationerasure, diffusiontraining or S-tier/novelty claim from this.

Engineering: pinned sampling identity/zero-sigma/seedvalidation CPU tests,
oneencode/fourdecodes/12detectorcall orchestration CPUmock regression and
geometry/scoringtests before GPU. Wholevolume padded16Mcap and40GB actualfree
GPU guard;96Ghost request from priorCPUoffload evidence. Homefree>=4GB,
uniqueprivateoutput, sourcehashes, no autoretry, preservefailure.
OneGPU/2CPU/96G/hard6min/330sprocess, max0.2charge-equivalentGPUh at billing2000.
Priorcampaignestimate0.947500/2, worst1.147500/2, posteddebitunknown.
No main47320183 change, large100259 remainsblocked/unreplaced.

## Submission

3297185 released after6CPUtests, bashsyntax/frozensource and heldresource
checks; lastPENDING. Request1GPU/2CPU/96G/6min, billing2000,no-requeue.
Actualhomequota7.364GBused against102.4GBsoftquota (not rawfilesystemcapacity),
plenty for thissmallcase; outputsprivate underfreshposterior_control_v1.
Weights/16Mpaddedshape/40GBactualfreeCUDA guards checked atruntime too.
No automaticretry or dataset/model/clinicalthreshold modification.
