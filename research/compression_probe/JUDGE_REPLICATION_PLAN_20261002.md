# Two-case replication plan, locked before new detector outcomes

## Motivation and completed pilot

Job3295395 completed0:0 in126 allocation seconds:0.070 charge-equivalent
GPUh at interactive2x. Scalar result preserved in paired_judge_result_3295395.json.
Native/clipped mean tumor PDAC probability0.1715648 versus reconstructed
0.0236760 (~86.2% lower); GT-overlapping candidate confidence0.821785 versus
0.125032 (~84.8% lower). Crop and dilated mask retain100% of GT in all arms;
identical native/control scores and identical crop bounds. These are model
scores, NOT clinical accuracy or recall. Single case, validation-selected
checkpoint and possible detector domain shift constrain interpretation.

## Locked selection and spending

Two additional patients:100430_00001 and100259_00001. Rule: the two smallest
stored batch1 histopathology-positive fold4 cases under60MB, excluding pilot.
Pinned eligibility/clinical hashes checked; all three distinct patient IDs
verified privately. No outputs used for selection. This is deliberately
resource/size-biased feasibility sampling, not representative patient sampling.
No replacement for original misses, failures, size guard rejection or small
effects. Full volumes only; GT scoring-only, no GT crop or threshold tuning.

Re-use exact pilot detector/VAE implementation, weights, mean posterior FP32,
input transforms, all three arms and raw/masked/candidate scores. Preserve
original-image misses and localizer failures explicitly. Never call candidate
overlap at an unfixed threshold a clinical detection event. Report each paired
change, crop/mask inclusion, and failure status separately. No pooled recall,
AUROC, significance claim or voxel-as-independent-sample analysis.

CPU download first: bounded Range/CRC, pinned labels, image-label geometry,
finite data, exact IDs; no GPU occupied during staging. Original32M resource
cap remains default; replication explicitly raises it to56M including padding,
before new detector outcomes. This changes no input pixels/spacing or model
operation. Case shapes512x512x123 and512x512x210:32.24M/55.05M voxels.
Pilot peak24.25GB for15.20M voxels suggests approximately88GB for the larger
case on120GB HBM, a sizing estimate NOT a proven memory bound. Runtime requires
at least110GB actually free GPU memory before a larger-volume run. OOM is retained
as a pipeline/resource failure, never rescued with a smaller crop. No
cropping/downsampling to fit. OneGPU/32GB/20min hard cap for both
cases, sequential new processes to release GPU allocations, no requeue.
Each case has480s timeout. Campaign estimate after pilot0.584722/2 chargedh;
maximum replication0.666667 => worst-case total1.251389. Posted debit unknown.
Protected PanTS evaluation47320183 and its environment unchanged.

## Decisions after results, not a promise of novelty

If loss repeats with retained tumor crop/mask, prioritize an independent
mechanistic control and the posterior-sampling distinction, then seek PI
alignment before a larger cohort or remedy training. If effects are mixed,
report heterogeneity rather than hunt for favorable cases. If native scores
are weak or pipeline fails, diagnose judge feasibility without labeling
compression harmful/safe. A drop alone cannot separate frozen-model domain
shift from irreversible task-information loss. No training planned yet.

All CT/masks/maps stay in private home storage, not GitHub. Publish code and
scalar findings only. Current preparation/transfer state belongs in handoff.
