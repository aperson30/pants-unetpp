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

## Resource correction and explicit amendment before new outcomes

3295549 failed in38 allocation seconds at the free-memory gate on BOTH cases,
before reconstruction/detector output:0.021111 charge-equivalenth. Retain both
failures and frozen v1 files. No clinical or compression result from this job.
NCSA architecture says GPU96GB/CPU120GB, while Slurm names its GRES120gb and
the software page also contains inconsistent120GB/~97GB usable language.
Actual runtime guard proves free memory <110GB, NOT an exact capacity reading.
Earlier120GB HBM sizing premise was not adequately verified; corrected here.
https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/architecture.html
https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/software.html

No blind lowering of the larger-volume gate. Large patient100259 remains
resource-blocked; no substitute. Smaller locked100430 gets an explicit33M
including-padding resource cap and >=65GB actual-free-memory gate. Pilot
scaling estimates~52GB for its32.5M padded voxels, leaving a meaningful margin
on the documented96GB GPU; estimate still not a proven bound. No changed
pixels/spacing/model settings. New code prints actual GPU memory before gate.
One case means at most two completed patients with pilot, not a three-patient
study; report the larger patient as unavailable, not an original-image miss.

3295577 verified held and released PENDING:1GPU/32GB/10min, no requeue,
540s process timeout. Sources/selection/batch SHA and bash/Python syntax pass.
The five integration/geometry tests passed before this resource-only amendment;
prediction weight contract and all neural operations remain unchanged.
Campaign estimate0.605833/2 chargedh; maximum new0.333333 =>0.939167 worst-case.
Do not automatically retry or change posterior/chunking to force the large case.
Any later larger-volume protocol needs its own compatible hardware/memory plan.

## Host-memory OOM and resource-only correction

3295577 FAILED1:0 in114s; step3295577.0 OUT_OF_MEMORY0:125 and logged Slurm
oom_kill. No completion or arm results. Charge-equivalent0.063333GPUh.
Actual logged CUDA free101495996416/total102087458816 bytes; GPU65GB gate
passed. Slurm host RAM limit32G. Sampled MaxRSS17328064K does not capture
the instantaneous killing peak and must not be cited as the true peak.

Installed MONAI1.5.1 MaisiGroupNorm3D.forward uses CPU concatenation when
max(inputs[0].size())>=500; _cat_inputs clones to CPU and repeatedly torch.cat
before returning to GPU. Thus512-wide volumes require substantial host memory
even with enough HBM. This supplies a concrete explanation consistent with
the Slurm OOM kill; failure location was not captured in a Python traceback.
No claim that GPU allocation remained below capacity at every instant.

One explicit resource-only retry3296011 verified held then released PENDING:
host96G instead of32G, same1GPU/billing2000,10min/no-requeue/540s timeout,
same patient and pixels, precision, VAE operations, detectors and65GBfree gate.
Added CPU peak-RSS and CUDA allocated/reserved logs at model/encode/decode/
release stages; no synchronization or model changes. Private small_v2 preserves
v1 artifacts. Frozen hashes/bash syntax and five CPU weight-loop/geometry
tests pass1.408s. Larger patient remains blocked, no new case or method.
Campaign estimate0.669167 used; max with new job1.002500/2 charged GPUh.
Posted debit unknown. Further failure requires fresh diagnosis, not auto-retry.
