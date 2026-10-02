# Approved bounded control round

User approved2 charged GPU-hours TOTAL for this new screening round, including
calibration, failures and evaluation. This is separate from prior campaigns.
No automatic retry, campaign expansion or environment mutation.

## Submitted jobs

| Job | Probe | Partition | Hard limit | Verified billing | Maximum charged hours |
|---|---|---|---|---:|---:|
|3290857|Matched insertion pair|ghx4-interactive|10min|2000|.33333|
|3290863|Real CT num_splits4|ghx4|10min|1000|.16667|
|3290864|Real CT sampled posterior|ghx4|10min|1000|.16667|
|Total|||||.66667|

Billing is normalized to1000 per one standard GPU-hour. Maximum reserve,
not actual debit. Remaining uncommitted campaign budget>=1.33333 hours.
One submitted interactive job per user is enforced by qos_ghx4int; second
interactive submission was rejected without any allocation. The other two
were submitted to regularghx4 instead; no attempts to bypassQOS restrictions.

Every job:oneGH200120GB,eightCPUs,96GhostRAM,one node,Requeue=0;
540s process watchdog plus10s termination grace. Slurm stored scripts byte-
matched the staged template, all dependent source hashes verified, jobs
initially held for resource/billing verification then released together.
Last check:insertion RUNNING ongh092; other two PENDING(Priority). Allocation
alone does not prove successful execution; require metrics+completion+exit0.
No reliable queue-start estimate was obtained for regular jobs.

## What is being tested

Same cached case120 CT/mask, pinned model,MONAI1.5.1,FP32,native spacing,
whole-volume inference, no crops/tiles. Input/model hashes checked. Current
preprocessing output exactly equals cached control before anyGPU run.

1. Insertion:one8mm/-20HU additive sphere in the largest-clearance fixed
   source-mask parenchymal location. Center225,259,74;111voxels at native
   spacing. Preserve underlying texture. Reconstruct unmodified and modified
   whole volumes in SAME job/model. Measure reconstruction difference inside
   insert and surrounding2-5mmring, raw and ring-corrected signed response.
   Not a matched-filterSNR, detector, realisticPDAC or healthy-host experiment.
   Host already has a real tumor. Radius/contrast not tuned after outcomes.
2. Split4:change36MaisiConvolution layers'num_splits to4, no other model
   setting changes. Compare against cached mean/splits1 reconstruction inHU,
   report max/mean whole-image and tumor-region differences. This is a real
   image/GPU extension of the CPU toy contract, not a full production replay.
3. Posterior:one fixed seed20261001 sampled latent, split1,FP32. Compare
   against cached mean reconstruction. Measures one example of posterior
   variation, not a distribution, equivalence or patient-level detection harm.

Cache comparisons are NOT same-allocation repeatability tests. CaptureGPU,
torch/MONAI,source manifest,baseline hash,seed,latency and memory. If changes
appear, separate runtime/config variation from lesion fidelity before claims.

## Efficiency and scientific limits

Use working DeltaAI environment/cache; Bridges master check failed, and no
Delta master was available. Avoid cross-cluster setup/transfers for these
small controls. Parallel Slurm requests permit overlap when scheduling allows;
do not equate faster wall-clock with lower charged GPU-hours.

Metrics-only output by default saves shared storage; no new CT images uploaded
or volume outputs saved. Existing raw inputs/baseline retained, source/config/
placement recorded for reproduction. --save-volumes is optional, NOT enabled
in submitted scripts. lfs groupdelta_bdyo reports~984GiB usage and no live
block limit in this check; this does not revoke documented quota/storage
cautions. Avoid writing hundreds ofMB merely because rawFS has headroom.

Two insertion unit tests passed in isolated remotePyTorch/MONAI runtime;
all three modes passed CPU preflight with real weights and verified real inputs.
Original12metric/ROI tests previously passed locally. test_insert_signal.py
requiresPyTorch/MONAI and should be run in that isolated runtime; local minimal
metric venv does not provide those dependencies. No dependency install needed.

Real-lesion PANORAMA detector and radiomics experiments NOT submitted. Patient
mapping/checkpoint-selection exposure/download/runtime gates remain. These
three controls do not replace those experiments or a qualified reader study.
Do not expand a favorable toy result into a clinical claim.

PanTS evaluation unchanged. No image-egress permission orMFA bypass granted.
Source code:bounded_controls.py; launcher:bounded_controls.sbatch;
source manifest:controls_source.sha256. Remote logcontrol_<job>.log;
resultscontrol_<mode>_<job>/{provenance,metrics,completion}.json.
