# Approved bounded control round

## Completed results — supersedes queued/running snapshots below

All three jobs COMPLETED,ExitCode0:0,metrics+completion artifacts verified.
Torch2.10.0+cu129,MONAI1.5.1,GH200120GB; identical cached-baseline hash
a2d31659c02f2d2b51eea9d0bf97656ed6cb477a204187977d47131fcd1ac1c1.

| Job | Allocation elapsed | Actual physical GPU-hours | Approx charged hours |
|---|---:|---:|---:|
|3290857|248s|.068889|.137778|
|3290863|126s|.035000|.035000|
|3290864|142s|.039444|.039444|
|Total|516 GPU-allocation seconds|.143333|.212222|

Remaining NEW2hour cap:1.787778charged hours. Values calculated from actual
Slurm allocation elapsed/billing multiplier, not posted account debits. No
failedGPU allocation or retries in this round. Prior campaigns are separate.

### Observations, not clinical conclusions

- Insertion:-20HU input signal; paired output response-8.71291HU inside
  inserted region(43.5645% raw retention). Surrounding2-5mmring response
  -1.29782HU; ring-corrected response-7.41509HU(37.0755% retention).
  Baseline/insert forwards107.53/105.59s,peakCUDA39.098GiB,
  RSS~35.266GiB. Strong attenuation for this ONE synthetic placement.
- Split4:whole-image mean/max and tumor mean differences from cached
  splits1baseline all0HU. Forward92.02s,peakCUDA39.098GiB,RSS~40.886GiB.
  Installed source inspection confirms num_splits>1 takes the actual chunk
  branch;36layers changed. Not proof of general parity or a timing speedup:
  baseline/current measurements are separate jobs/nodes,not paired timing.
  Splitting did not lower measured peakCUDA allocation here.
- Posterior:single seed20261001 versus cached posterior mean yields
  whole-imageMAE21.0671HU,max544.043HU,tumor-regionMAE21.1752HU.
  Forward108.03s,peakCUDA39.105GiB,RSS~34.935GiB. Difference is NOT error
  against originalCT,not loss of detection,not a distribution estimate.

### CPU Gaussian reference (post-hoc, illustrative)

Computed linear Gaussian response to the EXACT same rasterized111voxel/-20HU
signal; measurement halo covers insert+5mmring+4sigma,maxsigma2mm. This is
CPU measurement cropping only; model passes remained whole volume. Because
Gaussian filtering is linear, G(host+signal)-G(host)=G(signal); no full host
filter orGPU inference needed for this particular insertion-response statistic.

| Gaussian sigma mm | Raw response retention | Ring-corrected retention |
|---|---:|---:|
|.5|.901545|.901541|
|1|.792653|.790769|
|2|.480010|.449950|
|VAE measured|.435645|.370755|

Sigma choices.5/1/2mm were selected for this descriptive check AFTER the VAE
result,not preregistered. No noise-power matching,scannerPSF matching,matched-
filter observer,clinical threshold or statistical superiority established.
The VAE response is below these particular references; do not call this a
VAE-specific clinical failure. A stronger blur could also attenuate more.
Raw JSON/log stay remote:gaussian_response_v1.json/.log.

### Efficiency audit and next steps

Efficiencies verified:shared cache/runtime,no new CT/weight downloads;
CPU preflight/analysis offGPU;exact oneGPU perjob;regular partition1x for
two probes;short independent requests ran on three separate nodes;no auto
retry;metrics-only output and hash/source/job identity gates. This is an
audited bounded workflow,NOT proof every possible optimization is exhausted.

Remaining avoidable work in future runners:split/posterior modes unnecessarily
construct synthetic masks and modifiedCTs before selecting their branch;
prepare() computes anROI merely to validate labels and insertion repeats that
work. Remove those CPU-only allocations in a FUTURE version,with tests; do
not mutate historical frozen scripts/results or rerunGPU jobs just for this.
Unused model tensors can be released before decode where compatible; peak
memory changes require measurement,not a blanket claim. Full FP32 and whole
volume were deliberate quality/control choices,not accidental slow defaults.

Metrics-only retention is a tradeoff:these new altered volumes cannot now be
used for arbitrary retrospective texture/spatial metrics without recomputing.
Original mean reconstructions for005/120/165 ARE saved already and should be
used first. For future runs,define needed endpoints beforehand and either
compute them online or retain small same-grid measurement patches privately,
with indices/affine/source hashes. Output patches do NOT justify model cropping.

Prioritized next plan:
1. CPU extraction implementation/config tests then descriptive paired texture
   measurements on already saved005/120/165;no new VAE inference required.
   No duct diameter,diagnosticAUC or early-cancer conclusion from these scans.
2. CPU PANORAMA checkpoint->fold/patient-exclusion and single-case access
   audit in parallel;no data-independent claim until actual verification.
3. If useful,prepare a NEW prespecified insertion position/host(case165cached)
   before outcomes to check location/slice-thickness dependence.8mm sphere on
   4mmslices is coarsely rasterized; report actual voxel mask/physical burden,
   not nominal diameter alone. One current result does not define a response
   curve. Lower slice thickness still confounds cases;not causal isolation.
4. Real-lesion detector calibration only after metadata/runtime gates;count
   negative cases and both automatic localization/detection stages. Start one
   case with explicit walltime/billing reserve,not30+30blindly. Fixed-setting
   paired effects,not independently retuned classifiers or thresholds.
5. Additional posterior seeds/precision/spacing controls only if needed by
   the selected question. One seed proves paths differ,not a variance estimate.
   Do not rerun chunking simply because it looks faster in this one timing.

No newGPU job submitted during this audit. PanTS evaluation untouched. Further
probes must fit actual remaining cap and preserve source/runtime/data gates.

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
