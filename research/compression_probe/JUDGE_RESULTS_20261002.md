# Two completed patients: useful score-change lead, not clinical recall evidence

| Diagnostic | Pilot100226 | Locked replication100430 |
|---|---:|---:|
| Native/control mean tumor PDAC probability |0.171565|0.030071|
| Reconstructed mean tumor PDAC probability |0.023676|0.007681|
| Relative decrease |86.20%|74.46%|
| Native GT-overlapping candidate voxels |358|0|
| Reconstructed GT-overlapping candidate voxels |249|0|
| Native/reconstruction crop identical |Yes|No|
| GT retained in crop and postprocess mask |100%|100%|

Native/clipped tumor scores are identical within each case. CPU audit of actual
saved maps reproduces all reported means, maxima and candidate-overlap counts.
Neither case demonstrates a conversion from a GT-overlapping candidate to no
candidate. No independent clinical threshold was fixed; neither candidate
presence nor confidence changes establish recall, accuracy or irreversible
task-information loss. Frozen-detector domain shift remains unresolved.

For100430, native patient maximum0.732969 is elsewhere, not in the GT tumor.
Audited dynamic-fast cutoff0.293187 exceeds GT maximum0.233829. Reconstructed
cutoff0.221787 exceeds GT maximum0.103914. This posthoc diagnostic explains
why tumor candidate overlap is absent in both arms; it is not a tuned clinical
threshold. Do not count patient maximum as tumor detection or claim a newly
missed tumor. Reconstruction crop starts three slices earlier; retaining100%
of tumor does not eliminate context/window-placement confounding. Pilot crop
is unchanged, so that particular confound does not explain its decrease.

## Runtime and failure evidence

3296011 completed0:0 in220 allocation seconds,0.122222charge-equivalentGPUh.
VAE153.53s, GPU peak50.029GB. Producer ru_maxrss40314496KiB (~38.45GiB)
exceeds old32GiB host request;96G correction succeeded without scientific
changes. Slurm sampled MaxRSS21736896K is not the true instantaneous peak.
Campaign estimate0.791389/2chargedGPUh including recorded failures; posted
debit unverified. Big locked100259 remains resource-blocked, not replaced.
Protected PanTS evaluation47320183 unchanged. No new training.

## Next information-per-resource gate

3296141: one already-completed pilot case, compare MaisiConvolution splits4
against frozen splits1 reconstruction. This implements overlapping internal
convolution partitions, not a smaller patient crop or skipped context. Keep
full-volume GroupNorm, FP32, posterior mean, spacing, padding and weights.
TopologyCPU preflight confirms36 affected convolution modules, split axis1;
syntax and frozen source/reference SHA pass. Held resource fields checked,
released PENDING,1GPU/96G/5min/no-requeue,270s timeout. Max0.166667chargedh;
campaign worst-case0.958056/2. No duplicate or automatic retry.

Prespecified engineering screen: global maximum HU drift<=0.1 and finite
output. This arbitrary small implementation tolerance is NOT a medical
safety bound or detector-parity proof. Measure global/tumor drift, GPU/CPU
peak and elapsed time; retain failed gate without relaxing it afterward.
No automatic rollout to larger cases. If drift is small and memory materially
improves, a separately reviewed larger-case runtime becomes plausible. If
memory does not improve, stop this lever. Posterior sampling and eventual
independent/task-adapted judges remain distinct future questions; do not train
a remedy yet or claim an S-tier idea from these two feasibility cases.

Scalar originals and map audits are preserved alongside this report. Images,
maps and clinical workbook remain private; no raw patient data in GitHub.

## Split gate completed: parity passes, memory lever rejected

3296141 COMPLETED0:0 in114 allocation seconds, billing2000 =>0.063333
charge-equivalentGPUh. Scalar result split_parity_result_3296141.json.
Global and tumor maximum/mean HU differences exactly0 on this case; numerical
gate passes. GPU peak24250846720 bytes EXACTLY matches split1. Split4 elapsed
83.21s vs frozen split1 72.89s (~14.2% longer). Single cross-job timing, not
a repeated throughput benchmark; no speed advantage claimed. CPUpeak~23.04GiB.

Do not adopt split4 as a demonstrated memory optimization or assume it unlocks
the55M case. Full-output parity on one patient is not all-shape/clinical parity.
Stop this lever per prespecified decision rule. Campaignestimate0.854722/2
chargedGPUh, posteddebitunverified. No further GPU job submitted by this update.

Scientific next priority is adjudicating evidence, not adding method training:
native/reconstructed candidate status did not switch in either completed case;
only confidence weakened. Review clinical-threshold/independent-judge and
posterior-mean versus sampled-embedding controls before a broader harm claim.
Larger locked patient remains blocked; any larger-memory hardware/runtime
proposal needs its own preflight, no crop-based rescue or case substitution.
