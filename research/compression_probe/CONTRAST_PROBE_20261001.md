# Contrast/phase probe: submitted, results pending

## October 2 update: original storage stop and corrected submission

Job3291031 FAILED1:0 after36allocationseconds, before GPU inference. The
project filesystem reports0freebytes, so the original guard correctly stopped
instead of filling quota or silently losing metrics. This consumed0.010000
physical/charged-equivalentGPU-hours (billing1000); no automatic retry occurred.

Corrected job **3291159** submitted held, frozen script/resources verified,
then released. Last observed PENDING. Same scientific experiment and18min/
0.30charged-hour maximum. Small source/logs/JSONs now live in
`/u/asanjeev/compression_probe_smalloutputs`; CTs/weights remain read-only in
the old projectroot. Do not treat the original paragraph's queue state as current.

Check actual output parent before heavy input loading, including a1MiB temporary
write/fsync probe to test writeability/quota; raw filesystemfree alone is not
treated as per-userquota proof. `lfs quota` cannot query this non-Lustre home
mount. Actual homewrite passed; no dataset deletion or globalruntime change.
All4unit tests and corrected real-data CPU preflight pass. The cleanup test
proves the probe leaves an existing sibling file unchanged. No CTimages saved.

New-round actual spending estimate0.222222chargedh; with this reservation the
ceiling is0.522222of2hours. Posted debits still not independently verified.
No more retries are automatic; investigate any new failure before another job.

## Original submission record (historical)

Job **3291031**, DeltaAI `ghx4`, one GH200 GPU, 8 CPUs, 96 GB host RAM,
18-minute Slurm cap, 1020-second process watchdog plus 10-second kill grace,
no automatic retry/requeue. Actual held-job `ReqTRES` verified billing=1000,
gres/gpu=1 before release. Frozen scheduler script inspected. Last observed
state after release: PENDING; no output/quality conclusion yet.

Maximum additional charged cost: **0.30 GPU-hours**. Prior new-round actual
charge estimate remains 0.212222; combined actual-plus-reserved ceiling is
0.512222, within the user's two-hour total screening authorization. Pending
time is not counted as GPU allocation time. Posted allocation debits are not
yet checked. This is one discriminating experiment, not a larger sweep.

## Fixed design

- Cached MSD120 tumor-positive host, original image/label/checkpoint hashes
  verified. Same whole native-volume FP32/mean MAISI path; no crops/tiles.
- Reuse original binary 8 mm, 111-voxel insert centered at [225,259,74].
  Supersampling is intentionally not introduced: it would change the stimulus.
- Seven forward passes: original baseline; -20 HU repeated; -10, -40, -80,
  +20 HU at the original location; -20 HU after a +2-voxel x translation.
  Translation is 1.367188 mm on this case, not 2 mm or a physical slice shift.
- All amplitudes and the shifted mask passed CPU clipping/ROI checks. No
  per-amplitude location selection or shrinkage. Original insert regenerated
  exactly. Shift has no wraparound and preserves all111voxels.
- Baseline must match the frozen cached reconstruction within max0.001HU;
  otherwise stop after that pass. First-pass timing must predict completion
  inside the watchdog (with overhead allowance); no budget extension.
- Raw and 2-5mm ring-corrected response divided by signed inserted contrast.
  Negative signs are retained. Outputs saved one pass at a time; no image
  publishing and no accumulation of full-volume tensors between passes.
- Every pass uses `decode`'s CUDA tensor release/empty-cache discipline. Only
  scalar records accumulate. Output directory exclusive; JSON completion only
  after all seven passes. Failures never become a successful "done" marker.

## Validation before release

Three CPU unit tests passed on the isolated MONAI import path: a known linear
operator gives0.4retention for every amplitude/sign; mask translation preserves
voxels without wrapping; clipping/zero contrast/ROI escape are rejected.
Real-data CPU preflight passed, including strict checkpoint load and small
encode/decode shape/finite contract. Final version/source hashes checked;
shell syntax valid. Training environment unchanged.

Source: contrast_probe.py, test_contrast_probe.py, contrast_probe.sbatch.
Remote checksum manifest covers runner, helpers, metric import and pinned MONAI
wheel. The protected PanTS evaluation job47320183 was not queried or changed.

## Decision interpretation fixed before GPU results

Claude proposed these screening heuristics: -10HU corrected retention below
0.7times -80HU retention, or +20/-20 signed corrected retentions differing by
more than0.15. Report actual signed values, sign reversal and phase-control
change, not just a binaryGO. A ratio becomes uninformative near zero or with
sign reversal; do not force a ratio-based verdict then. One shifted point does
not establish the maximum phase sensitivity of other amplitudes/locations.

Amplitude/sign dependence rejects a fixed-linear-response explanation for this
construction; it does NOT identify learned-prior suppression as the unique
cause. Normalization, encoder/decoder nonlinearities, aliasing and host context
are alternative mechanisms. Flat local retention does not prove a globally
linear VAE or clinical safety. This is one host, not independent patient samples,
real PDAC, a detection endpoint or a publishable cohort result.

Do not make a syntheticNO-GO a mandatory barrier to a realistic lesion test.
Use the outcome to choose mechanism work versus a provenance-cleared judge
test, not more uncontrolled contrasts. Do not allocate further GPU work merely
because unused budget remains.

## October 2 measured response (job 3291159)

Same-allocation baseline matched the frozen reconstruction exactly: maximum
and mean difference 0 HU. Repeated -20 HU response matched the earlier probe.

| Inserted contrast | Raw signed retention | Ring-corrected retention |
|---|---:|---:|
| -10 HU | 0.37717044 | 0.31356637 |
| -20 HU | 0.43564539 | 0.37075455 |
| -40 HU | 0.58170099 | 0.51714212 |
| -80 HU | 0.76969829 | 0.72027682 |
| +20 HU | 0.29573705 | 0.23594213 |

These are retention of an inserted regional signal, not segmentation accuracy,
tumor recall or clinical detectability. The -10/-80 corrected-retention ratio
is approximately 0.4353, crossing the predeclared amplitude-prioritization
heuristic. The opposite-sign difference at 20 HU is 0.1348, below the proposed
0.15 heuristic. This does not make sign dependence absent or clinically safe.
Amplitude dependence is clear in this construction; its mechanism is unresolved.
The +2-x-voxel shifted -20 HU insert retained 0.43730130 raw and 0.37474595
ring-corrected, versus 0.43564539 and 0.37075455 unshifted: a corrected change
of 0.00399140 (0.399 percentage points). This single phase control does not
explain the observed amplitude curve, but does not exclude other phase effects.
All seven passes completed; completion marker true, no volumes saved.
Slurm verified COMPLETED, exit 0:0, allocation elapsed 815 seconds (13:35).
Inference-stage elapsed was 770.527 seconds; allocation time includes setup.
At regular one-GPU billing equivalence this is approximately 0.226389 charged
hours, not a verified posted allocation debit. Including the failed 36-second
attempt and earlier three controls, NEW-round total estimate is 0.448611 of
the authorized 2 hours. No automatic retry or additional GPU job submitted.
Peak CUDA allocation was 41,980,169,728 bytes per pass; each measured altered
volume pass took approximately 108-110 seconds. No images were saved/published.
