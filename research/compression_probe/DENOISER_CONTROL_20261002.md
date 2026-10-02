# CPU-only classical-denoiser comparison: locked preparation

Purpose: determine whether two fixed classical denoising controls can reproduce
the observed one-host MAISI signal-response curve under a limited baseline
texture-matching rule. No clinical endpoint, learned-prior proof, noise-power
equivalence or model-independent information-loss conclusion.

No GPU allocation. Runs on the local Windows workstation, one CPU thread,
not on an HPC login node. Protected PanTS evaluation 47320183 is untouched.
Cached input image/label/reconstruction reused; no new VAE passes. Task-local
scikit-image0.25.2 and lazy-loader0.4 wheels only, no shared/global install.

## Frozen comparison

- Same canonical case120 clipped control and saved mean/FP32 MAISI baseline;
  exact hashes and grids required. Original 8mm binary insert, center225/259/74,
  111voxels. Contrasts -20/-10/-40/-80/+20HU and shifted-20 (+2xvoxels).
- Same raw/ring-corrected retention, float32 regional-mean convention, 2-5mm
  ring generated on full source grid. All insert amplitudes checked for clipping.
- Fixed families: scikit-image TV-Chambolle and fast3D non-local means. No
  family switching based on results. TV weight is in native voxel-index space,
  not physical-gradient-weighted TV. NLM patchsize3/searchdistance2/sigma0,
  preserve_range=True. On the anisotropic grid, these voxel kernels have unequal
  physical extents: report this limitation, never call them physically isotropic.
- The library's bilateral implementation is 2D; do not silently apply it as3D.
  Official API: https://scikit-image.org/docs/stable/api/skimage.restoration.html

## Calibration and holdout

Fixed parenchymal source ROI excludes real tumor by5mm and organ edge by2mm.
Exclude locations within10mm of either inserted mask (outer ring5mm plus
another5mm clearance). Split eligible source positions at the median coordinate
of the longest-extent axis; use one half to fit and the other to report transfer.
At least1000voxels per half required. This is spatial holdout in ONE patient,
not an independent-patient validation split or pure homogeneous-noise sample.

Match standard deviation of `image - Gaussian(image, sigma=2mm, truncate=4)`
in fitting ROI to that of baseline VAE. Call this **high-frequency texture SD**,
not noise SD, SNR or detectability. Anatomy contributes to the statistic.
Baseline-only matching is outcome-calibrated to the VAE, not a fully independent
control; it is fixed before processing any insert.

TV strengths0.1..512HU; NLM h0.1..128HU. Two endpoints plus eight geometric
bisection steps, no parameter-budget expansion. Require bracket and monotonic
sampled scores, then closest sampled strength within2% of fitting target.
Unmatched/nonmonotone cases are explicitly unavailable, never matched by label.
Report held-out high-frequency error separately; >10% invalidates any claim
of a well-matched control on that spatial holdout, not the raw response curve.

## Numerical validity and resource gates

- Crops enclose all measurement/calibration masks with24mm and40mm halos.
  CPU-filter cropping does not alter the existing whole-volume VAE result.
- NLM finite-support behavior has a toy full-vs-crop regression test. TV is
  iterative: NO exact small-halo claim. Compare baseline outputs on all support
  voxels between crop sizes and TV200vs400iterations on larger crop.
- Baseline crop/iteration maximum difference must be <=0.1HU before curves.
  Curves use larger crop; all six responses are compared with smaller-crop
  responses. Require corrected retention differences <=0.01. TV-10/+20 also
  require200vs400iteration retention differences <=0.01, with each iteration
  setting's own matched-host baseline. These are engineering tolerances,
  not scientific safety thresholds or a proof of global convergence.
- Hard1800-second watchdog exits124; one CPU thread, no retry, unique outputs,
  no images saved. Each calibration/pass JSON is saved separately. Completion
  may honestly report unmatched/inconclusive families; no auto scientificGO.
- Cached source arrays and masks retained locally outside the Git fork.
  Only tooling, protocols and scalar findings may be committed.

## Tests and measured preparation

Four offline tests passed: constant-volume preservation, analytic linear
retention for both signs, NLM crop-interior agreement, global-sign-inversion
equivariance of both classical filters. This last check is NOT a proof of
local positive/negative-insert symmetry in heterogeneous anatomy.

One-thread96x96x32 toy timing: TV0.375s, NLM0.391s. These are workstation
component measurements, not projected full-run guarantees. Full real-data
hash/geometry/calibration preflight must pass before launch.

Final real-data preflight passed: image/label/VAE hashes exactly match; original
insert111voxels at225/259/74 preserved. Fit6570voxels/holdout6386; split onx at
voxel195. Target high-frequency SD18.168314HU. Crops251x179x25 and297x225x33.
Runtime NumPy2.5.3/SciPy1.18.1/nibabel5.4.2/scikit-image0.25.2 recorded.
Final runner SHA256 acd6617b4b84887d598cee8966371bc510ab1577edb469ff296a4b2991070395.
Four tests passed again after caching repeated baseline-filter work and adding
an actual1MiB output-write/fsync guard. First import failure was task-directory
access under the sandbox, not a filter defect; authorized execution passed.

Launched detached/hidden locally after preflight, one-CPU affinity, frozen
runner+ROI helper under work/denoiser_cpu_snapshot_v1/. Local PID11052 atlaunch.
Outputs/logs under work/denoiser_control_20261002_v1*. No verdict atlaunch.

## Interpretation

Report classical-vs-VAE raw/corrected response, spatial holdout mismatch and
numerical sensitivity. Similarity to one family supports a generic-denoising
explanation for this construction, not all VAE behavior or clinical safety.
Differences from both are unexplained by THESE controls, not proof of a unique
learned prior or proof that pixel-space diffusion is necessary.

Claude's0.05/0.10 heuristic comparisons may help prioritize, but cannot override
failed matching or numerical gates. Inconclusive synthetic controls do not
block a separately justified real-lesion test. Do not tune strengths, add
families or choose a paper claim after seeing inserts. Spending estimate stays
0.448611charged-equivalentGPUhours of the2-hour cap; CPU work adds zeroGPUhours.
