# Second size-ranked MSD case

Case165:445 tumor voxels/1060.9616mm3 (1.061mL), one 6-connected component,
native512x512x85. Second-smallest positive total GT burden among same120 masks,
selected before its own reconstruction outcomes. NOT below the1mL cutoff.
Same pinned mirror/model and unchanged whole-volume FP32 posterior-mean runner.
No diffusion/training/detection and no model-input cropping/decoder tiling.

First CPU download was truncated (10,292,481 vs expected25,525,318 bytes) and
failed hash validation. Preserved incomplete file; separate CPU download
attempt passed pinned hash/grid checks and model preflight. No GPU was spent
on invalid data. No automatic GPU retry.

## Observed metrics

Preprocessing control has zero tumor and boundary-band HU MAE and unchanged
local contrast. Reconstruction versus control:

- Tumor HU MAE42.12; boundary-band HU MAE45.04 (not boundary-position error).
- Mean signed local contrast11.79->11.20HU: ratio .94977 (~5.02% reduction).
- CNR .22660->.26608: ~17.42% increase, NOT proof of better detectability.
- Whole-image MAE69.25HU is secondary, not a tumor quality measure.
- Forward/copy96.11s, GPU peak allocated35,529,854,464 bytes (~33.09GiB).
- Python RSS peak32,063,360KiB (~30.58GiB).

Local CNR improves despite reconstruction error: surrounding-region standard
deviation can shrink under smoothing. Neither MAE nor CNR substitutes for
independent detection/clinical evaluation. Contrasts differ substantially
between cases, so this does not establish a causal size effect.

## Completed cost and budget

Job3290519 COMPLETED, ExitCode0:0, one completed case. Allocation128s =>
.03556 physical GPU-hours / ~.07111 allocation-equivalent hours (interactive2x).
Case120+case165 NEW campaign total .09000+.07111=.16111 charged-equivalent
hours, below approved .25 cap. Remaining .08889. Both original and new campaigns
combined: .16583 physical / ~.33167 allocation-equivalent hours. These are
elapsed/rule estimates, not posted account debits. No further job submitted.

## Scientific decision

Current paired screens show case-dependent reconstruction changes, not a
demonstrated rare-lesion erasure mechanism or a top-conference method. Keep
the cheap signal, but do not begin expensive lesion-preserving VAE training
based on it. Next decisive evidence needs an independent, appropriate lesion
assessment, explicit model/data overlap checks, and more prespecified cases.
PanTS training/test evaluation remains untouched.

## Post-hoc ring/slice sensitivity

All-tissue contrast ratios .6307/.9498/1.0898 for1-3/2-5/3-7mm rings;
GT-pancreas-only ratios .6307/.9443/1.1606. The sign of apparent improvement
depends on the surrounding-region definition. Tumor mean shifts-32.52HU;
standard ring-31.93HU, so almost matched intensity shifts conceal substantial
absolute reconstruction differences in the contrast summary.

Three axial slice contrast pairs (control->reconstruction):

- 25.86->16.27HU (37.1% reduction).
- 22.97->7.24HU (68.5% reduction).
- 0.85->13.41HU. A ratio here is numerically large because the reference
  contrast is near zero; report absolute HU, not a meaningful percent gain.

This is variation within ONE lesion, not three independent samples. Region
averages can hide spatially heterogeneous changes, but slice contrast is NOT
a boundary/detection metric either. No erasure or clinical harm claim.
Fixed-window PNG inspected: visibly smoother/different intensities, no clinical
verdict. Native-spacing diagnostic protocol may differ from MAISI's production
preprocessing; matched official-pipeline parity is not established.
