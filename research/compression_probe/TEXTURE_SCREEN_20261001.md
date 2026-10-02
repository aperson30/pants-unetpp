# CPU-only texture screen: completed

Purpose: cheaply challenge the idea that medical VAE reconstruction destroys
pancreatic texture unusually strongly. This is not a cancer detector, REDMOD
replication, prediagnostic study or clinical safety test. No new GPU inference.

## Setup and verification

PyRadiomics 3.0.1 failed to install under Python 3.12 because its versioneer
uses the removed `configparser.SafeConfigParser`. Nothing was changed in the
training environment. A separate Python 3.11.9 environment successfully built
the official source package, using NumPy 1.26.4, SciPy 1.14.1, SimpleITK 2.5.6,
PyWavelets 1.8.0, nibabel 5.3.2, setuptools 69.5.1 and wheel 0.45.1.

`texture_screen.py` passed identity-repeatability, analytical mean/variance,
whole-bin intensity-shift invariance and xyz/RAS-to-zyx/LPS geometry contracts
before each case. These are implementation checks, not full IBSI certification.
The three local source-ROI tests also pass.

One CPU thread, detached process, 180-second overall timeout; all three outputs
and `ALL_TEXTURE_SCREENS_COMPLETE` verified. No training runtime, PanTS test job,
reader key or CT-image publication changed. Additional GPU charge: **zero**.

## Fixed analysis definition

- Cases 005, 120 and 165: existing whole-volume FP32 posterior-mean MAISI outputs.
- Source-label pancreas parenchyma only, excluding tumor by 5 mm and organ edge
  by 2 mm using voxel-center distances. Same ROI across every image arm. It is
  not known healthy tissue; all three subjects have diagnostic tumors.
- Native grid, no resampling, fixed 25 HU bins, no intensity normalization.
  Eight features: mean, variance; GLCM contrast/correlation/joint entropy/Idm;
  GLRLM short/long run emphasis. Settings selected before this extraction but
  after the previous reconstruction/insertion results: exploratory, not a
  preregistered clinical endpoint.
- Secondary mean-bias removal subtracts the paired ROI's mean HU difference.
  It is outcome-dependent correction, not a proposed clinical fix.
- Gaussian sigma 1 and 2 mm applied to original controls as illustrative blur
  references. Not noise-, PSF- or distortion-matched. Measurement crops retain
  at least a 4-sigma halo; no model input was cropped or reinferred.
- Neighbor distances are in voxels on anisotropic native grids: compare arms
  within a case, not feature magnitudes across differently spaced cases.

## Results

All entries below except ROI count and HU bias are fractional changes relative
to that case's original control, rounded. They are not accuracy percentages.

| Case | ROI voxels | Mean HU bias, VAE | Variance change, VAE | GLCM contrast change, VAE | Contrast change, 1 mm blur | Joint entropy change, VAE | Entropy change, 1 mm blur |
|---|---:|---:|---:|---:|---:|---:|---:|
|005|54,102|-39.90|+8.2%|-31.5%|-58.7%|-2.9%|-19.8%|
|120|16,138|-8.96|-22.6%|-24.4%|-42.4%|-8.5%|-19.3%|
|165|25,526|-27.20|-21.4%|-26.0%|-65.2%|-7.0%|-24.7%|

Removing mean bias leaves texture changes similar: contrast changes become
-31.8%, -25.2%, -25.9%, respectively. Thus these measured texture changes are
not solely a uniform HU offset or its effect on bin boundaries. In every case,
the selected GLCM/GLRLM changes are smaller in magnitude than either chosen blur
reference; Gaussian smoothing also increases correlation/Idm/long-run emphasis
and decreases short-run emphasis. Variance is different: case005 increases
under the VAE while decreasing under blur. No claim that a matched Gaussian
explains every reconstruction effect is justified.

Numerical JSONs with input/script SHA-256 hashes and all eight features remain
private remote artifacts:
`/projects/bdyo/asanjeev/compression_probe_20261001/texture_features_v1_{005,120,165}.json`.
Completion log: `texture_features_v1.log`. Isolated working environment:
`texture_cpu_py311_v1`; failed Python3.12 attempt/log retained separately.

## Idea-screening decision

**Deprioritize "generic texture drift itself is the discovery."** These
descriptives do not support unusually severe texture destruction relative to
the selected blur controls. They also do not establish that ordinary smoothing
fully explains the VAE, preserves diagnostic information, or that a stronger
result would hold on a cohort. Three positive cases, chosen partly for tumor
size, cannot support statistical population claims or early-cancer AUC.

The useful remaining question is **task-specific signal preservation**, not
another broad feature sweep. The matched insertion's attenuation and sizable
parenchymal HU biases motivate this question but are not detection evidence.
Next GPU spend should require a frozen, trustworthy downstream judge plus a
paired original/reconstruction/control comparison that can change the research
decision. No new training or larger data download is justified by this screen.

## Detector access audit: still a gate

Re-read the official PANORAMA implementation: both stages default to a five-fold
ensemble; high-resolution detection uses `checkpoint_best_panorama.pth`.
Its README explicitly selects checkpoints using fold-validation AUROC/AP.
Using the ensemble on its training cohort is not independent evaluation;
out-of-fold inference alone also does not remove checkpoint-selection exposure.
Neither inference nor weights download was started. Actual checkpoint fold
identity, patient exclusions, suitable case access and a defensible exploration
versus confirmation split remain unresolved. The Zenodo API metadata request
failed in the web tool; it is not evidence that the dataset is unavailable.

Sources:

- https://pypi.org/pypi/pyradiomics/3.0.1/json
- https://raw.githubusercontent.com/DIAGNijmegen/PANORAMA_baseline/main/src/process.py
- https://raw.githubusercontent.com/DIAGNijmegen/PANORAMA_baseline/main/README.md

## Minimal next experiment, only after provenance is cleared

Predefine a very small feasibility panel on eligible patients, a fixed checkpoint
and preprocessing, and an original-image inclusion rule that does **not** discard
original misses. Measure paired lesion evidence/recall and false positives at a
fixed threshold, with reconstruction mean-bias and blur controls. Preserve the
whole original-to-judge pipeline: a localizer failure can be part of the effect.
No threshold/feature retuning per condition. Positive-only feasibility cannot
estimate specificity or AUC. Calibrate total charged cost before allocating;
small-panel results choose what to investigate, not establish safety or a paper.
