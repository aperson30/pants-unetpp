# Sub-mL MSD compression follow-up

One prespecified case120: 454 tumor voxels / 848.6181mm3, one 6-connected
component, native 512x512x104. Selected as smallest positive total GT burden
among first 120 lexicographic masks before inspecting reconstruction outcomes.
Public pinned mirror/model; training-set independence not established.

Unchanged whole-volume MAISI VAE, posterior mean, FP32, native spacing,
orientation-only reordering/padding. No crops/decoder tiles/training/diffusion.

Preprocessing control: tumor-region HU MAE zero, boundary-band HU MAE zero,
signed contrast unchanged. Thus observed tumor-region differences below are
from reconstruction versus control, not the intensity clipping step.

- Tumor HU MAE: 18.27HU.
- Boundary-band HU MAE: 20.08HU (NOT boundary-position accuracy).
- Signed local contrast: -65.55 -> -58.58HU; magnitude retention 89.37%.
- Signed CNR: -1.1691 -> -0.9785; magnitude retention about 83.70%.
- Whole-image HU MAE: 21.87HU; not the primary success measure.
- Forward/copy time: 107.89s; GPU peak allocated ~39.09GiB.
- Python RSS peak 37632640KiB (~35.89GiB).

These measurements show attenuation of local contrast in this one small
lesion, not disappearance, detector recall loss or clinical realism. A single
maximum-area slice/GT contour does not establish preserved 3D lesion shape.
Do not compare its error directly with the larger case as a size-effect estimate:
different anatomy/intensity/case characteristics are confounds.

This is an exploratory compression bottleneck screen. Independent downstream
detection and more cases would be needed before a medical method/paper claim.
PanTS training/evaluation remains unchanged.

## Completion and compute cost

Job3290494 COMPLETED, ExitCode0:0, completion.json confirms one case.
Allocation elapsed162s => 0.045 physical GPU-hours; interactive 2x charge rule
=> approximately0.090 allocation-equivalent hours, below the NEW0.25 cap.
This is estimated from actual accounting elapsed/rule, not a posted debit.
Internal complete runtime128.75s. No retry. Combined with previous pilot:
0.13028 physical GPU-hours / approximately0.26056 allocation-equivalent hours.
The earlier0.25 and new0.25 caps are separate; no single-cap overrun claim.

Downloaded JSONs, source manifest and PNG only, not the raw CT/mask/volume
reconstruction. Fixed-window PNG inspected: reconstruction visibly smoother;
no lesion-erasure or preserved-detection conclusion drawn from that image.

## Post-hoc CPU sensitivity audit

Three physical rings (1-3, 2-5, 3-7mm) yield contrast ratios .8725, .8937,
.9210 against all surrounding tissue. Restricting these same rings to GT
pancreas gives .8635, .8524, .8611: attenuation persists across these choices.
This restriction is defined by the original GT, not predicted healthy tissue.

Mean tumor intensity shifts -1.96HU; the standard2-5mm surrounding ring shifts
-8.93HU. Thus the reduced contrast also reflects changes to surrounding
tissue; it cannot be attributed solely to fading of tumor voxels.

Across the three tumor-containing axial slices, ratios are .8915, .8583,
1.0148. Slice averages do not all attenuate. These slices are correlated
measurements of one lesion, NOT three independent cases or significance tests.

Diagnostic uses local measurement halos AFTER whole-volume reconstruction,
never model-input cropping. Tests check identity/global intensity shift,
synthetic erasure, empty pancreas reference and invalid inputs. Nine total
metric tests pass; this validates implementation contracts, not clinical utility.
