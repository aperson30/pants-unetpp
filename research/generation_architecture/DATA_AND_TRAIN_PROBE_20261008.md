# Available data and architecture training probe

## Decision: CTVerse is optional, not mandatory

The user has CancerVerse and PanTS. Neither should be ruled out simply because
SMILE used CTVerse. Keep the PI-directed question: can a U-Net++ conditional
denoiser improve time-to-quality for CT enhancement versus a fair U-Net control?
Do not replace that question with unpaired translation or pseudo-target
distillation without an explicit experimental decision.

CancerVerse is the strongest available lead for real phase pairs. PanTS provides
tumor/organ labels for development and downstream checks, but its audited public
metadata do not establish patient-level phase correspondence. A non-contrast scan
and an unrelated contrast scan are not a supervised training pair. Do not use the
protected 901-case evaluation to select this generation method.

## Executed CancerVerse metadata check

Pinned release: `c15802c28cb31b71dee2e680319d522b3bc4cbbf`.
Official source: https://huggingface.co/datasets/BodyMaps/CancerVerse

- 24,422 rows, SHA256
  `3e58b8a1b8cb3b9664d446ef4d7ca393c51495af5073119c25596b2d4bb903d8`.
- 23 candidate same-patient/accession/date groups; 20 candidate patients;
  83 series. These are metadata candidates, not verified training pairs.
- Zero groups with a C25 code in this metadata screen. This does NOT prove
  absence of pancreatic lesions; tumor annotation verification remains unknown.
- Candidate inventory retains public CV filenames and anonymous group/patient
  indices only. No raw patient IDs, dates, accessions or reports were saved.
- All eligibility flags remain false until linkage, phase, registration as
  required by the chosen use, and annotations are checked. Repeated candidate
  patient indices must not be split across train/development sets.
- Previous bounded mask-prefix audit covered only four of these series and found
  empty pancreatic masks. The other 79 were not covered: do not extrapolate.
- No CT/mask downloads or GPU allocation for this metadata check. Web directory
  lookup for the first two candidates was inaccessible; pixel-level inspection
  has not happened and no geometry or registration pass is claimed.

Released SMILE `ReconCTDataset` selects phases within a patient/resolution group
and uses the same relative slice depth. It does not implement exact cross-phase
registration there. Therefore voxel-perfect registration is not automatically
a prerequisite for reproducing its training recipe; accurate paired pixel-level
fidelity evaluation needs suitable alignment, with residual motion disclosed.

Next data work, bounded and cheapest first:

1. Resolve exact public file paths for a few candidate groups; inspect geometry,
   phase evidence and duplicate/linkage consistency without downloading the whole
   CancerVerse release.
2. Inspect their tumor annotations and determine whether this is enhancement
   feasibility data only or supports tumor-preservation testing.
3. Freeze patient-independent training/development manifests and recipe. General
   phase pairs can support an architecture feasibility pilot, but cannot alone
   establish cancer preservation. PanTS masks cannot be transferred blindly.
4. Only then propose a separately budgeted matched training pilot. No full
   training authorization is implied by the smoke-test budget.

### Executed header follow-up

Exact pinned public paths resolved through the Hugging Face tree API. Only
64 KiB compressed prefix per CT was read (six files, 384 KiB total); exactly
348 decompressed NIfTI header bytes parsed. No voxel array/full CT retained.

- group001: CV_00010488 (NC hint), CV_00004804 (venous hint). Shapes
  512x512x226 versus 512x512x414. Stored sform z extents at voxel centers are
  approximately [-803.5,-466] and [-1690.1,-1070.6] mm: no overlap. Exclude from
  naive coordinate pairing pending image/source investigation. This does not
  distinguish anonymization-coordinate changes from a wrong linkage.
- group002: CV_00018276 (NC hint), CV_00008477 (venous hint). Both 512x512x373,
  0.73464x0.73464x1.5 mm, identical stored sforms. Public LFS content identifiers
  differ (c668a4798714530c1f0bec09b1c0c1961252dee744df4d663ce2fc775d31abba
  versus 595980567a1ce42d41cc88d8f2d7ac20aca06e0a214ecd96ee0876cef3098eb4),
  file sizes118253758/115815466 bytes. These are different compressed files,
  not proof of distinct voxel data or true phases. Strongest first pixel-audit
  candidate among the three groups screened, NOT training-approved.
- group003: CV_00011916 (NC hint), CV_00018699 (arterial hint). Both512x512x488,
  1.171875x1.171875x1 mm; stored origins differ about0.4mm in y and2mm in z.
  Second promising geometry candidate, NOT confirmed registration.

Header SHA256s and observations are saved in candidate_headers_20261008.json
and candidate_headers_followup_20261008.json. A header-only pass cannot establish
patient identity, anatomical motion, image equality, tumor visibility, labels,
CRC/full-file integrity or lack of protected PanTS overlap. No dataset-wide pass
or training eligibility is inferred from two promising candidates.
All28 CPU contract tests passed2.379s before the CLI follow-up; final rerun recorded
separately. Added explicit public-ID validation and an eight-header per-run cap.

## Completed GH200 training contract / cost probe

Job `3345424`: COMPLETED, exit 0, 18 allocated seconds. Raw output:
`architecture_train_3345424.log`. Torch 2.10.0+cu129, cuDNN 9.10.2, GH200 120GB.

| Tiny random fixture | Parameters | Median update time |
| --- | ---: | ---: |
| Plain U-Net | 1,096,516 | 17.67 ms |
| Nested U-Net++, deepest head | 1,202,536 | 22.34 ms |
| Nested U-Net++, all heads | 1,202,536 | 22.72 ms |

Three warm-up and eight timed forward/backward/AdamW updates per mode; BF16
autocast, FP32 weights, batch 1, identical reseeded random inputs and targets,
8-channel 64x64 latent input and 77x768 context. Shared conditioning blocks and
widths, NOT equal parameter counts or SMILE capacity. All-head supervision uses
normalized equal weights so this fixture does not conflate total loss scale.
Only the deliberately skipped intermediate output-head parameters have absent
gradients in deepest-only mode; expected gradients and updates were finite.

Nested deepest was 26.4% slower per update on this fixture. This does not establish
the cost ratio at production size, quality, convergence, or inference speed.
U-Net++ is not currently demonstrated faster. It would need better time-to-quality
or competent cheaper exits to pay for additional compute; the PI hypothesis
remains a hypothesis. Do not compare these random losses as medical performance.

All 25 CPU contract/inventory tests passed in 4.330 seconds under a 90s timeout.
Known four smoke allocations: 13+35+34+18 = 100 seconds. With observed factor 2,
estimated cumulative charged usage is 0.055556 GPU-hours before provider rounding,
below the authorized 0.25 smoke cap. No new training or evaluation was launched.
