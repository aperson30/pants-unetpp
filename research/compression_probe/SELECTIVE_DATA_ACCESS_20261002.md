# Selective PANORAMA access: CPU-only feasibility result

The public version-1 Zenodo batch archive honored bounded HTTP Range requests:

- URL: https://zenodo.org/records/10998332/files/batch_1.zip?download=1
- Archive length: 48,238,512,666 bytes.
- ZIP directory: 554 members, obtained with 49,683 bytes across 5 requests.
- Example `batch_1/100028_00001_0000.nii.gz`: 49,164,104 bytes,
  ZIP_STORED (compression method 0), local-header offset 0.
- No member was opened, downloaded, extracted or decoded. No GPU allocation.

Evidence stays in `/u/asanjeev/compression_probe_smalloutputs/` on DeltaAI:
`panorama_batch1_directory_v1.json` and its companion log. This establishes
directory access, NOT a successfully extracted CT or an eligible held-out case.
The earlier latest-version request returned 429; we stopped that request.
Version-1 access must not be silently mixed with another release's metadata.

## Reproducible metadata-only tool

`list_remote_zip.py` uses the standard ZIP parser over a bounded range reader.
It refuses HTTP 200 responses before reading their body, checks Content-Range,
limits each request to 4 MiB, all fetched bytes to 8 MiB and calls to 32.
Each request has a 15-second timeout; the remote run also had a 90-second
process timeout. Unbounded reads are prohibited except the parser's small EOF
directory tail. Three local and remote unit tests passed. The final source
additionally reserves the oversize-detection byte within its total-byte bound;
all three local tests passed again after that tightening.

## Cheapest next gate, before another GPU job

1. Read the public clinical workbook and manual-label IDs; join exact study IDs
   to this archive directory and the pinned split/checkpoint identity evidence.
   The labels repo currently has `clinical_information.xlsx`, not a root CSV.
2. Select a small prespecified set eligible for the intended held-out comparison;
   exclude patient overlap and unresolved dataset-version or selection exposure.
   Do not cherry-pick cases based on reconstruction or detector outcomes.
3. Implement a bounded single-member retrieval only after eligibility is settled:
   verify the local header, exact filename, compression/encryption flags, sizes,
   stable source identity, ZIP CRC and complete NIfTI geometry/finite data. Abort
   on ignored Range, quota/write failure, inconsistent metadata or rate limits.
   CRC checks transfer integrity, not authenticity; preserve release provenance
   and a SHA256 of the fetched file as well.
4. Check the intended judge's task, input convention and held-out provenance.
   Frozen-detector failure alone is not proof of irrecoverable information loss.
5. Only then price a tiny paired real-lesion test under the remaining authorized
   two-hour total. Preserve original-image misses and paired false positives.

At the initial directory-only snapshot, no clinical-workbook analysis or CT
extraction had occurred. The subsequent CPU-only checks below now resolve
those two access questions. No checkpoint download, reader-key publication or
GPU submission occurred. PanTS evaluation job 47320183 remains untouched.

## Completed metadata eligibility audit

`audit_panorama_metadata.py` read the public workbook without modifying it,
using bundled Python/pandas. Sources pinned to:

- panorama_labels `bf1d6ba3230f6b093e7ea959a4bf5e2eba2e3665`.
- PANORAMA_baseline `d08f2356fa70d9460881fec0aedba4ebd1566c7e`.
- Workbook SHA256 `2ae925de8067600db62e186a91844dec6555dab2b17cafd33b886b8c99f19ea7`.

Independent reproduction: 2,238 studies / 2,224 patients; 11 patients with
25 studies across validation folds. The two-stage fold memberships agree,
but JSON list order differs. The first check correctly stopped on strict
list comparison; corrected comparison uses sets while rejecting duplicate
IDs and cross-fold overlap. Both regression tests passed.

Excluding those patients and records whose `level` is MSD_dataset/NIH_dataset
leaves 380 manually masked PDAC studies (74/81/84/58/83 by folds 0-4).
Reference counts: histopathology 167, pathology 109, cytology 85, radiology 19.
Batch 1 contains 81 of these candidates (17/16/15/14/19 by fold).
These reproduce Claude's earlier full-cohort counts independently. They are
provisional candidates, not a finalized independent evaluation cohort:
checkpoint validation-selection exposure and release correspondence remain.

## Completed single-case transfer/integrity test

Study 100226_00001, fold 4, histopathology reference, was chosen solely as
the smallest non-radiology-only candidate for transport testing. This is NOT
a lesion-size selection or the scientific pilot cohort.

- Actual CT retrieved: 21,471,554 bytes, ZIP_STORED.
- Fresh archive CRC32 matched the full locally retrieved member: 1494263891.
- CT SHA256 `7af1c0d674c69dbf7a98ced8d8bf8059322f462bc55b98e0d30bd917579f9815`.
- Pinned manual-mask SHA256 `16a358e8c6abf6f18981b89db29651102da31a154599dd96654ec77fa4e4aa26`.
- Shape 512 x 512 x 58, spacing 0.68359375 x 0.68359375 x 5 mm;
  CT/mask affines identical, all voxel values finite, spatial units mm.
- Mask uses documented labels 0-6; tumor label 1 contains 1,756 voxels,
  approximately 4.103 mL. This is a data-integrity observation, not performance.
- Zenodo did not provide an ETag. CRC plus hashes and pinned metadata were
  verified; no claim of an ETag-protected immutable snapshot is made.

Initial imaging-runtime import failed because bundled Python lacks nibabel;
used the existing isolated NIfTI audit environment without installing anything
or touching training environments. Initial final-mask check stopped because
it expected binary labels, while these manual files also contain anatomy 2-6.
Corrected against the official label legend; reverified existing CT against
fresh archive CRC and mask against pinned source without redownloading the CT.
Completion marker is now true. No failed check was called successful.

Tooling: fetch_panorama_case.py, with 30 MB member ceiling, 40 MiB range-byte
budget, existing-input non-overwrite checks and geometry/finite/tumor gates.
Range-reader tests now cover changed ETag and expired deadline: five passed.
The subsequent verification fetched 49,683 archive metadata bytes; that number
is NOT the total original CT-transfer traffic. Full CT body was fetched once.
No images or per-patient clinical workbook copied into Git. Local inputs/results
are under work/research_inputs/panorama_metadata/ (outside the tracked fork).

Next gate: inspect judge preprocessing and both-stage single-fold execution;
prespecify paired arms, original misses, negatives and inference thresholds.
Do not use a CT-download-size-selected cohort to claim clinical effects.
No new GPU hours spent; screening charge-equivalent estimate remains 0.448611
of the authorized 2 hours, not a posted accounting debit.
