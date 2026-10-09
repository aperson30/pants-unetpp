# PI-directed U-Net++ generation pilot: executed gates and launch boundary

No full training job or cancer-preservation evaluation has been launched.
The original protected PanTS 2x2 evaluation remains untouched. This is architecture
research preparation, not an alternative clinical method or JEPA pivot.

## Data work executed

1. Reused the later complete CancerVerse annotation audit rather than repeating
   its work. Its evidence covers all83 strict metadata-candidate series; source
   file `phase_public_annotation_geometry_full_20261007.json` is separate prior
   research evidence, not a verified clinical annotation set.
2. The two initially promising geometry groups have empty pancreatic-lesion
   masks. They are not labeled tumor-preservation pairs.
3. Strict group015 has NC CV_00007899 and venous-hinted CV_00019311/CV_00019743.
   All three have identical CT shape512x512x378 and CT sform. Their masks contain
   39,444/687,140/767,050 foreground voxels respectively, with large centroid and
   bounding-box differences. Mask spatial units were unspecified in the earlier
   audit. No blind resampling/label transfer or interpretation as a verified
   pancreatic cancer reference is allowed.
4. Downloaded only those three pinned public CTs into isolated personal home:
   286,144,362 compressed bytes total, below300MB cap; exact public LFS SHA256s
   checked. All finite; full scaled-voxel hashes differ. No edits to source CTs.
   Small subsampled-body correlations0.950/0.922 NC-to-venous,0.987 venous-to-venous
   are descriptive, not registration or tumor fidelity metrics. Visual inspection
   of six axial levels shows broadly consistent anatomy and visibly different
   enhancement, but does not certify identity, acquisition phase or pathology.
   Montage kept locally outside Git; no CTs or images pushed to GitHub.
5. Checked an easy-to-miss discovery restriction: genuine phases might have
   separate accession records. A separately labeled same-patient/same-date
   candidate search expands23groups to28. It never joins across dates or marks
   a pair eligible; anonymous patient indices are specific to each inventory and
   must not be cross-joined by their index string. Public case IDs drive joins.
   Extended label-availability evidence and summary are recorded separately.

The extension is now complete:28candidate groups,27candidate patients,
109distinct NC/CE-hinted series. Masks inspected for all109; **only one candidate
patient has nonempty pancreatic masks in both phases**. There are no remaining
unknown masks within this pinned same-day candidate inventory. This is not a
claim that the entire CancerVerse dataset has only one pancreatic patient or
that broader, verified time-separated pairs cannot exist.

The initial local bounded archive stream hit its150s deadline; preserved its
partial evidence rather than retrying blindly. Cached the551,163,721-byte public
archive once in isolated cluster home, verified its full SHA256 against the
previous complete audit, and finished all29new target masks in a bounded CPU run.
Saved complete joined report `data_gate_report_complete_20261008.json`; no
compressed label archive, CTs, raw masks or montage added to Git.

### Direct mask overlay completed

Three original public masks retrieved from the verified cached archive, bounded
decompression and exact prior foreground counts checked. Stored mask affine
matrices match their CTs, but mask spatial units are unspecified. Index-space
overlays at four axial levels show substantially different labeled regions
between NC and venous scans. NC-to-venous index-space mask Dice is0.0664/0.0549;
venous-to-venous0.8666. These are **reference-mask comparisons, not model scores**.
They do not prove incorrect annotations, identify histology or measure registered
lesion agreement. They raise a material annotation-scope/correspondence question.
Do not equate label differences to model hallucination or assume transferring a
contrast mask to NC produces verified ground truth.

Initial overlay extraction ended nonzero before completion. No lingering process
or GPU allocation remained; one partial isolated mask was preserved. Corrected
the sparse-archive compressed-stream buffering path to explicit gzip decoding,
verified/reused the identical partial mask and completed a second bounded CPU run.
No automatic retry or source overwrite. Raw result:
`tumor_candidate_mask_overlay_20261008.json`; montage local outside Git.

## What a fair pilot would compare (proposal, not frozen recipe)

- Question: better **time-to-quality**, not merely fewer parameters or cheaper
  updates. Tiny nested fixture was26.4% slower per update; measure production
  shape, data/auxiliary costs, convergence and actual generation latency.
- First isolate deepest-only plain versus nested on identical verified pairs,
  CLIP/timestep/source conditioning, frozen VAE, normalization, diffusion schedule,
  stochastic VAE/noise/timestep draws, optimizer and update budget.
- Both original small architectures would start from scratch for a feasibility
  comparison. Random nested versus pretrained SMILE alone is not architecture
  evidence. Equal widths are not equal capacity: report counts and include a
  capacity-controlled follow-up before an architecture-only attribution.
- The existing independently written nested prototype is not the draft's full
  adaLN-zero implementation or a drop-in pretrained SD1.5 backbone. Production
  scheduler/model save/load/EMA/AMP/dataloader integration remains to implement
  and validate after recipe/data decisions; fixtures are not that integration.
- Deep supervision, pruned exits and timestep routing come only after the
  deepest-only comparison works; make each a separate ablation. Loss sums versus
  normalized averages change optimization scale and must remain explicit.
- Frozen VAE posterior parameters can be cached when transforms are fixed,
  retaining fresh shared posterior draws; never cache one draw forever silently.
  Cache phase prompt embeddings; profile triplet batching and auxiliary costs.

## Actual reference recipe is not a guessed default

SMILE source pinned23f5a28fe25ed0472024b688e19a79ce119c4833 `train.sh` requests
one process, no mixed precision,512resolution,batch1,gradient accumulation2,
LR1e-5,constant schedule,clip norm1,L1 denoising,200000updates,resume/init15000,
and staged auxiliary losses. It references private auxiliary paths and a prior
checkpoint; the supplied paper schedule is different. The separate GPU contract
probe used BF16/MSE/lr1e-4 on random targets and is NOT a reproduction recipe.
Do not deploy its settings as medically validated or quietly omit tumor/organ,
HU, phase and cycle supervision from an experiment called SMILE reproduction.
PI recipe confirmation is distinct from permitting bounded implementation tests.

## Safety preparation executed

`pilot_safety.py` rejects unverified pair flags, patient/scan split leakage,
duplicate pairs, protected-overlap unknowns, empty tumor splits, missing frozen
recipe/safety margins and absent finite positive charged-compute authorization.
Passing minimum manifests does not establish adequate sample size or clinical
quality. It is a tumor-pilot gate, not a claim that all training requires perfect
registration; the released relative-z recipe is distinguished from aligned
pixel-level paired fidelity evaluation.

New CPU test serializes and reloads model, AdamW momentum, scheduler, Python/NumPy/
Torch RNG and sampler position, then verifies the next stochastic update exactly.
It rejects changed recipe/data hashes before loading weights. This helper is not
wired into production, and GPU AMP/scaler/EMA, multiprocessing-worker and
preemption/atomic-checkpoint durability are not yet end-to-end verified.

All38 CPU tests passed6.386s under a90s subprocess timeout, including the expanded
metadata discovery, unknown-mask handling, revision/overwrite rejection, mask
counter and stochastic resume contract. Full CT staging and pixel audit ran in
the isolated cluster environment; these CPU tests do not certify that production
model training/medical scoring or GPU resume is ready.

## Launch / medical evaluation gates still open

Do not fabricate a frozen training split: real patient independence, phase
linkage, adequate tumor labels and exclusion of protected PanTS overlap remain
unverified. A single tumor-positive candidate patient cannot serve as independent
tumor-positive training and development data. More candidate masks do not fix
that unless they pass identity/correspondence review.

Before a real training launch, obtain a finite separate charged-GPU-hour cap,
including failures/staging/checkpoints; the0.25h smoke cap is not a training cap.
Freeze a feasible recipe and success margins on development data before outcomes.
Medical endpoints: fixed-detector lesion sensitivity at a fixed false-positive
operating point; missed/created lesions; tumor Dice and size strata; negative
scans; aligned-reference fidelity where available. Intensity correlation, image
appearance, organ Dice and denoising loss cannot clear the tumor-preservation gate.
No method selection on the protected test set. A generative architectural pilot
on non-tumor pairs could test mechanics, but would not answer the cancer question.

This is not proof that the architecture project needs paired tumors for every
endpoint. A separately defined study could train enhancement on verified general
phase pairs and evaluate tumor preservation/detection on independent labeled
non-contrast PanTS development scans: that downstream endpoint does not require a
paired contrast target. It cannot establish paired tumor-fidelity or show that
tumors were adequately represented during enhancement training. It still needs
verified source labels, adequate positives/negatives, protected-set exclusion and
CancerVerse/PanTS patient-overlap checks. Do not silently substitute that design
for the requested paired tumor-aware comparison or call the current gate universal.
