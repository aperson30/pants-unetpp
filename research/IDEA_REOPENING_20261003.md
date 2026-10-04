# New idea search: synthetic utility versus validity

2026-10-03. Literature, pinned author-source/resource inspection and a small
CPU algebra contract. No data/weight payload download, GPU allocation or main
PanTS evaluation action in this pass. Follow-up audit:
[SYNTHETIC_UTILITY_SOURCE_AUDIT_20261003.md](SYNTHETIC_UTILITY_SOURCE_AUDIT_20261003.md).
The prior NO-GO in IDEA_SEARCH_DECISION_20261002.md remains valid; this is not
a replacement cohort, a revival of failed compression probes, or an S-tier claim.

## Question examined

Could segmentation-based synthetic-data filtering discard genuinely useful hard
lesions, while high-loss selection retains generator defects? Can an inexpensive
check distinguish these without retraining the generator or requiring new
clinical annotations for an initial model-behavior screen?

This is a question, NOT an observed failure in our data. Neither mask conditioning
nor a detector's score certifies that a generated lesion is clinically valid.
Public masks support technical consistency tests, not clinical adjudication.

## Novelty collisions found before spending compute

1. [Native Adversariality Mining author repository](https://github.com/JackCD99/Native-Adversariality-Mining):
   already mines hard seeds using a frozen generator/anchor and a trained miner.
   Its README explicitly warns about defective generator modes and documents
   threshold replacement, VQA filtering, conditional-prediction reranking and
   attention-based mitigation. Generic hard-example mining or filtering is not
   a new contribution. Initially only README inspected; subsequent pinned
   source reads of HAT, LSRS and QSF succeeded through ordinary public HTTPS.
   See the follow-up audit for actual limits; no reproduction claim.
   The extended-repository claims must not all be attributed to its CVPR paper.
2. [When Sample Selection Bias Precipitates Model Collapse](https://arxiv.org/html/2606.13732v2):
   already studies biased selection eliminating distributional tails under
   recursive synthetic training and fragmented verifier references. This does
   not establish the same mechanism in one-round medical augmentation, but
   defeats a broad novelty claim that filtering can remove rare modes.
3. [PRISM](https://arxiv.org/html/2609.05028v1), September 2026 preprint:
   already separates intensity/texture, structural and semantic verification
   and uses ordered rewards for generator fine-tuning. It reports that reward
   design requires domain expertise. Merely composing multiple quality scores
   is therefore not new; avoiding generator training alone is insufficient.
4. [Rethinking hard training sample generation](https://github.com/Bbinzz/Rethinking-Hard-Training-Sample-Generation-for-Medical-Image-Segmentation):
   author abstract targets hard examples transferable across downstream models
   with feature/prototype discrepancy and fidelity constraints. A second detector
   or cross-model agreement is not automatically a distinct contribution.
5. [PCaPaint](https://arxiv.org/html/2609.37350v1), September 2026 workshop paper:
   already examines condition-copying shortcuts in prostate MRI inpainting,
   proposing noise-filled conditions and lesion-weighted training. Its assumed
   shortcut incurs higher expected loss with noise fill; this is not a theorem
   guaranteeing a trained network avoids shortcuts. Generic lesion-focused loss
   or noise-fill fixes should not be presented as ours.

Verdict: PARK the broad filtering/mining idea. The narrower hard-valid versus
hard-defective distinction also overlaps NAM's mitigation and PRISM. No verified
gap yet justifies coding a new selector or spending remaining screening hours.

## A lower-cost resource lead, not a selected method

[LeFusion current author repository](https://github.com/HINTLab/LeFusion)
advertises pretrained LIDC/EMIDEC generators, 20 healthy lung-nodule-region
inputs with masks, and pre-generated examples at three histogram controls.
[Data listing](https://huggingface.co/datasets/YuheLiuu/LeFusion_Preprocessed_Data/tree/main)
and [model listing](https://huggingface.co/YuheLiuu/LeFusion_Pretrained_model/tree/main)
were accessible. Subsequent API reads pinned advertised sizes/hashes and licence
metadata; payload integrity, upstream data rights and pairing remain unchecked.
The two small LIDC archives total 47,638,016 bytes. Region examples cannot support claims
about full-volume context or pancreatic tumor detection.

Authors report ~40 seconds/image on an A100 40GB at jump_length=2 and
jump_n_sample=2. Twenty fresh examples at that configuration imply about
0.222 physical GPU-hours of generation alone; this is author-reported timing,
NOT our measured runtime or charged allocation cost. Reusing published examples
would avoid that generation, if their provenance/content supports the question.
Reported generator training is five A100s for 2.5 days (~300 GPU-hours), making
retraining an unsuitable first step for this resource-minimal search.

## Next cheap decision, BEFORE experiments

1. Audit actual author implementation of NAM mitigations and PRISM verifiers,
   with pinned revisions. Identify a failure they do not already address;
   otherwise discard this branch, rather than inventing an adjective for it.
2. If a substantive distinction remains, inspect LeFusion data metadata and
   licence, then a bounded pre-generated subset outside Git. Freeze case and
   control selection before outcomes. Do not install or execute author code
   just to inspect resources; do not download the full pathological archive.
3. Only if those assets suffice, write a discriminating test specification:
   fixed mask/texture controls, valid-difficulty versus broken-condition controls,
   basic intensity/contrast and mask baselines, published mitigation baselines,
   and an independent task model whose native adequacy is demonstrated.
   A rule beating only one weak detector or an uncontrolled corruption is NO-GO.
4. A cheap diagnostic is not a cheap completed paper. Any augmentation-utility
   claim needs matched downstream training, equal data/training budgets and
   repeated held-out evaluation. No such cost has been measured here. Reject
   routes requiring new large generator training or expert-labelled cohorts
   for the initial decision; do not promise inference-only utility validation.

For any efficiency claim, count generation, scoring, rejected attempts, selector
training and downstream continuation, not just accepted images. The old total
2 charged GPU-hour cap persists; last reconstructed usage was 1.225833 hours,
not refreshed posted account debit. No new submission until accounting and a
specific affordable hypothesis pass. Main 901-case evaluation stays protected.

## Honest progress

Follow-up: [batch-conditioned utility CPU screen](BATCH_CONDITIONED_UTILITY_20261004.md)
found isolated/full-batch gradient disagreement, but the 29-channel aggregate
endpoint reversed only 1/256 decisions and the exploratory rare-class target
reversed none. The sufficient-statistic correction contract passes; novelty and
meaningful rare-class benefit do not. Park the standalone paper proposal rather
than spending GPU hours on its stronger-looking binary toy result.

This pass ruled out several tempting combinations and located reusable author
assets. It did not find an S-tier idea, demonstrate better quality, or establish
that no good idea exists. The best immediate investment remains a bounded source
and asset audit, not another favorable-outcome search or multi-job launch.
