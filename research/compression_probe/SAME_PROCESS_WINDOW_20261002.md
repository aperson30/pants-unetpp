# Separate same-process window control, after repeatability measurement

3296939 COMPLETED0:0,46allocationseconds,billing2000 =>0.025556chargedh-
equivalent; campaignestimate0.901944/2, posted debit unknown. Both default
repeats matched exactly; both deterministic repeats matched exactly.
Historical maximum drift0.001598(default)/0.001856(deterministic) fails original
1e-4gate. Tumor mean default0.030070854, deterministic0.030073991 (~3.14e-6
difference). Rootcause of historical drift NOT proven; no retrospective pass.

New study: original/reconstructed100430 x both saved publisherwindows, each
cell predicted TWICE in SAME process under measured deterministicpolicy.
Eight full stage-2 predictions; no VAE, training/newpatient, alteredmirroring,
GT-selectedcrop or tuned clinicalthreshold. Originalfailedcontrol preserved.
The scientific question is posthoc: can the changedwindow alone explain the
score decrease in this patient? Use freshnative references in sameprocess,
not a relaxed historicalmapcomparison. GTlabel1 scoring-only, not PanTS28.

Fresh repeated-control gate: maximum probability drift over ALL7classes<=1e-4
AND identical segmentationlabels in everycell, unchangedbackendflags.
Failedcell stops completion, preservespartialresults; no silentrelaxation.
Engineeringrepeatability does not establish clinicalsafety. Record allfour
cells' rawtumor mean/max, candidateoverlap/confidence, GTinclusion/mask,
windowbounds, repeatdrift and timing. Keep croppedraw/masked/candidatemaps
private for independentaudit. Report everycell, including originalmisses.

Compare reconstruction/native ratios WITHIN eachwindow and windowresponse
WITHIN eachimage. Persistence in bothwindows rules out windowchange ALONE
for thiscase, not allpreprocessing/domainshift and not irreversibleerasure.
Mixed results retained. Neither response proves clinicalrecall/newmiss,
representativecohort, diffusionlatentfidelity or a novelpublication method.

CPU regression mocks neuralcompute but exercises8calls/4cells and failclosed
on repeatdrift; exactphysicalgeometry and scoringhelpers tested separately.
Does not substitute for actual scheduledpredictions.

OneGPU/2CPU/32G,3minhardlimit/150sprocess/no-requeue, max0.1chargedh-equivalent
at billing2000. Campaignworst1.001944/2. Submitheld/verifyfields/hashes then
releaseonce. No automaticretry, largecase100259 staysblocked,47320183untouched.
