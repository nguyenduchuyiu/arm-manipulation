# E175 — Current best checkpoint preview

Status: complete.

Userrequested visualize currentbest during ongoing50ktraining. Read one atomicbest.pt snapshot and preserve its exactbytes/hash/step alongside generated predictions; do not change training/weights/sourcecode. Model copied byte-for-byte with hash matched against checkpointprovenance. Only originalSAM3features/XYZ/pixelgraph/task as inputs; encoder+decoder frozen for this inference only, FP32/TF32off. No GTcentroid alignment,rescaling,rotation correction or additional training. E175 source training:772018trainableparams,canonicalE157,noSE3,EMDonly,B8,AdamW3e-4→3e-5 across50k; earlybest currentlyexpected12k but actualstep read from snapshot, not hardcoded.

Test4trainscenes×4tasks for preview,all128IIDscenes×4tasks=512queries for independent exactHungarianEMD/centroid/alignedCD/visiblevoxel4mmcoverage5mm. Visualpanels fixedscenes0/1/2/3 withtasks0/1/2/3,train andIIDrows; GT128green,predictionorange,rawXYZ. One staticIIDPNG and one~5MBinteractiveHTML. Additionally4IIDscenes/fullSO3+translation for pointwiseequivariance spotcheck, not trainingaugmentation. No noise,flowsteps,framegeneration or stochastic inference. Checkpoint remains stored so viewer can be reproduced after sourcebest is overwritten.

Own diag/e175-bestworktree,selective model/geometry helpers andsourcehashes in sources.json,no imports from activeexperimentcode. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` from testroot; creates freshresults. Attachedtmux ssh_tmux:e175-best,stdout+stderr mirrored to run.log. Viewer existingforwarded localhost:17070/e175_best.html (separate from final50kviewer). Canonical notes docs/research/e175_best.md. Training E175 continues independently; E172/E173 remainpaused.

Snapshotstep12000, IIDmetrics{'EMD_mm': 4.418513549868462, 'centroid_error_mm': 2.0467287258743365, 'aligned_CD_mm': 2.778516101988928, 'coverage_5mm_percent': 91.753907364587}. Codecommit `ce239669a9fa090afc78bc6ef26fbeababcc1118`. See results/summary.json.

Snapshot selected validationEMD4.373554mm atstep12000. Independent512IIDqueries:EMD4.418514mm,centroid2.046729mm,alignedCD2.778516mm,coverage5mm91.7539%. Four-sceneSE3spotcheckRMS0.0011365mm,max0.0041035mm. Viewer localhost:17070/e175_best.html HTTP200,4865343bytes; staticIIDPNG visuallyinspected. RawXYZ, noalignment. Active50ktraining continues; snapshot static.
