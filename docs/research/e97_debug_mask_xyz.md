# e97_xyz/test_debug/mask_xyz

Status: complete. Matched no-auxiliary-loss control using
existing per-view target-amodal masks only for attention evaluation. Prediction
inputs remain only3RGB views, target ID and timestamps; masks are training
labels only. Output remains XYZ. Native ImageNet-initialized architecture adds
a small9->64->3geometric residual reading attention-weighted patch XY and view
mass. No camera calibration or future labels as inputs; no segmentation head.

Diagnostic stage: fixed native spatial features, train native fusion read/FF,
target/context embeddings and XYZ/geometry with MSE only; attention CE measured for diagnosis but zero training weight.
Frozen backbone and zero feedback are a controlled head-learning test, not the
full-finetune claim.1000updates,batch128: first500same-reveal contexts x4IDs;
then50050%reveal/50%opened queries. AdamW3e-4,WD1e-4,BF16,clip1,seed97.
Full-finetune verification then enables all parameters for50additionalupdates,
T1batch16same mixture, backbone3e-6/head1e-4 and zero auxiliary loss.
Source official local ImageNet ViT-B/16 spatial weights, zero temporal residuals,
fresh fusion/XYZ. Original train-only XYZ stats. Data512train/32val; no test
optimization. Select opening val; test ordinary30past/current frames on
composition64/spatial32/combined64. No hidden tracking success claim.
No actions/H/K/flow/robot rollout. Best weights only; no feature dumps.
ArtifactsREADME/config/curve/progress/best/test/predictions/causality. Tmux
huy:e97_debug; stdout/stderr tee workspace_analysis/logs/e97_debug.log.

Complete. Head best1000: 21.334mm; full50updates T30val20.736mm. Tests in test.json, trained causality checks passed.

Superseded/duplicate checkpoint weights removed; evaluation metrics retained.
