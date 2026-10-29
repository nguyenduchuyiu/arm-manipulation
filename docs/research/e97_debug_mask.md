# e97_xyz/test_debug/mask

Status: complete. Test missing query/target binding supervision directly using
existing per-view target-amodal masks as an auxiliary attention loss. Prediction
inputs remain only3RGB views, target ID and timestamps; masks are training
labels only. Output remains XYZ. Native ImageNet-initialized architecture adds
a small9->64->3geometric residual reading attention-weighted patch XY and view
mass. No camera calibration or future labels as inputs; no segmentation head.

Diagnostic stage: fixed native spatial features, train native fusion read/FF,
target/context embeddings and XYZ/geometry with MSE+1.0attention cross entropy.
Frozen backbone and zero feedback are a controlled head-learning test, not the
full-finetune claim.1000updates,batch128: first500same-reveal contexts x4IDs;
then50050%reveal/50%opened queries. AdamW3e-4,WD1e-4,BF16,clip1,seed97.
Full-finetune verification then enables all parameters for50additionalupdates,
T1batch16same mixture, backbone3e-6/head1e-4 and same auxiliary loss.
Source official local ImageNet ViT-B/16 spatial weights, zero temporal residuals,
fresh fusion/XYZ. Original train-only XYZ stats. Data512train/32val; no test
optimization. Select opening val; test ordinary30past/current frames on
composition64/spatial32/combined64. No hidden tracking success claim.
No actions/H/K/flow/robot rollout. Best weights only; no feature dumps.
ArtifactsREADME/config/curve/progress/best/test/predictions/causality. Tmux
huy:e97_debug; stdout/stderr tee workspace_analysis/logs/e97_debug.log.

Complete. Head best600: 11.327mm; full50updates T30val11.857mm. Tests in test.json, trained causality checks passed.

Selected fully finetuned checkpoint:full.pt. Geometry+mask control20.736->11.857mm T30validation. See opening_compare.png, id_binding.json and train_preflight.json. Removed disabled history-gate metadata only; tensors verified unchanged.

Checkpoint full.pt step1050 is a job counter; weights start at best head600
then50fullupdates. Normalization restored exactly. No frozen parameters in
full-stage training. Earlier head-only best retained for comparison.
