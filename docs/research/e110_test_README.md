# e110_trackon — held-out XYZ test and comparison videos

Status: planned. Primary checkpoint is train_2/best.pt, selected on validation before testing: epoch79/step8374 from the run completed through100total epochs. Fixed baseline is train/best.pt, epoch10/step1060. Both checkpoints use identical frozen perception features and measured states on the same test observations; checkpoint selection does not use this test. Checkpoint SHA256/config/source snapshot are recorded in config.json.

Training provenance: full2,048query episodes,134,726stride10points per epoch, batch1280; initial10epochs plus90resumed epochs. Train visual projections/view embeddings/four66token960Dcontext blocks/XYZ head with normalized current XYZ MSE weight1. Freeze original official Track-On-R/DINOv3 (one target click +20x20support grid,memory72), ImageNet ViT-B/16 on cam-left as a temporary wrist substitute, and reach-trained FM/state encoder (source115000,10D EE-delta/rotation6D/jaw). Initial LR1e-4context/1e-5head with10epoch cosine; continuation preserved optimizer/RNG at constant1e-5/1e-6. No checkpointing or action training.

Test: fixed96held-out query episodes, exactly32new-order,32new-layout,32both. Reuse e108/e105 case identities via evaluate_context_xyz.cases; none overlap train/val paths. Quantitative evaluation uses540fixed visible/shuffle/post-shuffle/exposed/lift endpoints, selected with the same stride10protocol. Track each causal sampled stream from frame0, including segment terminals. Four target prompts retain independent memory/support grids; share only the current image encoder between prompts. Each expert branch continues its own context-final tracker memory. No GT current XYZ, masks, IDs, future frames or actions enter policy conditioning; initial click is generated from frame0center annotation only.

Report mean/median/p90, threshold rates, phase errors, by-split results and post-shuffle nearest-target binding over24contexts/96queries. Identical side/state inputs across the four prompt choices are checked. Compare10epoch and selected79epoch readouts on exactly the same frozen features. Verify frozen hashes unchanged and recomputed validation trajectory agrees with stored selected-checkpoint predictions to0.5mm. GT XYZ supplies labels and diagnostic overlays only.

Videos: five fixed cases chosen before evaluating results: first train episode, first validation episode, first held-out new-order/new-layout/both case. Frame0initial prompt, top-camera GT center and two XYZ predictions projected back into the top image, current cam-left view, phase/time/error and prediction age. Left panel:10epochs (orange). Right panel: selected best after100epochs,epoch79 (pink). GT green in both. Native top input is center-cropped240x320. Update inference at stride10 (plus terminals), hold predictions causally at25FPS with no future interpolation. Encoded1280x900H264/yuv420p. Every individual MP4 and comparison.mp4 must fully decode with exact source frame counts. Dense video hold errors are distinct from fixed phase-endpoint test metrics.

These are recorded expert demonstrations with XYZ inference, not learned manipulation rollouts. FM is frozen and not used to control the robot. H/K/flow steps/noise/frame budget are not applicable. Actual wrist-camera robustness is not tested. The native query cache created during this test is temporary; remove after prediction trajectories are saved. Keep only small predicted/GT trajectories, endpoint results, source/config, images and videos.

Run from /workspace/arm-manipulation:

```bash
.venv/bin/python -u -m OATFlow.shuffle_bc.evaluate_trackon_xyz --output /workspace/robotics_assets/experiments/e110_trackon/test_xyz
```

Fresh output required; trainer copies this README. Execute in ssh_tmux:1 with stdout/stderr through tee to test_xyz.log. Results pending; canonical notes are docs/research/e110_trackon.md. Exact video paths and per-case metadata are saved in videos/summary.json.
