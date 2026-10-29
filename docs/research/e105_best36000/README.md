# e105_tcow — best checkpoint evaluation at step 36,000

Status: interrupted on user request; evaluation is incomplete. Shared-GPU inference preflight passed (peak allocated1.401GiB, reserved1.539GiB); the main trainer continues.

Purpose: evaluate the current best validation-FM checkpoint before the one-epoch training completes. This is an interim checkpoint evaluation, separate from the planned final-checkpoint test.

Training: run `train_2`, step 36,000, 143,620 samples / 296,122 clips = 0.4850 effective epoch; best validation FM loss 0.0386015748. Full training split: 2,048 episodes / 512 shared contexts. TCOW starts from official RGB/query weights and FM from reach step 115,000. All modules trainable on expert segments; XYZ-only TCOW updates on context segments. Strides context10/expert3, physical batch4, BF16, no gradient checkpointing or accumulation. LR backbone1e-5 / FM2e-5 / new heads1e-4; AdamW(.9,.95), weight decay1e-4, warmup100 then constant. Expert loss L_FM + L_XYZ; no masks or numerical XYZ input to FM. Actions: measured fixed-anchor EE deltaXYZ3, relative rotation6D6, absolute jaw1, H25 at25Hz.

Checkpoint is frozen as `/workspace/robotics_assets/experiments/e105_tcow/best36000.pt`, SHA-256 `4b8cf6663bb9708087ba4ee6d853ecffbbb270ecd8dc7429b096e580fedc0da6`. Snapshot metadata: `best_eval_snapshot.json`; exact training configuration: `train_2/config.json` beside server artifacts.

Test: 128 cases (32 train / 32 unseen permutation / 32 unseen layout / 32 both unseen), H25/K10, 10 Euler steps, inference noise seed0, budget900 policy frames (36s), policy25Hz / servo50Hz / physics500Hz. Initial recorded post-shuffle state only; policy generates cover opening/deposit, target grasp and lift. No expert opening, home reset, GT goal correction, or IK lift probe. Full success: correct cover deposited, target lifted>4cm with bilateral contacts for10frames, other cover displacement<=15mm; wrong-object lift fails.

Run directory: `/workspace/robotics_assets/experiments/e105_tcow/test_k10_2` (reserved; created by evaluator). The planned one-epoch final test retains `test_k10`. Microbatch1 inference, CUDA allocator cap1.65GiB to share GPU with training; numerical inference and success criteria are unchanged. Exact runtime configuration will be recorded in `config.json`; rollouts in `results.jsonl`, trajectoriesNPZ, and target0 videosMP4. Live output in tmux `ssh_tmux:testbest`, log `logs/test_k10_2.log`.

Partial results: 0/48 completed cases; planned128 cases were not completed. This interim result must not fill the one-epoch result table in the slides.
