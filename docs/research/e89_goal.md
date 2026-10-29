# e89_goal

Status: stopped by user. Training: /workspace/checkpoints/e89_goal.

Goal-only ablation of e87_rand/e88_joint. Policy conditioning is absolute goal
XYZ3 in the same base frame; no measured EE pose, joint state, gripper state,
history, elapsed episode time or chunk index. Goal is not made relative to current EE.
One goal context token; proprio MLP and state projection removed. Same16layer720wide
FM expert,goal encoder,action projections and unchanged10D output.
Internal empty state tensor[B,0] carries no values. Diffusion/flow time and Gaussian
noise remain FM internal inputs, not measured robot state or trajectory progress.

Same e86_grid/ee_delta_25hz_h25_s2 labels:32017trajectories,2065104chunks,
H25 at25Hz,chunk-start stride2,padding masked. Output delta XYZ3,relative
rotation6D6,absolute right-jaw command1. Anchors remain fixed measured EE at each
chunk start. Rotation label R_future@R_anchor.T; first two columns,column-major.
Controller alone uses current EE pose to decode target=anchor+delta and
R_target=decoded_delta_R@R_anchor, then identical bounded poseIK/servo. It uses
measured joints for IK/physics, without feeding them to the policy.

Same voxel-disjoint seed78 split:25613/3202/3202goals and
1651684/206772/206648chunks; eval seed79. No filtering to initial chunks,
no new data and no copied action arrays. Same shuffled chunk sampler,seed0.
Goal XYZ and output pose9 use train-only mean/std, padded slots excluded.
Output jaw fixed range g_norm=-1-2*g/0.0255,inverse g=-0.0255*(g_norm+1)/2;
mean/std jaw0/1. No input state normalization exists. Dataset and sample outputs
remain physical units. Verify exact paired splits/eval indices/action+goal stats.

Training from random initialization, no learned model/backbone/optimizer loaded;
all remaining learned parameters train.60000updates,B256,BF16autocast/FP32labels,
fusedAdamW betas0.9/0.95,weight_decay1e-4,clip1;100step warmup to2e-4,cosine
to2e-5 at60000. Concurrent RTX4090 runs share compute; update schedules unchanged.
Retain resumable latest(model/optimizer/Torch+CUDA+NumPy RNG/sampler),best model;
final hardlinked to latest,no milestone copies. Fresh output created by trainer.

Preflight passed: split/evaluation indices exactly equal to e87/e88, action/goal
statistics identical to e87. No state arrays loaded, no proprio/state projection
modules, empty state statistics and one goal context token. Actual B256 CUDA
BF16 loss/backward/fusedAdamW and goal gradients passed,104864842parameters,
peak4.43GiB CUDA. Identical goal+noise yields bit-exact identical plans across
two different expert anchors; sample jaw inverse/rotation validity verified.
Two physical rollouts after one disposable random-model update completed,0/2
success with collisions/clipping; execution wiring check only, not learned
performance. No preflight weights retained for training. Exact results:
workspace_analysis/results/e89_goal/preflight.json and logs/preflight.log.
32goal oracle before training.
Validation every5000updates:2048chunks/32goals,H25/K15,flow10,seed0,300frames;
same PoseIK max40nfev,rotation0.05m/rad,posture1e-4,bias/gain compensation.
Selection stable5mm success then median final EE error. After60000:3202initial
held-out chunks,8192expert-state chunks,128goals/noise0and1,32train rollouts,
128goal oracle. These learned evaluations are not completed.

Interpretation: the same goal labels many different expert chunk anchors, with
different remaining displacements/rotations/gripper commands. Without state or
progress the model learns their conditional mixture; it cannot identify the
current trajectory phase. Keep all chunks for this requested input ablation.

Canonical description here; trainer copies README to artifacts. Config.json,
normalization.json,split.npz,evaluation_indices.npz,startup_checks.json,
progress/history/validation_history,checkpoints and final summary record actual status.
Tmux:huy:e89. Log:workspace_analysis/results/e89_goal/logs/train.log.
Environment:project.venv,Python3.11,Torch2.7.0+cu128,MuJoCo3.15.0,RTX4090.
Run from repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/e89_pycache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/fm_ee_train.py --state none --updates 60000
```

Resume own latest with the same command plus --resume.

Launch verified on 2026-10-14: running in huy:e89, reached update300/60000, recent mean loss0.297167. First resumable latest.pt saved at100updates. Actual training artifacts rechecked: splits/evaluation indices and action/goal normalization exactly match e87/e88; state normalization empty. Startup source=null,input state[B,0],output[B,25,10],no state modules, nonzero goal gradient. Oracle32/32 stable1mm success, no collisions/clipping. e87/e88 remain active concurrently; learned validation/final held-out evaluation pending.

Stopped by user on 2026-10-14. Last logged update17400; intact resumable latest.pt at update17000. All checkpoints, logs and prior validation retained. No final60000update/held-out evaluation completed. Reason: goal-only input cannot distinguish current states/phases with different chunk-relative action labels. e87/e88 continue.
