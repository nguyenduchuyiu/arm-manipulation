# e87_rand

Status: complete. Training: /workspace/checkpoints/e87_rand.

Train reach FM from scratch on regenerated e86_grid data. No pretrained model,
backbone, state encoder, context projection, action head, optimizer or learned
checkpoint is loaded. PyTorch seeded initialization plus standard zero residual/
unit normalization initialization, seed0. All learned modules train; no frozen
modules and no FK inside the policy. Source is null in config.json, with a SHA256
fingerprint of the initial learned parameters.

Dataset:/workspace/datasets/e86_grid/ee_delta_25hz_h25_s2,32017physically verified
trajectories,2065104chunks. Fresh voxel-disjoint80/10/10split,seed78; absolute
olde08chunk indices are not reused. Evaluation sampling seed79. Normalization
uses only new training chunks, with padded action slots excluded.

Input10:EE base XYZ3 + absolute rotation6D6 + measured right-jaw position1.
Goal input:base XYZ3. Output10:world/base-axis delta XYZ3 + relative rotation6D6
against fixed measured chunk-start orientation + absolute jaw command1. No arm
joint input. Rotation encoding:first two columns,column-major. Execute decoded
R_target=delta_R@R_anchor, never compose with a moving anchor inside the chunk.
Pose/goal mean/std are train-only. Gripper fixed range g_norm=-1-2*g/0.0255,
open=-1,closed=+1; inverse g=-0.0255*(g_norm+1)/2. Dataset and model.sample outputs
remain physical units. Gripper mean/std are identity to prevent double scaling.

Architecture:original16layer720wide FM with2goal/state context tokens; recovered
source OATFlow/policy/model.py from repository commit1035c15, vendored as
workspace_analysis/vendor/oatflow_model.py. Code reuse only; no weights loaded.
Environment:project.venv,Python3.11,Torch2.7.0+cu128,MuJoCo3.15.0,RTX4090.
The missing interpreter was repaired; installed CUDA Torch packages reused.

Training:60000optimizer updates,B256,BF16autocast/FP32weights and labels,fused
AdamW(beta0.9/0.95,weight_decay1e-4),gradient clip1. LR2e-4,warmup100updates,
cosine decay to2e-5 at60000; no6krestart. Mask padding in loss. Report samples
seen and effective passes over training chunks rather than a fixed epoch count.
Checkpoint retention:latest with model/optimizer/RNG/sampler,best model,final
hardlinked to latest. No milestone copies because disk space is limited. Atomic
latest writes check free space and leave the previous checkpoint intact on failure.

Preflight:actual CUDA BF16 forward/backward and fusedAdamW atB256, plus oracle
pose-IK replay on32fixed validation goals before training. Validate every5000
updates on2048fixed chunks and32fixed physical rollouts. H25/K15,flow10,
max300policy frames,25Hzwaypoints/50Hzmidpoint servo/500Hzphysics. Bounded local
poseIK,max40evaluations,position+rotation residual,weak posture regularization
1e-4m/rad,bias-force/gain compensation. Gripper commands are absolute metres.

Select checkpoint by stable5mm position/gripper/velocity success, then lower
median final EE position error; rotation prediction and IK residual are reported
separately. After60000updates evaluate the selected checkpoint on all3202initial
heldout chunks,8192heldout expert-state chunks,128heldout goals/noise0and1,
32training goals, and recorded-EE oracle control. Tests are planned, not completed.

Tmux:huy:e87. Log:workspace_analysis/results/e87_rand/logs/train.log.
Artifacts:config.json,normalization.json,split.npz,evaluation_indices.npz,
startup_checks.json,history.json,validation_history.json,latest.pt,best.pt,
final.pt,summary.json. Exact actual counts/settings will be in config.json.

Run from repository root, requiring a fresh output directory:

```bash
PYTHONPYCACHEPREFIX=/tmp/e87_pycache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/fm_ee_train.py --updates 60000
```

Preflight passed on actual RTX4090:B256,H25,state10/action10,BF16 loss/backward,fusedAdamW;104939178parameters,peak4.46GiB CUDA memory. Fresh split:25613train/3202val/3202test goals;1651684train/206772val/206648test chunks. Full train-only normalization passed; no source weights were loaded. Disk space restored on 2026-10-14 by deleting16old checkpoint files (28.10GB); configs, logs and evaluation artifacts retained. Checkpoint cleanup inventory: checkpoint_cleanup_20261014.json. Training launched; checkpoint retention needs about3GBpeak.

Resume this run from its own latest checkpoint (model/optimizer/RNG/sampler and pending validation are restored):

```bash
.venv/bin/python -u workspace_analysis/fm_ee_train.py --updates 60000 --resume
```

Python bytecode cache uses /tmp/e87_pycache because the restored repository __pycache__ contains stale overlay handles.

Launch verified on 2026-10-14: running in huy:e87, reached update400/60000 with loss0.091603. First resumable latest.pt written at update100. Initial shape/gradient checks passed, no source weights loaded; all learned modules train. Oracle preflight32/32 stable1mm successes, median final error0.015896mm, no collisions or clipping. Training and held-out evaluation remain incomplete.


Completed results:

```json
{
  "status": "complete",
  "completed_updates": 60000,
  "best_step": 45000,
  "effective_train_passes": 9.299447109737699,
  "test": {
    "initial_n0": {
      "chunks": 3202,
      "valid_actions": 80050,
      "EE_delta_mean_mm": 9.917328834533691,
      "EE_delta_RMSE_mm": 18.62169647216797,
      "EE_delta_median_mm": 4.1278228759765625,
      "EE_delta_p90_mm": 26.018856048583984,
      "rotation_mean_deg": 4.402851343064536,
      "rotation_p90_deg": 12.123973021458058,
      "gripper_MAE_mm": 0.030792685225605965,
      "seed": 0
    },
    "chunks_n0": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.7001620531082153,
      "EE_delta_RMSE_mm": 4.601681232452393,
      "EE_delta_median_mm": 0.8493605256080627,
      "EE_delta_p90_mm": 3.065643548965454,
      "rotation_mean_deg": 0.5692230655776246,
      "rotation_p90_deg": 0.9166927741482909,
      "gripper_MAE_mm": 0.5143520832061768,
      "seed": 0
    },
    "rollout_n0": {
      "cases": 128,
      "final_mean_mm": 10.66868174105384,
      "final_median_mm": 7.84228319848253,
      "final_p90_mm": 19.101267414971925,
      "collision_rate": 0.0703125,
      "clipped_frame_fraction": 0.5103199852759446,
      "success_1mm": 0.015625,
      "success_2mm": 0.0859375,
      "success_5mm": 0.3515625,
      "success_10mm": 0.703125,
      "success_20mm": 0.8828125,
      "ik_over_1mm_fraction": 0.014427083333333332,
      "ik_mean_rotation_deg": 0.11343983863109885
    },
    "initial_n1": {
      "chunks": 3202,
      "valid_actions": 80050,
      "EE_delta_mean_mm": 9.940196990966797,
      "EE_delta_RMSE_mm": 18.48074722290039,
      "EE_delta_median_mm": 4.186327934265137,
      "EE_delta_p90_mm": 26.226625442504883,
      "rotation_mean_deg": 4.40724260616472,
      "rotation_p90_deg": 12.044434281122353,
      "gripper_MAE_mm": 0.029914475977420807,
      "seed": 1
    },
    "chunks_n1": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.7207294702529907,
      "EE_delta_RMSE_mm": 4.778609752655029,
      "EE_delta_median_mm": 0.8481283783912659,
      "EE_delta_p90_mm": 3.0703351497650146,
      "rotation_mean_deg": 0.5748609368585488,
      "rotation_p90_deg": 0.9275229765427456,
      "gripper_MAE_mm": 0.5054186582565308,
      "seed": 1
    },
    "rollout_n1": {
      "cases": 128,
      "final_mean_mm": 9.360079590217193,
      "final_median_mm": 7.24519926080868,
      "final_p90_mm": 15.99458036995958,
      "collision_rate": 0.0625,
      "clipped_frame_fraction": 0.5183592110784725,
      "success_1mm": 0.015625,
      "success_2mm": 0.0546875,
      "success_5mm": 0.328125,
      "success_10mm": 0.78125,
      "success_20mm": 0.9140625,
      "ik_over_1mm_fraction": 0.012239583333333333,
      "ik_mean_rotation_deg": 0.09939921193361284
    },
    "train_rollout": {
      "cases": 32,
      "final_mean_mm": 9.45328218385452,
      "final_median_mm": 6.004878694112883,
      "final_p90_mm": 22.46946712564171,
      "collision_rate": 0.03125,
      "clipped_frame_fraction": 0.515625,
      "success_1mm": 0.0,
      "success_2mm": 0.03125,
      "success_5mm": 0.53125,
      "success_10mm": 0.71875,
      "success_20mm": 0.84375,
      "ik_over_1mm_fraction": 0.003854166666666667,
      "ik_mean_rotation_deg": 0.061617294378485794
    },
    "oracle_EE_IK": {
      "cases": 128,
      "final_mean_mm": 0.0177242433281493,
      "final_median_mm": 0.014425894984281843,
      "final_p90_mm": 0.03397941413584372,
      "collision_rate": 0.0,
      "clipped_frame_fraction": 0.0,
      "success_1mm": 1.0,
      "success_2mm": 1.0,
      "success_5mm": 1.0,
      "success_10mm": 1.0,
      "success_20mm": 1.0
    }
  },
  "initialization": "random_from_scratch",
  "dataset": "/workspace/datasets/e86_grid/ee_delta_25hz_h25_s2",
  "elapsed_s": 10535.308171375014
}
```
