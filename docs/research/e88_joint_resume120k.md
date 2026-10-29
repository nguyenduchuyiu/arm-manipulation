# e88_joint/resume120k

Status: complete. Training: /workspace/checkpoints/e88_joint/resume120k.

User requested60000additional optimizer updates for the two completed state-input
experiments; e89_goal remains stopped. Continue from /workspace/checkpoints/e88_joint/latest.pt at60000,
which is hardlinked to the parent's final.pt. Train steps60001–120000; no new
random initialization or reset of optimizer,RNG,sampler. Original initialization
was random(seed0). Preserve all parent checkpoints, validation and held-out results.
New artifacts are under this fresh subdirectory; do not pre-create it.

State input: measured five arm joints(rad),measured right-jaw1(m).
Goal base XYZ3; output10: measured future deltaXYZ3,relative rotation6D6 with fixed
measured chunk-start pose anchor,absolute jaw1. Same e86_grid EE labels,H25 at25Hz,
chunk stride2,masked padding. Same seed78 voxel-disjoint train/val/test and seed79
evaluation arrays, copied byte-for-byte; existing normalization reused unchanged.
Pose/joints/goal/output mean/std from train chunks; input/output jaw fixed range
 g_norm=-1-2*g/0.0255,inverse g=-0.0255*(g_norm+1)/2,never delta/double-normalized.

Same16layer720wide FM,two goal/state context tokens,all existing learned parameters
train. Model AND fusedAdamW moments/step restored. B256,BF16autocast/FP32labels,
betas0.9/0.95,weight_decay1e-4,clip1. LR remains2e-5 for every additional update;
continuous at60000, no warmup/cosine restart. CPU/CUDA torch RNG,NumPy sampler RNG,
shuffled chunk order,cursor,samples-seen restored exactly from saved checkpoint.
Checkpoint preflight passed: optimizer states at60000, dataset hashes and statistics
unchanged; exact cursor/next batch proof in local results resume120k/preflight.json.

Carry entire validation history and COPY parent best.pt as incumbent. Checkpoint
selection remains global best across both stages: stable5mm success then lower
median final EE error. If no continuation checkpoint improves, keep parent best.
Never hardlink a writable best checkpoint to the parent. Save resumable latest
at60100 and every1000thereafter, retain best,final hardlinked to latest at120000.
Parent final/latest60000 remain untouched.

Same validation every5000:2048fixed chunks,32fixed physical rollouts,noise0,
H25/K15,flow10,max300frames,25Hzwaypoints/50Hzmidpoint servo/500Hzphysics;
PoseIK max40nfev,rotation0.05m/rad,posture1e-4,bias/gain compensation.
At120000 evaluate selected global-best checkpoint on same3202initial held-out
chunks,8192expert-state chunks,128goals/noise0and1,32train goals,and128goal oracle.
Continuation training/held-out evaluation are not complete.

Tmux:huy:e88r.
Log:workspace_analysis/results/e88_joint/resume120k/logs/train.log.
Parent specification: [e88_joint.md](e88_joint.md). Actual configs in config.json,
normalization.json,split.npz,evaluation_indices.npz,resume_checks.json,history.json,
validation_history.json,progress.json and final summary.json.

Run from repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/e88_joint_resume_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/fm_ee_train.py --state joint --updates 120000 --resume-from /workspace/checkpoints/e88_joint/latest.pt --output /workspace/checkpoints/e88_joint/resume120k
```

Resume the continuation itself with --resume in place of --resume-from PATH;
keep the same --state,--updates120000 and --output.

Launch verified: active continuation at last logged update60500/120000; new resumable latest.pt verified at60100. Optimizer step60000→60001, exact CPU/CUDA RNG restoration and first sampler batch verified. Saved split/eval/normalization copies match parent byte-for-byte. Source60k final/latest remain hardlinked and intact. LR2e-5. Additional60000updates/held-out eval still in progress.


Completed results:

```json
{
  "status": "complete",
  "completed_updates": 120000,
  "best_step": 115000,
  "effective_train_passes": 18.598894219475397,
  "test": {
    "initial_n0": {
      "chunks": 3202,
      "valid_actions": 80050,
      "EE_delta_mean_mm": 7.829586029052734,
      "EE_delta_RMSE_mm": 15.610199928283691,
      "EE_delta_median_mm": 2.7845396995544434,
      "EE_delta_p90_mm": 21.25517463684082,
      "rotation_mean_deg": 3.6914104161147114,
      "rotation_p90_deg": 10.385838423810032,
      "gripper_MAE_mm": 0.027641603723168373,
      "seed": 0
    },
    "chunks_n0": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.348360538482666,
      "EE_delta_RMSE_mm": 3.973177909851074,
      "EE_delta_median_mm": 0.6276323795318604,
      "EE_delta_p90_mm": 2.3977813720703125,
      "rotation_mean_deg": 0.46849672773462436,
      "rotation_p90_deg": 0.7289292538882172,
      "gripper_MAE_mm": 0.5633904337882996,
      "seed": 0
    },
    "rollout_n0": {
      "cases": 128,
      "final_mean_mm": 8.00009130271748,
      "final_median_mm": 4.624245125956605,
      "final_p90_mm": 12.61961658843703,
      "collision_rate": 0.0546875,
      "clipped_frame_fraction": 0.3140068429567939,
      "success_1mm": 0.0390625,
      "success_2mm": 0.1171875,
      "success_5mm": 0.5390625,
      "success_10mm": 0.8359375,
      "success_20mm": 0.921875,
      "ik_over_1mm_fraction": 0.010130208333333333,
      "ik_mean_rotation_deg": 0.11020925507224771
    },
    "initial_n1": {
      "chunks": 3202,
      "valid_actions": 80050,
      "EE_delta_mean_mm": 7.901693344116211,
      "EE_delta_RMSE_mm": 15.716795921325684,
      "EE_delta_median_mm": 2.7924790382385254,
      "EE_delta_p90_mm": 21.398286819458008,
      "rotation_mean_deg": 3.696106204903039,
      "rotation_p90_deg": 10.395331594589193,
      "gripper_MAE_mm": 0.02701610140502453,
      "seed": 1
    },
    "chunks_n1": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.3768221139907837,
      "EE_delta_RMSE_mm": 4.185581684112549,
      "EE_delta_median_mm": 0.6257807016372681,
      "EE_delta_p90_mm": 2.419334650039673,
      "rotation_mean_deg": 0.47548539712298016,
      "rotation_p90_deg": 0.7312052789090769,
      "gripper_MAE_mm": 0.5376734733581543,
      "seed": 1
    },
    "rollout_n1": {
      "cases": 128,
      "final_mean_mm": 5.768533778333644,
      "final_median_mm": 4.203327124452876,
      "final_p90_mm": 10.456484857211903,
      "collision_rate": 0.03125,
      "clipped_frame_fraction": 0.28736631368210314,
      "success_1mm": 0.015625,
      "success_2mm": 0.1171875,
      "success_5mm": 0.6328125,
      "success_10mm": 0.8828125,
      "success_20mm": 0.9609375,
      "ik_over_1mm_fraction": 0.0008333333333333334,
      "ik_mean_rotation_deg": 0.03866864631388724
    },
    "train_rollout": {
      "cases": 32,
      "final_mean_mm": 4.53754804405697,
      "final_median_mm": 3.6202702422180435,
      "final_p90_mm": 8.056145643932286,
      "collision_rate": 0.03125,
      "clipped_frame_fraction": 0.31661409043112515,
      "success_1mm": 0.0625,
      "success_2mm": 0.1875,
      "success_5mm": 0.6875,
      "success_10mm": 0.875,
      "success_20mm": 0.96875,
      "ik_over_1mm_fraction": 0.000625,
      "ik_mean_rotation_deg": 0.03279957294107089
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
  "elapsed_s": 21282.48486135603
}
```
