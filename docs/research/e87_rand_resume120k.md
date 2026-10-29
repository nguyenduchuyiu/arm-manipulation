# e87_rand/resume120k

Status: complete. Training: /workspace/checkpoints/e87_rand/resume120k.

User requested60000additional optimizer updates for the two completed state-input
experiments; e89_goal remains stopped. Continue from /workspace/checkpoints/e87_rand/latest.pt at60000,
which is hardlinked to the parent's final.pt. Train steps60001–120000; no new
random initialization or reset of optimizer,RNG,sampler. Original initialization
was random(seed0). Preserve all parent checkpoints, validation and held-out results.
New artifacts are under this fresh subdirectory; do not pre-create it.

State input: measured EE base XYZ3,absolute rotation6D6,measured right-jaw1.
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

Tmux:huy:e87r.
Log:workspace_analysis/results/e87_rand/resume120k/logs/train.log.
Parent specification: [e87_rand.md](e87_rand.md). Actual configs in config.json,
normalization.json,split.npz,evaluation_indices.npz,resume_checks.json,history.json,
validation_history.json,progress.json and final summary.json.

Run from repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/e87_rand_resume_cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/fm_ee_train.py --state ee --updates 120000 --resume-from /workspace/checkpoints/e87_rand/latest.pt --output /workspace/checkpoints/e87_rand/resume120k
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
      "EE_delta_mean_mm": 7.906952857971191,
      "EE_delta_RMSE_mm": 15.72999382019043,
      "EE_delta_median_mm": 2.8630528450012207,
      "EE_delta_p90_mm": 21.1666259765625,
      "rotation_mean_deg": 3.734701290433146,
      "rotation_p90_deg": 10.562584413537277,
      "gripper_MAE_mm": 0.023280546069145203,
      "seed": 0
    },
    "chunks_n0": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.1480125188827515,
      "EE_delta_RMSE_mm": 3.574540138244629,
      "EE_delta_median_mm": 0.5421082973480225,
      "EE_delta_p90_mm": 1.9335488080978394,
      "rotation_mean_deg": 0.3994416788827631,
      "rotation_p90_deg": 0.5803156032125776,
      "gripper_MAE_mm": 0.3862825632095337,
      "seed": 0
    },
    "rollout_n0": {
      "cases": 128,
      "final_mean_mm": 7.766831430585946,
      "final_median_mm": 5.278160814724953,
      "final_p90_mm": 17.46276705571308,
      "collision_rate": 0.046875,
      "clipped_frame_fraction": 0.39157353481294926,
      "success_1mm": 0.0234375,
      "success_2mm": 0.1328125,
      "success_5mm": 0.484375,
      "success_10mm": 0.7578125,
      "success_20mm": 0.890625,
      "ik_over_1mm_fraction": 0.012083333333333333,
      "ik_mean_rotation_deg": 0.06743333835782635
    },
    "initial_n1": {
      "chunks": 3202,
      "valid_actions": 80050,
      "EE_delta_mean_mm": 7.847898960113525,
      "EE_delta_RMSE_mm": 15.611927032470703,
      "EE_delta_median_mm": 2.8511147499084473,
      "EE_delta_p90_mm": 21.013254165649414,
      "rotation_mean_deg": 3.7173432273270772,
      "rotation_p90_deg": 10.410991602958958,
      "gripper_MAE_mm": 0.023693125694990158,
      "seed": 1
    },
    "chunks_n1": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.1686843633651733,
      "EE_delta_RMSE_mm": 3.8877651691436768,
      "EE_delta_median_mm": 0.5435518026351929,
      "EE_delta_p90_mm": 1.934887409210205,
      "rotation_mean_deg": 0.40422509180526073,
      "rotation_p90_deg": 0.5877497667405117,
      "gripper_MAE_mm": 0.3775569796562195,
      "seed": 1
    },
    "rollout_n1": {
      "cases": 128,
      "final_mean_mm": 7.496104346324227,
      "final_median_mm": 4.9311294889512745,
      "final_p90_mm": 16.043732087217144,
      "collision_rate": 0.046875,
      "clipped_frame_fraction": 0.3971150781000956,
      "success_1mm": 0.0390625,
      "success_2mm": 0.1171875,
      "success_5mm": 0.5234375,
      "success_10mm": 0.765625,
      "success_20mm": 0.90625,
      "ik_over_1mm_fraction": 0.005598958333333334,
      "ik_mean_rotation_deg": 0.05582268067958604
    },
    "train_rollout": {
      "cases": 32,
      "final_mean_mm": 6.470905966370368,
      "final_median_mm": 5.912103363510413,
      "final_p90_mm": 13.345143023725416,
      "collision_rate": 0.03125,
      "clipped_frame_fraction": 0.40510416666666665,
      "success_1mm": 0.0,
      "success_2mm": 0.15625,
      "success_5mm": 0.46875,
      "success_10mm": 0.75,
      "success_20mm": 0.96875,
      "ik_over_1mm_fraction": 0.0016666666666666668,
      "ik_mean_rotation_deg": 0.038883894281140995
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
  "elapsed_s": 21173.790803882992
}
```
