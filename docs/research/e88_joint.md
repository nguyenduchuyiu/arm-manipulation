# e88_joint

Status: complete. Training: /workspace/checkpoints/e88_joint.

Ablation of e87_rand: replace measured EE pose input10 with measured arm-joint
positions5(rad) plus measured right-jaw position1(m). Goal is the same base XYZ3.
No joint commands, velocities, extra EE inputs or visual inputs enter the policy.
Output remains unchanged10D: base-axis measured future delta XYZ3, relative
rotation6D6 against the fixed measured EE pose at chunk start, absolute jaw1.
R_target=decoded_delta_R@R_anchor. Same H25,25Hz,chunk-start stride2 and padded masks.

Shared labels: /workspace/datasets/e86_grid/ee_delta_25hz_h25_s2,
32017trajectories/2065104chunks. Additional measured input array only:
/workspace/datasets/e88_joint/state_joint_position.npy,2065104x6,float32.
Every anchor audited against source full qpos, measured EE anchor/jaw and chunk order.
No source actions/labels are modified or copied. Fresh export config/audit in that directory.

Same voxel-disjoint split seed78 and evaluation seed79 as e87. Train/val/test:
25613/3202/3202goals;1651684/206772/206648chunks. Verify exact split/evaluation
index equality with e87 before launch. All output and goal normalization matches
e87 exactly. Five measured input joints use train-chunk-only per-channel mean/std.
Input/output jaw uses explicit fixed range g_norm=-1-2*g/0.0255, inverse
 g=-0.0255*(g_norm+1)/2; mean/std jaw entries0/1 avoid double normalization.
Dataset and model.sample outputs remain physical units. Padding excluded from statistics/loss.

Random initialization, seed0; no model/backbone/optimizer weights loaded.
Same16layer720wide FM, two goal/state context tokens; only the first proprio
linear input changes10to6. All learned modules train, no FK inside policy.
B256,BF16autocast/FP32labels,fusedAdamW betas0.9/0.95,weight_decay1e-4,clip1.
60000optimizer updates,100step warmup to2e-4,cosine decay to2e-5 at60000.
Same shuffled chunk order and masked objective. Both e87 and e88 run concurrently
on the RTX4090; throughput differs due to shared GPU, update schedule unchanged.

Preflight completed: all2065104measured-state anchors audited; split/evaluation
indices and action/goal normalization exactly equal to e87. Actual CUDA B256
BF16 forward/backward/fusedAdamW passed, state[256,6]/action[256,25,10],
104938666parameters,peak4.46GiB CUDA. Captured state encoder inputs match
train-only joint mean/std plus fixed-range jaw exactly. Sample jaw inverse scaling
and valid rotation reconstruction verified. Two physical joint-conditioned
rollouts after one disposable random-model update completed (0/2success,
collisions/clipping present); this checks execution wiring, not learned quality.
No preflight weights are retained or used for training. Exact results:
workspace_analysis/results/e88_joint/preflight.json and logs/preflight.log.
32-goal oracle pose-IK preflight before updates. Validation every5000updates:
2048fixed chunks and32physical rollouts, H25/K15,flow10,300frame budget,seed0.
Same PoseIK/controller as e87,max40nfev,rotation0.05m/rad,posture1e-4,bias/gain
compensation. Select stable5mm success then median final EE error. Post-training:
3202initial held-out chunks,8192expert-state chunks,128goals with noise0/1,
32train rollouts and128goal oracle control. These learned evaluations are not run yet.

Retain latest resumable checkpoint(model/optimizer/Torch+CUDA+NumPy RNG/sampler),
best model; final hardlinked to latest, no milestone copies. Config, normalization,
split/evaluation indices, startup checks, histories, progress and summaries are
saved under the fresh training output. Do not pre-create that output directory.
Canonical settings/status here; server README copied by trainer.

Tmux:huy:e88. Logs:workspace_analysis/results/e88_joint/logs/{export,preflight,train}.log.
Python:.venv, cache /tmp/e88_pycache (restored repository cache has stale handles).
Run from repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/e88_pycache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u workspace_analysis/fm_ee_train.py --state joint --updates 60000
```

Resume own checkpoint with the same command plus --resume.

The first export attempt stopped because a fixed absolute tolerance was too tight
for float32 joint storage above2radians. The audit now requires exact equality
after converting recorded full qpos to float32; all anchors pass. Partial output
removed before fresh export; failure retained in export.log/failed_export_config.json.

Launch verified on 2026-10-14: active in huy:e88, reached update400/60000, mean recent loss0.102650. First resumable checkpoint written at100updates. Actual saved split/evaluation arrays and action/goal normalization rechecked identical to e87. Startup input6/output10 gradient checks passed, source=null, all learned modules train. Oracle32/32 stable1mm success, median0.015896mm, no collisions/clipping. e87 remains active concurrently. Learned validation and final held-out evaluation are pending.


Completed results:

```json
{
  "status": "complete",
  "completed_updates": 60000,
  "best_step": 60000,
  "effective_train_passes": 9.299447109737699,
  "test": {
    "initial_n0": {
      "chunks": 3202,
      "valid_actions": 80050,
      "EE_delta_mean_mm": 8.603557586669922,
      "EE_delta_RMSE_mm": 16.67057991027832,
      "EE_delta_median_mm": 3.3168323040008545,
      "EE_delta_p90_mm": 22.749778747558594,
      "rotation_mean_deg": 3.9017993765493,
      "rotation_p90_deg": 10.819407270300973,
      "gripper_MAE_mm": 0.025469545274972916,
      "seed": 0
    },
    "chunks_n0": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.5215599536895752,
      "EE_delta_RMSE_mm": 4.277917861938477,
      "EE_delta_median_mm": 0.728908121585846,
      "EE_delta_p90_mm": 2.7247462272644043,
      "rotation_mean_deg": 0.5080246600631562,
      "rotation_p90_deg": 0.8087613155792718,
      "gripper_MAE_mm": 0.552436113357544,
      "seed": 0
    },
    "rollout_n0": {
      "cases": 128,
      "final_mean_mm": 8.126824609887361,
      "final_median_mm": 5.898535101379423,
      "final_p90_mm": 15.366023466532582,
      "collision_rate": 0.0703125,
      "clipped_frame_fraction": 0.6314568673442794,
      "success_1mm": 0.0078125,
      "success_2mm": 0.0859375,
      "success_5mm": 0.4453125,
      "success_10mm": 0.8359375,
      "success_20mm": 0.8984375,
      "ik_over_1mm_fraction": 0.0042968749999999995,
      "ik_mean_rotation_deg": 0.08405713792672412
    },
    "initial_n1": {
      "chunks": 3202,
      "valid_actions": 80050,
      "EE_delta_mean_mm": 8.745512008666992,
      "EE_delta_RMSE_mm": 16.8311824798584,
      "EE_delta_median_mm": 3.3328752517700195,
      "EE_delta_p90_mm": 23.4158935546875,
      "rotation_mean_deg": 3.9461831683047057,
      "rotation_p90_deg": 10.925667174237153,
      "gripper_MAE_mm": 0.024802109226584435,
      "seed": 1
    },
    "chunks_n1": {
      "chunks": 8192,
      "valid_actions": 185829,
      "EE_delta_mean_mm": 1.5324445962905884,
      "EE_delta_RMSE_mm": 4.441810131072998,
      "EE_delta_median_mm": 0.7248337268829346,
      "EE_delta_p90_mm": 2.7170019149780273,
      "rotation_mean_deg": 0.5067066277451159,
      "rotation_p90_deg": 0.8028607430937711,
      "gripper_MAE_mm": 0.5599232912063599,
      "seed": 1
    },
    "rollout_n1": {
      "cases": 128,
      "final_mean_mm": 7.650617408143788,
      "final_median_mm": 6.578153008337432,
      "final_p90_mm": 12.874812763508643,
      "collision_rate": 0.046875,
      "clipped_frame_fraction": 0.627265625,
      "success_1mm": 0.0,
      "success_2mm": 0.0625,
      "success_5mm": 0.390625,
      "success_10mm": 0.8125,
      "success_20mm": 0.9296875,
      "ik_over_1mm_fraction": 0.0011458333333333333,
      "ik_mean_rotation_deg": 0.0489968563078194
    },
    "train_rollout": {
      "cases": 32,
      "final_mean_mm": 5.603139104403978,
      "final_median_mm": 4.690305710693078,
      "final_p90_mm": 9.94625528892256,
      "collision_rate": 0.03125,
      "clipped_frame_fraction": 0.6497505042998195,
      "success_1mm": 0.03125,
      "success_2mm": 0.125,
      "success_5mm": 0.625,
      "success_10mm": 0.875,
      "success_20mm": 0.96875,
      "ik_over_1mm_fraction": 0.0005208333333333333,
      "ik_mean_rotation_deg": 0.037090774690125304
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
  "elapsed_s": 10576.80981566204
}
```

Storage consolidation (2026-10-16): measured joint inputs now live in /workspace/datasets/e86_grid/ee_delta_25hz_h25_s2/joint_states. The old /workspace/datasets/e88_joint path is a symlink for saved checkpoint paths. Values/chunk order/labels/statistics are unchanged; see /workspace/datasets/e86_grid/ee_delta_25hz_h25_s2/joint_states/relocation.json.
