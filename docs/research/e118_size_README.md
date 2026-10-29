# E118 — Cloud-size control at one original held-out reach goal

Status: COMPLETE — rollout and video verification passed. No training or fine-tuning.

Test original best120k PointCloudReach (point MLP/maxpool/goal projection/FM)
on the first fixed held-out reach goal: trajectory12662,voxel30254,
world center[0.459369022999,0.1439727628473,0.29]m. Selection precedes results,
not selected for baseline success. All three conditions share the same original
home state, reachable goal, original air-reach physics, checkpoint, camera,
128point count, sampling rule, action units and identical noise at each replan.
Only the cube's extent changes:20mm(training size),32mm,48mm. This tests object
extent, not increasing point count alone.

Input rendering exactly follows E118: ideal pinhole ray/axis-aligned-box depth,
camera at base+[0,-1.2,.95] looking at base+[0,0,.25],640×480,verticalFOV70°,
2mm optical-axis depth quantization, back-project into robot base coordinates.
Restore the original per-goal RNG state and verify the20mm cloud is bit-exact
to the saved E118 cloud before running physics. Larger cubes use the same
camera/goal/sampling rule, with128points. No per-cloud centering/scaling.

H25/K15,Euler10,paired noise seed0,300frames,policy25Hz/servo50Hz/physics500Hz.
Every condition receives identical sampled noise at every replan; camera and
physics do not change between conditions. The voxel is input/visual geometry
only, not a physical collider, matching original E118 training. No RGB, E120
decoder/weight, object mask inference, scripted grasp/lift or goal correction.

E118 source training:120k scratch updates,B256,25,613training voxel goals and
1,651,684chunks (~18.6passes),AdamW peak2e-4,warmup100,cosine through60k to2e-5,
then constant. H25 predicts anchor-relative EE deltaXYZ,relative rotation6D,
absolute jaw. All modules trained in the source experiment; no updates here.

Score original reach criterion: final25frames below position threshold, jaw
within1mm closed, EE speed<10mm/s, no penetration>0.2mm throughout. Report final
and minimum position error plus collision, not merely a final screenshot.

Video: three synchronized columns, actual input clouds and EE paths, same-scale
cloud insets. Render camera is explicitly checked against input camera pose.
12.5FPS replay samples every second25Hz state;151frames,1920×960. Verify full
decode, unchanged saved rollout/checkpoint hashes and replay EE agreement.

One held-out goal and one noise seed cannot establish global size robustness.
Exact inputs/config in `config.json`, `clouds.npy`; results in `results.json`;
physical traces in `rollout.npz`; video `comparison.mp4`, checks `verification.json`.

```bash
MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.reach.size_rollout
```

Results (same goal/noise, all3no collision):

| Cube side | Final EE error | Stable10mm |
|---|---:|---|
| 20mm | 6.166mm | True |
| 32mm | 10.295mm | False |
| 48mm | 15.847mm | False |

The baseline20mm cloud is bit-exact to the original cached sample. Native visible pixel counts29/80/192; network inputs each contain128sampled points. Accuracy decreases with larger extent at this one goal; this is not an overall robustness estimate. All three fail the stricter stable5mm criterion. No physical voxel/table interaction was introduced.

The main video uses a2.67× display-only crop of the verified original camera view, making robot motion visible. `comparison_full.mp4` retains the uncropped original camera view. Input cloud/render calibration is unchanged. Both versions fully decoded (151frames); replay EE, trace and checkpoint hashes verified.
