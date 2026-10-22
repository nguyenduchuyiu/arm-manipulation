# Top-view home-start pick

Four visible cubes, no covers, only the original top camera. Reuses the native
scene, contact expert, physical-state recorder, reach FM and pose controller.
Experiment outputs live at `/workspace/robotics_assets/experiments/e102_top`.

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m OATFlow.top_pick.collect
.venv/bin/python -m OATFlow.top_pick.audit
.venv/bin/python -m OATFlow.top_pick.render
.venv/bin/python -m OATFlow.top_pick.samples
.venv/bin/python -m OATFlow.top_pick.features
.venv/bin/python -m OATFlow.top_pick.train
.venv/bin/python -m OATFlow.top_pick.evaluate
```

Run long stages in tmux with stdout/stderr piped through tee. Outputs must be
fresh; no source checkpoint or dataset is changed. `collect --pilot --output PATH`
creates16queries on two layouts for preflight. `evaluate --oracle` uses the retained
pilot recorded EE trajectory to test the25/50Hz controller; it does not test the
learned policy.

32train/4val/4test layouts,16seen/8held-out permutations,4paired target queries.
Geometry and yaw are fixed within each layout. Rejected complete layouts are
recorded. Spatial test samples are new draws in the same range, not extrapolation.
All initial cubes must be visible to the top camera. Query states are bit-identical.

ViT-B/16 is frozen and cached at320px, pooled to64spatial tokens. A two-layer
256-wide target-query decoder replaces the reach oracle XYZ token. All16FM layers,
joint encoder, action projections and normalization come from reach best step115000.
Policy sees current top RGB, target color and measured joints; it receives no
object positions or slot/permutation/layout labels. FM and the new visual query
stack train. Stride1 H25 labels use measured future EE poses relative to the
measured chunk start, with a fixed absolute jaw convention and masked terminal
padding. No simulator kinematics enter the policy.

Training uses3000updates,batch128,100warmup,constant FM2e-5/visual1e-4,BF16,AdamW,
clip1. Best checkpoint is selected by fixed validation flow loss. Report learned
rollout success separately on32queries each for train, composition, spatial and
combined. H25/K10,Euler10,seed0,300frame budget,25Hzpolicy/50Hzservo,PoseIK. Success
requires the correct target lifted>4cm with bilateral jaw contact for10consecutive
frames; lifting another object fails. Results and actual configuration stay with
artifacts. Canonical research note: `docs/research/e102_top.md`.
