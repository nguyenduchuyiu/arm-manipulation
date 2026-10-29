# E116 dynamic shuffle dataset

Status: PAUSED by explicit user request (2026-10-21). Physical collection is
complete;304rendered segments are retained. Remaining rendering and full audit
are deferred. config.status=rendering_paused. No model training.

40layouts,928shared contexts,3712expert queries, with original train/val/test
INITIAL-permutation/layout partition. Expected episodes:train2048,val256,
composition1024,spatial256,combined128. Final within-cover order is continuous and
can change; do not claim final arrangements remain discretely unseen.

Exactly two cameras,320x240@25FPS:
- Front-oblique: RGB, ideal optical-Z depth uint16 2mm/unit + FFV1, masks, calibration.
- Robot-mounted wrist: RGB fisheye and masks/calibration ONLY. No wrist depth file
  or wrist depth policy input. Synthetic equidistant175deg diagonal, nominal
  Arducam B0261 reference, not measured calibration of that camera or the user's
  unidentified lens. Five90deg cubemap faces supply wide-angle RGB and segmentation.

Both cameras retain intrinsics and framewise camera-to-world/base matrices.
Calibration explicitly stores pinhole/fisheye projection model, K, ideal distortion
coefficients, depth_views=['front'] and display rotation. Wrist RGB/masks rotate
180deg ONLY for visualization; stored raw pixels and calibration stay aligned.
The wrist XML is a mounting reference, not the final projection: reproduce images
with OATFlow.data_tools.wrist_camera.Fisheye. No old top/left/right cameras.
No arbitrary physical focus blur or minimum RGB distance is assumed.

Context: externally driven covers move through mocap welds; cube free joints evolve
under500Hz MuJoCo contact dynamics. No cube qpos/velocity forcing after setup.
Welds release and scene settles before four expert continuations. Complete state,
commands and drive targets permit exact replay. Context action labels masked.
Expert opens/deposits cover, then grasps/lifts target. Finite wrist-orientation
search starts each candidate from the same context terminal state; only successful
demos retained, rejections logged. No object attachment or teleport. Failed plans
and layout seeds retried. Masses/friction/contact softness remain original, not
identified from real hardware; some dropped-cover/table impacts penetrate about
12mm under the soft contact model. No calibrated real-world dynamics claim.

Labels include visible/amodal/occluder/container masks, target FOV validity,
measured joints, timestamps and initial visible query masks from front frame0.
Later GT is supervision/diagnostic only. Shared context stored once per four queries.

Current camera preflight:2contexts,8expert queries,3391unique frames. All10segments
pass full RGB decode, exact uint16 FFV1 roundtrip, front-only depth schema, camera
calibration, and full physics replay with qpos/qvel error0. Independent576-ray
fisheye check:575exact semantic matches, all576 within1pixel boundary tolerance;
OpenCV fisheye projection roundtrip error below6e-14pixels. Payload39.56MB suggests
roughly18.36GB for928contexts (small-sample estimate). New previews:train and
unseen-initial-permutation composition, e116_dynamic/preview_rgb. No learned policy
or tracker in these clips; wrist visible masks and transparent cutaway are GT.

Earlier pilot:4contexts/16successful expert queries, exact replay/codec validation,
all cube-cover contacts, relative motion up115mm. Its wrist was pinhole and had a
near-plane artifact: MuJoCo's default znear scaled with scene extent8m, producing
80mm clipping and revealing geometry behind close occluders. Current rendering uses
absolute1mm/10m geometric clip planes. Original preview/ videos predate this fix
and are not camera-realism evidence. Near clipping is independent of sensor range.

Old dataset correction: original shuffle context manually translated object qpos
with cover position and used mj_forward without stepping physics. Relative offsets
were constant to about1e-13mm. Only its expert continuation was dynamic.

Production:/workspace/robotics_assets/datasets/shuffle_pick_rgbd_dynamic.
Logs/progress:/workspace/robotics_assets/experiments/e116_dynamic, full.log,
ssh_tmux:e116_full. Renderer uses8workers, checks free disk >=3GiB before each
segment, resumes completed matching-schema segments. Collector source/scene hashes
are preserved during this active run; rendering updates camera-output metadata
when it starts. Its initial physics-stage config may still list the superseded
wrist depth plan until render starts. Final output schema is front RGB-D+wrist RGB.

Canonical code:collect_dynamic.py,dynamic_shuffle.py,render_dynamic.py,
wrist_camera.py,audit_dynamic.py,preview_dynamic.py under OATFlow/data_tools.
Collection scene/camera mounting and physics were not changed by the camera fix.
Camera rationale/sources:e116_camera.md. Essential preflight configs, manifests,
audit/render summaries and preview videos are retained; heavy preflight states,
RGB-D payloads and masks are removed after validation.

Preview completion: train554frames and composition678frames fully decoded. Camera
check summaries/manifest/sample calibration retained as e116_dynamic/wrist_check_*
and wrist_calibration_sample.npz; finished preflight payload directory removed.

2026-10-21 recovery: original full collector stopped on invalid layout seeds960009
and960018 (covers overlap). layout_spec was outside the retry try/except.38layouts
were committed despite the main exception:880contexts/3520successful expert demos.
No full RGB-D segments were rendered. collect_dynamic now includes layout sampling
inside the existing retry handler and records the attempted seed even when sampling
fails. Both bad seeds and their valid next seeds were verified; mocked retry and
completed-layout reuse passed. All original source/scene hashes were verified before
editing, original source/config archived, and collection_resume_fix.json records
old/new collector hashes and committed-layout row hashes. Resume verifies every
other source hash unchanged; no integrity checks are bypassed. Current camera-output
metadata explicitly specifies front-only depth. Resume will collect missing layouts
9/18, then render8workers and audit. Original failed log retained in full.log;
resumed output appended there in ssh_tmux:e116_full. Dataset remains incomplete.

User pause:2026-10-21, E116 full renderer stopped;304completed render markers retained.
All physical collection40layouts/928contexts/3712queries is complete. Remaining
rendering and full audit deferred by explicit user request. config.status=rendering_paused.
