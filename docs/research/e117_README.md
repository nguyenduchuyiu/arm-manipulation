# E117 wide-pick RGB-D dataset

Status: COMPLETE. Full collection, rendering and final audit PASSED.
40 layouts, 3712 episodes, 536888 frames; episode payload 5,731,436,026 bytes
(5.73 GB). Physics collection 168.45 s, rendering 5194.19 s, audit 248.63 s.
Old E102 pick data/cache retired successfully; checkpoints/results retained.
Completed log: e117_wide/full.log; final checks: pick_wide_rgbd/audit.json.
User approved the camera samples and requested full rendering and removal of the
old pick dataset. E116 full is paused separately at the user's request.

Four32mm cubes, random XY points in [0.30,0.78]x[-0.42,-0.10]m, pair separation
>=55mm, yaw uniform[-pi,pi], sorted by X only for permutation labels. No fixed
spawn centers. Proposals are filtered by the original physical contact pick
oracle; accepted distribution is selection-biased and the rectangle is not a
certificate of grasp feasibility. Entire layout rejected on any expert failure,
up to1000deterministic seeds per layout; all rejection seeds/reasons retained.

40layouts,32train/4val/4test;16seen/8unseen color permutations;3712expert episodes
(train2048,val256,composition1024,spatial256,combined128). Every episode starts at
the same settled home with all four objects on the table; no shuffle or cover
opening. Controls are six absolute joints (rad,m),25Hz,500Hz MuJoCo physics.
Full MuJoCo integration states, measured joints/EE, commands, phases, timestamps,
object poses, paired query metadata and successful stable bilateral lifts saved.
No training or learned policy; weights/epochs/batch/LR/H/K/flow/noise seed N/A.

Exact camera scene from approved camera_samples/scene.xml. Native320x240@25FPS:
front-oblique RGB + uint16 depth2mm/unit FFV1; mounted wrist RGB synthetic175deg
DIAGONAL equidistant fisheye. No top/left/right or wrist depth. Camera models/K/
distortion and per-frame camera-to-world/base calibrated. Raw wrist orientation
retained; visualization rotates wrist180deg. Lens/depth are ideal simulation, not
measured real-sensor calibration, focus or noise. Geometric clipping1mm/10m.

Each episode exports RGB front/wrist MP4, lossless front depth MKV, visible-instance
masks, packed target visible/amodal masks, FOV flag, frame0 front query mask,
calibration.npz and input.json. Later XYZ/masks are labels, not policy inputs.
Renderer verifies complete RGB decode and exact integer FFV1 roundtrip. Resume
skips completed matching-schema episodes and overwrites interrupted partial media;
source hashes checked. Stop before starting an episode with less than3GiB free.

Final audit checks all media markers/calibration/prompts, split counts/disjointness,
query-independent initial states, all terminal stable bilateral lifts and no wrong
object lift; replays p00 all four targets in each layout with recorded integration
state and controls (pilot:all episodes). Only then config.status becomes complete.

New data:/workspace/robotics_assets/datasets/pick_wide_rgbd.
Experiment/logs:/workspace/robotics_assets/experiments/e117_wide.
Pipeline:python -m OATFlow.top_pick.wide [--resume] -> wide_render -> wide_finish
--retire-old; use .venv/bin/python,tmux ssh_tmux:e117_full and tee full.log.
Code:wide.py,wide_render.py,wide_finish.py. Original pilot/images and approved
four camera samples retained; trial details:e117_trial_README.md,e117_camera_README.md.

Authorized deletion is restricted to /workspace/robotics_assets/experiments/
e102_top/data and e102_top/cache, after the new full3712episode dataset passes
audit. Archive old configs/manifests/layouts/scene as retired_data_* first; retain
old checkpoints/results. Other shuffle/reach datasets and paused E116 are outside
this deletion scope. New source scene is independent of the old data path.

Final-format preflight:16episodes/2330frames, exact full-state physics replay,
all terminal bilateral lifts, query-independent starts, masks/prompts/calibration,
complete RGB decode and uint16 depth codec roundtrip PASSED. Render33.45s with
4workers; audit3.97s. Tested deletion guard refuses a16-episode replacement before
any old-data removal. Heavy completed preflight payloads removed after archiving
configs/audit/manifest/scene/media checks/calibration sample. Full job uses8workers
for physics and rendering. Source hashes frozen before launch; checkpoint/model
artifacts are not deletion targets. Authorized old data+cache total about1.8GiB.

Completion verified 2026-10-21: all 3712 render markers present; all terminal
bilateral lifts and query-independent initial states passed. Old E102 data/cache
paths are absent. Retirement receipt: e102_top/data_retired.json; replacement audit
SHA256:23fc22ea5a588f78229539d1c3a42e05cec4f89e62c7fc8c66afb9d119efd965.
E116 remains paused; no new model training or policy evaluation was performed.
