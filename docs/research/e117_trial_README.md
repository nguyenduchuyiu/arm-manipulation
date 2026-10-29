# E117 free XY pick

Status: pilot COMPLETE; user requested trial and comparison images only. Full
generation and training are not launched. User requested stronger spawn randomization than E102.
Same four32mm cubes, common home, top RGB320x320@25Hz, original contact pick oracle,
geometry/camera/lighting, six absolute joint commands and500Hz physics. No training,
weights, frozen modules, epochs, batch, LR, H/K, flow steps or noise seed applicable.

Propose four uniform points in world XY:[0.30,0.78]m x[-0.42,-0.10]m, reject pair
center distance<55mm, yaw uniform[-pi,pi]. No fixed spawn centers or per-slot jitter
boxes. Sort accepted points by X solely to define consistent left-to-right color
permutation IDs. Whole-layout rejection on any oracle failure, retaining rejected
seed/reason; proposals are not a claim that all of this rectangle is grasp-feasible.
Actual accepted coverage and rejection bias must be reported after physical checks.

Pilot: layouts0,36, permutations0,1 and all four targets =16expert demos. Full plan,
if requested:40layouts,3712queries, same32/4/4 train/val/test layout split and16seen/
8unseen color permutations as E102. Spatial test uses new draws from the same wider
range; this does not create a spatial out-of-distribution test. Existing successful
layout is cached for resume. Source hashes checked on every run.

Code:OATFlow/top_pick/wide.py; minimal shared collect.group extension accepts an
explicit sampler/seed base while default E102 behavior remains the same. Pilot
output:e117_wide/pilot; full output:e117_wide/data. Logs in experiment root and tmux
ssh_tmux:e117. Inspect physical stable bilateral lifts, actual top-camera visibility,
accepted XY spread, nearest train-layout distance and visual comparisons. Keep
summaries and samples; remove finished pilot state/image payloads after verification.

Actual pilot:2accepted layouts,2permutations x4queries each=16successful expert
demos. All16independently checked for terminal target lift>4cm and bilateral jaw
contact for>=10recorded frames, no wrong-object lift; four targets visible from top
in all16initial states. Initial qpos/qvel/ctrl match exactly across four queries
within each layout/permutation. Physical collection13.56s; this is not a learned
policy evaluation. Camera comparisons rerender both old and new states with the
identical current visual model and camera, preserving square image aspect ratio.

Accepted XY extent among eight sampled points:X[0.3009,0.7785]m,Y[-0.3458,-0.1528]m.
Nearest old32training-layout distance, averaged over four X-ordered slots:99.47mm
and71.10mm. Original E102 four spatial test layouts averaged18.77mm. These two
examples demonstrate stronger spatial changes; they do not estimate full-distribution
coverage or generalization. The proposed48x32cm rectangle is not a feasible-workspace
certificate.59proposal layouts rejected before the two acceptances:17separation,
25IK,17physical-pick failures. Selection by the fixed oracle biases accepted poses.

Artifacts:e117_wide/samples/comparison.png (top:old,bottom:new),spawn_positions.png
and.svg (old32training layouts, eight accepted new points, proposed bounds),
verification.json, raw old/new initial PNGs. Code:wide.py,wide_samples.py and the
small collect.group sampler/seed-base parameters. To reproduce fresh trial output:
python -m OATFlow.top_pick.wide --pilot --output NEW_PATH; then MUJOCO_GL=egl
python -m OATFlow.top_pick.wide_samples --data NEW_PATH --output NEW_IMAGES_PATH.
Use .venv/bin/python and tmux+tee for collection. Completed heavy pilot payloads
removed per guide after preserving pilot_config/audit/manifest/progress/scene,
accepted_layouts.json, verification and images. The retained manifest is provenance,
not a ready-to-train dataset after payload cleanup.
