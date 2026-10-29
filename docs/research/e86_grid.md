# e86_grid

Status: complete; collection, full audit and EE-chunk export passed. Regenerate the full uniform2cm voxel feasibility grid, then
physically verified reaching trajectories and H25 EE-pose delta chunks.

Fresh source data:/workspace/datasets/e86_grid. Source scene is the exact saved
user robot/table configuration at workspace_analysis/results/e08_arm/test_k64/scene.xml.
It already serializes the user's shoulder torque+/-5Nm, elliptic cone, Newton
solver, impratio10 and timestep0.002s. The original source module/robot files are
currently unavailable; no substitute robot or default configuration is used.
Each worker verifies all arm dynamics and table/collision geometry against the
saved reference. The old Sobol IK seed pool is reused solely for initialization;
all72030voxel center checks and dynamic trajectories are recomputed.

Grid:base X/Y multiples of0.02m, worldZ0.01,0.03,...,0.59m. Position-only IK,
orientation unrestricted. Common settled home/open gripper; collision-checked
joint path, Cartesian alignment,0.6s close and0.5s closed hold. Commands50Hz,
physics500Hz. Arrival/final EE error<=1mm, closing/hold drift<=2mm; no penetration
beyond0.2mm. Only robot/table, no objects. Feasibility certifies tested centers
and saved paths, not every point in a voxel or a grasp on an object.

Collection:15CPU workers, random seed42, exact-zero home plus nearest collision-free
Sobol seeds; bounded analytic-Jacobian IK and straight path/RRT-Connect. Smoke
test64centers before full collection; audit all accepted episodes and physically
replay16deterministic episodes after collection. All records/provenance saved.

Final chunks:/workspace/datasets/e86_grid/ee_delta_25hz_h25_s2. H25,25Hz targets,
chunk-start stride2. Action10:world/base-axis delta XYZ3, relative rotation6D6,
absolute jaw command1. Both anchors are the measured EE at chunk start; relative
rotation=R_future@R_anchor.T, first two columns stored column-major. Targets are
measured source states min(2*(start+k+1),T), including odd terminal clamping.
No actuator-command FK labels. Mask tail padding; repeat the final label.

Observation10:EE base XYZ3, absolute rotation6D6, measured jaw1. Base origin
from source config; base/world axes are aligned. Goal XYZ saved in both frames.
Physical units stored; no mean/std applied during export. Model training uses
train-only pose/goal mean/std and fixed-range gripper normalization
g_norm=-1-2*g/0.0255 (open=-1,closed=+1). Gripper remains absolute, never delta.

No training, pretrained weights, optimizer, frozen modules, epochs, batch size
or learning rate is involved in data generation. No learned policy checkpoint,
execution K, flow steps, noise seed or policy frame budget is used. The original
60k training request is not launched by this data-generation experiment.

Logs:workspace_analysis/results/e86_grid/logs/. Tmux:huy:e86.
Exact settings/results:dataset config.json,summary.json,audit.json and chunk
config.json,audit.json. Local research documentation remains here.

Completed: 72,030 voxel centers tested, 34,452 IK-valid, 32,017 dynamically verified trajectories, 8,211,79650Hz commands. All trajectories audited;16physically replayed. Max replay XYZ mismatch 4.44e-08m. Mean/max final EE error 0.03744/0.99847mm.

Final export:2,065,104H25chunks, 1,680,900full and 384,204padded; 46,822,728valid action slots. actions.npy[N,25,10],state_ee_pose_base.npy[N,10]; all pose/mask/state audits passed.

Regeneration changes accepted membership for4voxel IDs relative to olde08 (2added,2removed), and chunk counts/offsets differ. Rebuild training splits from the new voxel IDs/offsets; do not reuse old absolute chunk indices. See comparison_to_e08.json and generation_provenance.json. Training not run.

Storage consolidation (2026-10-16): measured joint inputs now live in /workspace/datasets/e86_grid/ee_delta_25hz_h25_s2/joint_states. The old /workspace/datasets/e88_joint path is a symlink for saved checkpoint paths. Values/chunk order/labels/statistics are unchanged; see /workspace/datasets/e86_grid/ee_delta_25hz_h25_s2/joint_states/relocation.json.
