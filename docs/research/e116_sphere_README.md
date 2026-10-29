# E116 sphere geometry preview

Status: COMPLETE. One video fully decoded598frames (23.92s), with verified
front-only depth and exact context/expert physics replay (qpos/qvel errors0). One physical shuffle/open-cover/grasp/lift video. Change only
four target geometries from32mm cubes to spheres with32mm diameter; retain colors,
30g mass, original friction/contact parameters, robot/covers/camera mounting and
same g001/p00 layout and original shuffle-plan distribution. The cube-accepted
plan lets a sphere escape, so the existing20-attempt context retry policy is used. Both visual and collision geometries
change; spherical inertia is recomputed (2/5*m*r^2). Re-simulate physical dynamics;
do not reuse cube trajectories as sphere motion. Internal Cube body IDs are kept
for existing oracle compatibility; display target is Red sphere.

Camera configuration matches current E116: front RGB-D2mm/unit FFV1, wrist RGB
synthetic equidistant175deg diagonal, no wrist depth. Masks/calibration for both.
Preview rotates wrist RGB/masks180deg, includes opaque camera views and clearly
labelled GT cutaway/relative-motion panels. No learned tracker/policy or training;
checkpoint/epochs/batch/LR/H/K/flow/noise seed not applicable. Same finite oracle
wrist-orientation list from E116, identical terminal context state for each attempt;
rejections recorded. Only one selected successful continuation, not a benchmark.

Outputs: e116_dynamic/test_sphere (scene/config/replay diagnostics) and its
preview/one video. Verify geometric types/inertia, full physics replay, RGB frame
counts, front-depth lossless roundtrip and absence of wrist depth. Preserve small
scene/settings/results and video; remove finished preflight payloads after checks.
Canonical generation code: existing collect_dynamic/dynamic_shuffle/expert and
render_dynamic/wrist_camera/preview_dynamic. Temporary single-case orchestration
is removed after completion. Original full E116 collection continues unchanged.

Collected result: one successful Red sphere demo,250context +348expert =598frames
at25FPS (23.92s). Four context plans rejected for containment before the accepted
plan; seven grasp orientations rejected before wrist[0.8,pi/2] succeeded. Same
original contact/oracle parameters; no attachment, no teleport, no visual-only
shape substitution. Target lift117.6mm, bilateral lift/hold streak21frames; cover
lift118.8mm, streak34frames. Maximum cube-ID sphere displacement relative to cover
about117mm. These are selection diagnostics for one demonstration, not success-rate
or generalization results. Source body IDs retain Cube names; actual geometry is
sphere in both collision and rendering.

Audit: original soft-contact parameters give sampled maximum penetration15.36mm
in context and10.01mm in expert; this is not calibrated real-world contact physics.
Front depth codec roundtrip bit-exact; camera-model/calibration/timestamp and mask
shapes checked; no wrist depth. preview/sphere.mp4 contains the single requested
video. Scene, context plan, rejected attempts, episode metrics, source manifest,
audit, renderer summary and sample calibration retained. Finished intermediate
RGB-D/mask/state payloads and temporary orchestration removed after verification.
To visualize a retained dataset case, preview_dynamic now accepts --episode PATH,
which renders only that manifest entry. Code change is limited to single-episode
selection and a readable sphere target title; current default selection is retained.
