# e83_ee

Status: preparing. Change FM action representation to measured EE displacement
relative to the current chunk's initial EE position, then execute through IK.
Data: full e08_arm, original2,065,012H25chunks at25Hz/stride2. Same voxel-disjoint
e78split(seed78). No multimodal expansion is used in this controlled comparison.

Label k: recorded EE[min(2*(start+k+1),T)] - recorded EE[2*start],k=0..24.
Frame is world XYZ; first label is the position at the end of the first40ms,
not a forced zero. Odd terminal20ms intervals clamp to true terminal EE. Keep
the original absolute gripper command as channel4. Mask original padded tails.
Labels come from measured states, not FK of compensated actuator commands.
Every predicted chunk uses one fixed current measured EE anchor for execution.

Input unchanged from e79: target XYZ,current measured joints,absolute current
grasp_site XYZ from fixed exact FK. Same2context tokens and16layer720wide FM.
Initialize backbone/proprio/KV from e70step11710; new4channel action projections
are random. All learned modules train; FK fixed. Train B256,60,000updates,BF16,
AdamW(.9,.95),wd1e-4,clip1,LR2e-4/min2e-5 with e79's recorded schedule.
Normalization from train only; input stats reused from e79, new action stats.

Execution: H25/K15,Euler10,noTE,fresh noise/replan,25Hz/50Hz midpoint servo,
500Hz physics. At each executed point, solve position-only bounded local IK
from measured current joints with weak posture regularization1e-4m/rad,
max30evaluations,analytic Jacobian; add bias-force/gain compensation. IK sees
only the predicted EE waypoint, never the true goal for control. Orientation
unconstrained. Existing collisions and stable1/2/5/10/20mm success metrics apply.

Preflight: replay recorded EE waypoints through the same IK controller on32fixed
validation goals before training. Evaluate every5k on fixed2048validation chunks
and32validation rollouts; select by stable5mm success then median final EE error.
After training: selected checkpoint,128held-out rollouts/noise0and1,all3202initial
held-out chunks,8192expert-state chunks,and32train rollouts. Report position
prediction error separately from physical rollout error and IK residual.

Labels: /workspace/datasets/e83_ee. Training: /workspace/checkpoints/e83_ee.
Tmux huy:e83. Log /workspace/memory_occlusion/logs/e83_ee.log.
Preflight artifacts: workspace_analysis/results/e83_ee/test_ik.
