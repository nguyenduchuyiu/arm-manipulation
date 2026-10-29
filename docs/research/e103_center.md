# e103_center

Status: complete. Training,conditioning probe and128learned rollouts completed.

User request: train identically with an auxiliary target-object-center loss at
tracker hidden state. The actual e102 visual binding module is its two-layer
target-query decoder,not a TCOW tracker. Attach LN256+Linear3 at its256D hidden,
before the existing256to960 goal projection. The FM conditioning path remains
the original hidden-to-goal-token path. Ground-truth centers never enter policy
inputs. XYZ auxiliary predictions are not substituted for the goal token.

L=L_FM+1.0*MSE(predicted normalized current target XYZ,normalized label).
Means/stds fit only on the285600training action-valid frames. Read current
body center from data labels in robot-base metres at the exact input frame,
including approach/grasp/lift. Same train/val frame order and frozen ViT cache
as e102;verify all measured states,target IDs and timestamps when aligning labels.

Same fresh e102 initialization:reach best115000 plus ImageNet ViT-B/16,seed102,
visual decoder initialized identically. Do not start from e102 trained weights.
Auxiliary head seed103 is isolated with fork_rng so adding parameters preserves
original CPU/CUDA training RNG. Shared model parameters must match exactly.
Same3000updates,batch128,383904samples,1.344passes,FM2e-5/visual1e-4,
100warmup then constant,AdamW(.9,.95),wd1e-4,clip1,BF16. ViT frozen;FM and visual
conditioning train. Same fixed512validation chunks and noise/time RNG;select
best by action FM loss only,not total/auxiliary loss. Log auxiliary MSE and XYZ
error separately. Auxiliary gradients should reach visual/target/decoder and
center head,not frozen ViT or downstream FM/action-goal projection directly.

Actual evaluation:exact same128learned rollouts,two layouts x4permutations
x4queries per train/composition/spatial/combined group,H25/K10,Euler10,seed0,
300frames,25Hzpolicy/50Hzmidpoint servo,PoseIK40nfev. Correct target>4cm,
bilateral jaws10frames;wrong lift fails. Same928home-scene conditioning probe,
four target IDs with identical image/joints/noise,seeds0/1/2,Euler10. Compare
H10/H25query separation and correct-expert-branch retrieval,plus goal linear
slot/XYZ readouts with train-only fitting. Neither test selects checkpoints.

Artifacts:/workspace/robotics_assets/experiments/e103_center.
Original e102 data/cache/checkpoints/results remain intact. New center labels
and statistics stay in the new experiment. Run stages in tmux e103 with tee
logs. Code:OATFlow/top_pick/center.py,center_train.py;evaluate.py/probe.py gain
checkpoint/output arguments and strictly load the auxiliary checkpoint model.
No changes to original action sampling,case selection,noise,controller or scoring.


Completed results:source/shared model initialization,RNG and initial action loss matched
e102 exactly. Old64query probe outputs remain bit-exact after helper extraction and
all128rolloutcase paths/orders match. Source snapshot hashes verified. ViT150tensors
unchanged;reach action/state normalization unchanged. Training513.8s,3000updates,
383904samples/1.3442passes. Selected step2500(same asbaseline),FMval0.0103367 versus
0.0113957baseline. Selected center-head mixed-frame val error10.36mm;final3000error8.89mm.

Rollout e102->e103:train1/32->5/32,composition0/32->4/32,spatial0/32->6/32,
combined0/32->7/32. Total1/128->22/128. New105timeouts,1wrong lift,22successes;
old121timeouts,6wrong lifts,1success. SameH25/K10/Euler10/seed0/300frames/25-50Hz
pipeline. Learned test283.97s. This is one short seed-matched ablation,not converged
training or an exhaustive generalization evaluation.

Conditioning probe onidentical928homes,4targets,3seeds:unseen-permutation goal linear
slot readout27.34%->99.90%;correct expert-action-branch retrieval25.03%->53.91%;
query-inducedH25XYZseparation0.293->8.295mm(expert12.18mm). Wrong/correct targetMSEratio
1.00007->4.333. Fornewpositions/combined,the newlinear readout95.70%/96.88%,
actionbranch49.74%/50.52%. Loss improves target binding and its influence onactions;
action competence is still incomplete. Linearprobe is separately trained and its
accuracy must not be reported as policy grasp accuracy.

Direct auxiliary head onhome states:meanXYZerror33.25mm(composition),45.67mm(spatial),
44.27mm(combined),nearest-correct-object84.08%/64.06%/65.63%. These are worse than
mixed-frame validation10.36mm and do not establish a unique cause of action failures.
Outputs:comparison.json,conditioning_comparison.png,rollout_comparison.png,
probe_task/center_overlay.png,train/validation.png,all stage configs/README/metrics,
checkpoints and physical traces/videos. Code/run references in the experiment README.
