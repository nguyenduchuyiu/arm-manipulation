# e102_top

Status: complete. Data,3000-update training and128learned rollouts completed; poor closed-loop performance.

User request: four random-spawn visible objects, common home start, top camera only,
frozen ViT-Base and FM initialized from reach; evaluate unseen color permutations,
unseen spatial layouts and both. Existing shuffle_pick_multiview was inspected but
its t_obj begins at cover drop and leaves the other cover on the table. User chose
a separate home-start dataset. No source data or checkpoints are modified.

Data: reuse the retained scene, snapshot/physical-state format and contact pick
expert; park both covers outside the workspace. 40 complete feasible layouts:
32 train,4 val,4 test. Four nominal slots perturbed by +/-18mm X,+/-45mm Y,
plus +/-6mm XY offsets, yaw uniform[-pi,pi]. Layout geometry fixed across all24
color permutations and four paired queries; every initial state is query-independent.
16 seen/8 unseen permutations, balanced color/slot marginals. Group sizes:
train2048,val256,composition1024,spatial256,combined128. Reject entire layout on
any query failure, retaining all rejection seeds. Top RGB320px25Hz only; common
settled home. Physical expert success: correct target >4cm with bilateral jaw
contact for ten recorded frames. No cover, shuffle, placement or artificial
mid-trajectory intervention. All action-valid frames become stride1 H25 chunks.

Model: frozen torchvision ViT-B/16 ImageNet1K V1 with320px positional interpolation,
64 pooled spatial tokens. New two-layer256-wide target-query decoder produces
one960-dimensional visual goal token, replacing the reach model's oracle XYZ token.
The second context token retains its pretrained measured-joint encoder.
Exact entire10D H25 FM and normalization initialized strictly from
robotics_assets/models/reach_joint_delta_ee/best.pt,step115000. ViT freeze verified;
visual query stack and FM train. No object XYZ, slot, layout or permutation ID enters
policy. Target color identity and six measured joint/jaw coordinates are inputs.

Actual pilot training: 3000 updates,batch128,AdamW beta(.9,.95),wd1e-4,clip1,
BF16; FM LR2e-5,new visual stack1e-4,100-update warmup then constant.
Masked flow loss, preserved reach normalization. Fresh optimizer; not a continuation
of the reach optimizer. Cache frozen top features once. Validation selects best
masked loss using fixed noise/time samples; held-out test splits never select models.

Actual tests: H25/K10,Euler10,noise seed0,300frames,25Hz policy+50Hz midpoint servo
and500Hz physics,reach PoseIK,max40nfev. Paired target queries per scene. Report train,
composition,spatial,combined separately. Correct target >4cm,bilateral contacts for
10 consecutive frames; any wrong-object lift fails. Test settings and actual
checkpoint step are recorded in each test README/config. Learned-action tests are complete; results below. Training loss is not grasp success.

Code: OATFlow/top_pick/. Artifacts: /workspace/robotics_assets/experiments/e102_top/.
See data/config.json,data/audit.json,train/config.json and test_k10/config.json for
actual configuration. All stages/logs in tmux e102, tee logs beside artifacts.

Preflight completed: exact tensor equality for the full initialized FM against reach
best.pt; all frozen ViT parameters have requires_grad=False,remain eval under
policy.train(),receive no gradients; new visual stack and FM receive finite gradients.
Pilot generation16/16 successful queries,two layouts,two permutations. Top renderer
checks all four objects have visible pixels. Controller-only measured-EE oracle
with25/50Hz PoseIK completed4/4 pilot lifts,so the action representation/controller
can perform this task. This is not a learned-policy result.

Initial spatial audit: four held-out layouts have average per-slot displacement
15.62–24.21mm to their nearest training layout(mean18.77mm). Composition shares
train coordinates exactly and excludes all16train identity permutations. Spatial
and combined use four new layouts from the same sampling range.


Completed result:3000updates,383904chunks,1.3442train passes,434.9s training.
Best step2500; validation loss0.0113957 versus initial0.9324305. Exact post-training
verification:all150ViT tensors unchanged,FM weights updated,normalization preserved.
128preselected learned rollouts atH25/K10,32each:train1/32(3.125%),composition0/32,
spatial0/32,combined0/32.121timeouts,6wrong-object lifts,1success. Evaluation296.5s.
Success requires correct target lift>4cm and bilateral contact for10frames.
Videos/traces and actual configuration are in test_k10. No test-based checkpoint
selection or hyperparameter change was performed. Test only samples two of the
four held-out layouts and four permutations per group; it is a pilot,not a full
3712-query evaluation.

Low validation loss did not translate to closed-loop grasping. Additional fixed
seed0/Euler10 teacher-forced diagnostic on64random training episodes:EE error
from home mean11.49mm,median6.89mm,rotation3.98deg; random expert-state error
mean3.15mm,median1.40mm,rotation0.53deg. This is a measured discrepancy,not proof
of a unique failure mechanism. With only1.344passes and train success3.125%,
this run does not isolate generalization failure or establish that frozen ViT
cannot learn the task.

All experiment artifacts/readmes finalized under
/workspace/robotics_assets/experiments/e102_top. Reproduce fresh stages with
OATFlow.top_pick modules or the saved run.sh. Actual config/hashes/weights and
sample images remain reviewable. Source dataset and reach checkpoint untouched.
