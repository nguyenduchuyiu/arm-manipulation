# From Reaching to Manipulation

19 main slides + 3 appendix slides. Each slide has one main idea. Technical details belong in speaker notes.

## 01. Manipulation starts with reaching

*01 · THE FOUNDATION*

![Manipulation starts with reaching](figures/01_voxel_workspace.png)

- Reach a 3D point.
- Reuse that motion skill.
- Add perception and task context.

**Key message:** First learn how to move. Then learn where to act.

**Speaker notes**

The starting assumption is that useful manipulation needs controlled motion through the robot’s feasible workspace. We first learn reaching from coordinates, then transfer the learned motion weights to visually conditioned behavior cloning. The goal is arbitrary-point reaching within the feasible workspace; finite held-out tests do not certify every continuous point.

---

## 02. The basic task: a coordinate in, a motion out

*01 · THE FOUNDATION*

![The basic task: a coordinate in, a motion out](figures/02_reach_policy.png)


**Key message:** FM predicts end-effector deltas; IK computes the corresponding joint motion.

**Speaker notes**

The policy receives a 3D goal in robot-base coordinates plus five measured arm joint positions and gripper position. It predicts a chunk of end-effector deltas and gripper commands. A downstream pose IK solver and servo convert those predictions to joint commands. The policy does not learn a voxel class or directly output joint angles. Each action has translation XYZ, relative rotation 6D, and an absolute gripper command.

---

## 03. Make the continuous workspace finite

*01 · THE FOUNDATION*

![Make the continuous workspace finite](figures/01_voxel_workspace.png)

- Use a 2 cm voxel grid.
- One center defines one candidate goal.
- Keep physically validated targets.

**Key message:** The grid defines training goals; the policy still accepts continuous XYZ.

**Speaker notes**

We cannot enumerate infinitely many continuous goals. We discretize the candidate workspace into 2 cm cubes and use their centers as candidate end-effector targets. The image shows actual accepted centers around the robot, with the original detailed arm visual mesh at its recorded home pose. Candidate grid dimensions are 49×49×30. Acceptance certifies the tested centers and trajectories, not every point inside a cube.

---

## 04. Validate the target and the path to it

*01 · THE FOUNDATION*

![Validate the target and the path to it](figures/03_feasibility_pipeline.png)


**Key message:** Voxel-based reachability mapping + collision-aware planning + dynamics validation.

**Speaker notes**

There are four stages. 1: create 2 cm voxel-center goals. 2: position-only inverse kinematics, leaving orientation free, with joint limits and endpoint collision checks including gripper closure. 3: plan from a common settled home using a straight joint path, then a local RRT-Connect implementation when needed. 4: simulate reaching, gripper closure over 0.6 s, and holding for 0.5 s in MuJoCo. Arrival/final error must be ≤ 1 mm and closure/hold drift ≤ 2 mm, with valid collision checks. 72,030 candidate centers yield 34,452 IK-valid targets and 32,017 physically validated trajectories. Pipeline code is project-specific, built on SciPy least_squares and MuJoCo kinematics, collision, and dynamics APIs.

---

## 05. Turn valid trajectories into motion examples

*01 · THE FOUNDATION*

![Turn valid trajectories into motion examples](figures/04_reach_demonstration.png)

- Record measured EE poses.
- Predict 25 future actions.
- Use one fixed anchor per chunk.

**Key message:** A trajectory becomes many supervised action chunks at 25 Hz.

**Speaker notes**

Raw reaching commands are recorded at 50 Hz with 500 Hz physics, then exported to H25 action chunks at 25 Hz, using chunk-start stride 2. The final dataset has 2,065,104 chunks, including 1,651,684 training chunks. Goal-disjoint trajectory splits are 25,613 train/3,202 validation/3,202 test. Translation and relative orientation are measured against the chunk-start EE pose. Tail padding is masked. The plotted orange curve is a recorded successful expert demonstration, not an FM prediction.

---

## 06. Train FM on the reaching task

*01 · THE FOUNDATION*

![Train FM on the reaching task](figures/05_flow_matching_concept.png)


**Key message:** Learn a flow from Gaussian noise to demonstrated action chunks.

**Speaker notes**

FM means Flow Matching. We initialize the policy randomly and train it on reaching demonstrations. A normalized expert action chunk a and Gaussian noise epsilon define a_tau=(1−tau)epsilon+tau*a; the network predicts velocity a−epsilon with masked MSE. Inference uses 10 Euler integration steps. The 16-layer, width-720 action-expert architecture is adapted from SmolVLA, but learned SmolVLA weights are not loaded for reaching pretraining. Training runs 120,000 updates; the selected checkpoint is 115,000. This diagram is a conceptual synthetic interpolation in action space, not an observed rollout.

---

## 07. Reaching generalizes to held-out targets

*01 · THE FOUNDATION*

![Reaching generalizes to held-out targets](figures/06_reach_results.png)

- 128 held-out goals.
- 83.6–88.3% at 10 mm.
- Median error: 4.20–4.62 mm.

**Key message:** This learns reaching and gripper closure in free space—not object grasping.

**Speaker notes**

Evaluate checkpoint 115,000 on the same 128 held-out goals with inference noise seeds 0 and 1. Use H25/K15, 10 flow steps, a 300 action/12 s budget, 25 Hz policy, 50 Hz midpoint servo, 500 Hz physics. At 1 mm, success is 3.9%/1.6%; at 5 mm 53.9%/63.3%; at 10 mm 83.6%/88.3%. Median final errors are 4.62/4.20 mm and collision rates 5.5%/3.1%. These are two inference-noise evaluations of one trained model, not independent training seeds. Full table is in the appendix. The reaching scene has no objects, so this is not a learned grasp/lift policy.

---

## 08. Grasping adds localization to motion generation

*02 · FROM REACHING TO GRASPING*

![Grasping adds localization to motion generation](figures/07_picking_observations.png)


**Key message:** Localization: find the target. Behavior cloning: learn how to grasp and lift it.

**Speaker notes**

The next task is to pick one of four visible colored cubes from a common home pose. Localization must associate the requested object with its spatial position; manipulation is learned from expert action demonstrations. A target ID says which cube to pick. We vary color-to-slot permutations and physical layouts independently. Unseen permutation means a new color arrangement, not a new multi-grasp order. The collected dataset contains 3,712 episodes; training uses all 2,048 episodes in the training split, exported as 285,600 action chunks. There are 40 sampled spatial layouts (32 train / 4 validation / 4 test), with 16 seen and 8 held-out color-to-slot permutations. The reported evaluation uses a selected 128-case benchmark, with 32 cases in each of four groups. All images here are real recorded initial observations, not policy rollouts.

---

## 09. Connect the visual encoder to FM

*02 · FROM REACHING TO GRASPING*

![Connect the visual encoder to FM](figures/08_visuomotor.png)


**Key message:** FM uses weights learned on reaching, then fine-tunes on grasp demonstrations.

**Speaker notes**

A frozen ImageNet-pretrained ViT-B/16 extracts image tokens. A new target-query decoder uses the task/color ID to form a target latent, projected into the goal-token interface of FM. Joint/gripper state forms the second context token. FM, its state encoder, and projections start from the weights learned on reaching; all FM weights and new conditioning modules are fine-tuned by behavior cloning. Only the ViT remains frozen. The original coordinate-goal MLP is replaced by visual conditioning. Architecture labels use simply FM; its weight provenance is described here.

---

## 10. Shape the latent with auxiliary XYZ supervision

*02 · FROM REACHING TO GRASPING*

![Shape the latent with auxiliary XYZ supervision](figures/09_visuomotor_aux.png)


**Key message:** Train with L = L_FM + L_XYZ. At inference, the XYZ head is unused.

**Speaker notes**

Predict the current target center from the shared 256D target latent with a LayerNorm+Linear XYZ head. XYZ is normalized using training-set statistics and supervised with MSE, coefficient 1. The purpose is to shape the latent so it carries target-location information. FM consumes the shared latent, not ground-truth XYZ or the coordinate prediction. A coefficient of 1 does not imply equal numerical losses or equal gradient contributions. This is e103; removing the auxiliary head/loss gives the e102 baseline.

---

## 11. Auxiliary supervision improves the short-run policy

*02 · FROM REACHING TO GRASPING*

![Auxiliary supervision improves the short-run policy](figures/10_auxiliary_ablation.png)

- Same 3,000 updates.
- 2,048 training episodes.
- Success: 0.8% → 17.2%.

**Key message:** Supervising target location helps the policy use the requested target.

**Speaker notes**

This is a matched short-run ablation: e102 without auxiliary center supervision succeeds 1/128; e103 with it succeeds 22/128. Both train 3,000 updates, about 1.34 epochs, and test best validation checkpoints at 2,500. Use H25/K10, 10 Euler steps, seed 0, 300 frames. A separate diagnostic linear readout shows that target-location information in the latent improves; that readout accuracy is not grasp accuracy. This comparison does not establish a 10-epoch no-auxiliary result. The short-run temporal ensemble test is kept in the appendix.

---

## 12. Ten epochs yield 92.2% grasp success

*02 · FROM REACHING TO GRASPING*

![Ten epochs yield 92.2% grasp success](figures/11_generalization.png)

- 118 / 128 overall.
- 86 / 96 held-out.
- No wrong-object lifts.

**Key message:** New arrangements work well within the tested object set and spatial range.

**Speaker notes**

Continue e103 to a total of 10 epochs, 22,320 updates, and evaluate the final checkpoint. Train 32/32; unseen permutation 32/32; unseen layout 26/32; both unseen 28/32. Overall 118/128=92.1875%; excluding train cases 86/96=89.5833%. The 10 failures are timeouts. This evaluation is a selected 128 case subset, not every collected scene. Known object geometry and the same camera are retained; layout generalization is interpolation within the sampled range. Success requires correct target lift > 4 cm with bilateral contact for 10 consecutive frames.

---

## 13. Weights learned on reaching improve grasping

*02 · FROM REACHING TO GRASPING*

![Weights learned on reaching improve grasping](figures/12_weight_initialization.png)

- Same architecture and loss.
- Same 10-epoch budget.
- 62.5% → 92.2% success.

**Key message:** +29.7 percentage points when FM starts with weights learned on the reach task.

**Speaker notes**

e104 randomly initializes every learned FM parameter, including the state encoder and FM projections, while keeping the frozen ImageNet ViT, the new visual modules, auxiliary loss, data, normalization buffers, and training recipe matched. Evaluate final step 22,320 with K10. Random FM: 80/128 versus FM weights learned on reaching 118/128. The observed gain is 29.6875 percentage points under one training seed and fixed learning rates. This motivates the same FM weight initialization in the final TCOW + FM experiment, but does not prove random initialization could never catch up under another budget or tuned recipe.

---

## 14. A point prompt replaces the task ID

*03 · OBJECT PERMANENCE AND ACTION*

![A point prompt replaces the task ID](figures/13_point_prompt_sequence.png)


**Key message:** Specify the object once, then preserve its identity through the shuffle.

**Speaker notes**

A dot at the target center in the first top-view frame specifies which object to follow. Covers then occlude and shuffle the objects. The model must retain target identity from observation history and localize it later. Subsequent prompt channels are zero. All four panels show recorded observations; the final reveal is from the recorded expert continuation. The images are recorded observations; a separate, explicitly labeled pilot video compares actual GT and predicted centers.

---

## 15. Adapt TCOW to point-prompted 3D tracking

*03 · OBJECT PERMANENCE AND ACTION*

![Adapt TCOW to point-prompted 3D tracking](figures/14_point_prompt_tracker.png)


**Key message:** Top-view observation history + initial point prompt → target-center XYZ.

**Speaker notes**

TCOW stands for Tracking through Containers and Occluders in the Wild, Van Hoorick et al., CVPR 2023. The original model tracks object permanence through occlusion/containment using target, occluder, and container segmentation outputs. We adapt its official causal RGB+query-channel backbone to our top camera, replacing segmentation heads with spatial pooling and an XYZ regression head. A 30 frame past/current prefix and the initial point are inputs; no object ID, ground-truth coordinates, camera calibration, or object masks are model observations. The initial pixel annotation can be generated offline from simulation; only the dot enters TCOW. This slide presents the interface, not a claim of achieved tracking quality. Source: https://tcow.cs.columbia.edu/.

---

## 16. Compare GT and predicted object centers

*03 · OBJECT PERMANENCE AND ACTION*

![Video poster](videos/poster.png)

[Play tracking demo](videos/tracking_demo.mp4)

- Green: ground truth.
- Magenta: prediction.
- Click the video to play.
- Tracking pilot · step 1,250.

**Key message:** Specify the target once; compare its estimated center with simulator ground truth.

**Speaker notes**

Embedded video: videos/tracking_demo.mp4. It shows the two predetermined e100 point-prompted tracking-pilot cases: validation g032/p00/t0, then both-unseen g036/p01/t0. The best checkpoint is step 1,250 after 1,500 training updates on a 512/2,048-episode subset with random phase-weighted prefix sampling. RGB and predictions are from the original completed pilot. Green is dense simulator GT; magenta is the saved causal model prediction, updated at stride 3 plus segment-terminal frames and held until the next output. Only an initial radius-3 dot and 30 past/current RGB frames enter the model. Camera calibration and GT are used only for annotation/evaluation. The XYZ plots show the depth and lift failure as well as image-plane tracking. Robot actions in these recordings are expert demonstrations. These videos are a qualitative pilot example; the e105 full-policy result fields remain blank. The PDF contains a poster; the PPTX embeds the MP4, also supplied separately under videos/.

---

## 17. Train TCOW and FM end-to-end

*03 · OBJECT PERMANENCE AND ACTION*

![Train TCOW and FM end-to-end](figures/15_end_to_end.png)


**Key message:** TCOW supplies a target latent; FM learns cover removal, grasping, and lifting.

**Speaker notes**

In e105, the current TCOW hidden representation is projected to FM’s goal token. Its auxiliary XYZ head shapes the hidden representation; neither ground-truth nor predicted coordinates are fed directly to FM. TCOW is initialized from the original official checkpoint. FM is initialized from the policy trained on reaching, motivated by the earlier matched initialization ablation. With expert action labels, the entire TCOW + FM pipeline is fine-tuned. This replaces a separately evaluated explicit-XYZ cascade; the earlier tracking video is explicitly labeled as a qualitative e100 pilot demonstration.

---

## 18. Learn from video before actions are available

*03 · OBJECT PERMANENCE AND ACTION*

![Learn from video before actions are available](figures/16_training_segments.png)


**Key message:** Use XYZ supervision throughout; add the action loss when expert control starts.

**Speaker notes**

Within each shared context, chronological endpoints at stride 10 cover reveal, occlusion, shuffle, and hold. All action labels are masked; only TCOW, pooling, and XYZ-head parameters update, and FM/goal forward and optimizer updates are skipped. Then expert endpoints at stride 3 train the entire pipeline with L_FM + 1 × L_XYZ. XYZ supervision covers all 30 observed frames. This is segment-dependent training within each shared context, not a dataset-wide TCOW pretraining phase. The current full-data run targets one epoch on 2,048 train episodes: 61,688 context clips and 234,434 expert clips, 296,122 total; batch 4 directly, 74,230 updates, no gradient accumulation or checkpointing.

---

## 19. Evaluate the complete manipulation sequence

*03 · OBJECT PERMANENCE AND ACTION*


**Key message:** Remove the correct cover, deposit it, grasp the target, and hold a stable lift.

**Speaker notes**

End-to-end e105 results are intentionally left blank. Planned evaluation uses the final checkpoint, 128 cases: 32 train, 32 unseen permutation, 32 unseen layout, 32 both unseen. H25/K10, 10 Euler steps, noise seed 0, 900 policy frames/36 s. Only the initial post-shuffle state is restored. The policy must generate all subsequent cover/grasp actions; no expert opening, home reset, or separate lift probe is allowed. Full success requires the correct cover deposited in the designated zone, the correct target lifted > 4 cm with bilateral jaw contact for 10 frames, and other-cover XY displacement ≤ 15 mm. The table must stay empty until actual learned-policy evaluation is available.

| Experiment | Train /32 | Unseen permutation /32 | Unseen layout /32 | Both unseen /32 | Total /128 |
|---|---:|---:|---:|---:|---:|
| TCOW + FM | | | | | |

---

## 20. Reach evaluation: full results

*APPENDIX A · REACH EVALUATION*


**Key message:** One trained checkpoint, two inference-noise seeds; 128 held-out goals.

**Speaker notes**

Checkpoint 115,000 after training to 120,000 updates. H25/K15, Euler 10, 300 actions. Full closed-loop results are shown here. Offline expert-state chunk prediction is a different metric: mean EE delta error 1.35/1.38 mm on 8,192 held-out chunks. Expert-EE pose replay through the controller succeeds 128/128 at 1 mm with median 0.0144 mm and no collision. That reference is not learned FM performance.

---

## 21. Picking ablations: keep the budgets separate

*APPENDIX B · ABLATIONS*


**Key message:** Short-run auxiliary and execution tests are separate from the 10-epoch comparison.

**Speaker notes**

The first two rows use best step 2,500 after 3,000 updates. The K1+temporal-ensemble row uses the same short e103 checkpoint and exponential coefficient 0.01; it achieves 8/128 versus 22/128 for K10. This does not isolate the effect of replanning frequency from ensembling and was not repeated on the 10-epoch checkpoint. The last two rows both use final step 22,320 after 10 epochs. There is no 10-epoch no-auxiliary control in these results. Complete group counts are exported to data/picking_results.csv.

---

## 22. What these experiments establish

*APPENDIX C · SCOPE*

- A reusable EE-delta motion interface.
- Auxiliary XYZ supervision helps target conditioning.
- FM weights learned on reaching improve the tested grasping recipe.

**Key message:** Known objects, fixed camera, simulation, one training seed; end-to-end results pending.

**Speaker notes**

The reaching map certifies sampled centers and paths, not the entire continuous workspace. The picking results cover a selected 128 case benchmark in a fixed simulation with known cubes and one camera. Spatial test layouts are sampled from the same range. No real-robot, novel-shape, or novel-camera result is claimed. A 2D point in TCOW is an input prompt, not a replacement for 3D XYZ auxiliary supervision. All figures generated here distinguish actual data/observations from conceptual illustrations. See sources.md and figure_captions.md for provenance and TCOW attribution.

---
