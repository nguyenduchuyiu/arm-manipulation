# e102 target conditioning probe

Status: complete.

Purpose: test whether the actual e102 checkpoint uses the color target and binds
it to the correct object position,especially on unseen identity permutations.
The pick task is constant; the intervention variable is target_id,not a task label.
No architecture/action-model retraining or change. Pretrained checkpoint:
/workspace/robotics_assets/experiments/e102_top/train/best.pt,step2500;original
training3000updates,batch128,1.344passes,FM2e-5/visual1e-4,ViT-B/16frozen,
H25x10D EE-delta plus absolute jaw. Reach initialization115000.

Use all928independent dataset scenes and all4home-start expert branches.
For each scene,repeat exactly the same top RGB from target0's lossless start.png,
measured joints and Gaussian noise; change only target_id=0,1,2,3. Seeds0/1/2,
Euler10,H25,also evaluate the firstK10predicted actions. There is no physical
rollout or servo execution in this offline probe. Previously completed rollout
configurationH25/K10/25Hzpolicy/50Hzservo/300frames remains unchanged.

Compare correct query against the other three target-ID interventions using the
same expert label; quantify normalized10D MSE,XYZ RMSE,query-pair trajectory
separation relative to experts,and nearest-expert-branch retrieval(chance25%).
Both predictions and four expert references come from the same exact physical
home state. Query sensitivity alone is not evidence of correct target binding.
Compare changes at goal-token and initial velocity-output levels.

Fit fixed linear ridge(lambda10) readouts on2048training-home goal activations
only,train-only feature standardization. Read out target slot(4classes) and
baseXYZ. Evaluate independently on val/composition/spatial/combined. Controls:
image-only averaged goal tokens repeated for all queries,and target-ID-only
chance25%(balanced slot/color). No hyperparameter selection using test splits.
Linear decodability is not a bound on nonlinear representational information.

Run: python -m OATFlow.top_pick.probe from repository with its .venv activated.
Run in tmux e102p with stdout/stderr tee to the experiment logs. Results belong
in /workspace/robotics_assets/experiments/e102_top/probe_task. Exact config.json,
sensitivity.json,linear_probe.json and raw arrays.npz make the intervention
reviewable. No existing experiment outputs are overwritten.

Actual results:sensitivity.json,linear_probe.json,arrays.npz.

Completed result:changing target causes only0.293mm average predictedH25XYZ trajectory separation on train,
versus12.180mm between expert branches(2.41%). FirstK10separation0.0947mm versus1.920mm(4.93%).
Nearest expert-branch retrieval:train25.03%,unseen permutations25.03%,spatial25%,combined25%(chance25%).
Wrong query increases normalizedH25predictionMSE by only0.0059%ontrain and0.0067%onunseen permutations.
Goal-token relative query change9.06%,initial velocity change0.45%. There is a nonzero target pathway,
but its effect on generated initial actions is extremely weak.
Linear ridge slot readout:train53.47%,unseen permutations27.34%,spatial26.95%,combined28.12%;
image-only control25%throughout. Onunseen permutations,linear XYZ error97.18mm.
This readout is trained separately and is not the policy's slot-prediction accuracy.
Conclusion:current checkpoint nearly ignores the target when generating initial-home actions;
it has not established useful target-to-position binding,even ontrain. This does not prove the
architecture cannot learn stronger conditioning or that targets are ignored at every later state.
Plots:conditioning.png,trajectories.png. Exact metrics:sensitivity.json,linear_probe.json,summary.json.
