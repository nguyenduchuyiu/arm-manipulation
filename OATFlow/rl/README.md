# Online FM refinement

Start from a pretrained BC prior. RL collects fresh simulator interactions.

The actor keeps the prior architecture and observations: current overview/wrist
RGB, six normalized joint positions, task12 and target4 embeddings. Only the
FM velocity modules train: action projections, time MLP, 16 action transformer
layers, and output norm. Freeze both cameras' ViT, visual adapter, all
embeddings, proprio/state encoder, context decoder, and context K/V projections.
The actor receives no privileged object coordinates or oracle trajectory.

The [ReinFlow](https://reinflow.github.io/) implementation adds a bounded learned
Gaussian transition-noise head during online PPO. Store the complete 10-step
latent path and sum its actual Gaussian log densities over H25/6 coordinates
and denoising steps. Initial standard Gaussian noise has no trainable density
and cancels in the PPO ratio. The frozen encoding is cached on CPU; update
passes do not re-encode images. Preallocate the bf16 K/V cache and fill it
during collection, avoiding a second full copy at rollout completion.
A privileged critic uses simulator poses,
velocities, contacts, task/query, prior servo command and completion counters.
Discard critic/noise at deployment and use the normal FM ODE sampler.

For pick-only counterfactual priors, the action horizon comes from the BC
checkpoint. `--data` supplies the reset template and held-out scenes. Fresh
training resets randomize slot/object XY with the original pick generator;
robot initial joints, object yaw and color permutation stay matched to the
spatial BC experiment. Four target queries share each training layout.
The pick reward requires the requested object to rise over4cm with both jaws
for ten waypoints; wrong-object lifts and objects falling off the table fail.
There is no cover-removal reward. Both the shape cache and Gaussian path use
the checkpoint horizon, including H50.

Example e70 refinement (fresh output, eight CPU cores and one GPU):

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 20-27 python -u -m OATFlow.rl.train \
  --checkpoint /workspace/checkpoints/e70_cf10/latest.pt \
  --data /workspace/datasets/e68_spatial \
  --output /workspace/checkpoints/e71_rein \
  --execute-chunk 15 --max-frames 300 \
  --iterations 50 --rollout-steps 64 --envs 8 \
  --batch-size 256 --microbatch 4 --lr 3e-7 \
  --eval-every 5 --eval-cases 16 --seed 0
```

The 16 development evaluation cases are g004/g005 ×four targets ×noise0/1.
Evaluate the source prior before any update with the same FP32 FM sampler
used by RL; compare RL against that measured baseline. Frozen encoding is
bf16. Baseline/evaluation uses the regular ODE sampler, not the exploration
noise head. During collection, privileged coordinates enter reward/critic
only. At rollout completion, report actor updates, KL stops, likelihood
recomputation errors and frozen-parameter checks in the run artifacts.

`GraspEnv` resets directly with the same `setup_scene` used by the prior: two
closed covers, two randomized objects each, random cover/table positions,
object yaw, slot order and robot pose. No reveal, shuffle, context rendering or
post-cover curriculum. Balance 12 tasks × two targets × two slot orders.
Derive/verify the task from physical containment at reset and keep it fixed.

H25, Euler10, execute K5 at 25Hz; linear interpolation commands servo50Hz.
TE is off. Each macro transition executes up to five waypoints; physical
contacts, rewards and terminal checks run after every 40ms waypoint. Sum
discounted waypoint rewards; use gamma^m and (gamma*lambda)^m for the actual
executed length m. A timeout bootstraps its final snapshot, stops GAE at the
reset boundary, and never bootstraps from the next episode.

Rewards before a common 0.01 training scale:

- +10 once when the selected cover is supported on the table inside the green
  zone, released from the jaws, and settled.
- +100 terminal when that cover is deposited and the requested target is
  >4cm above its initial height with both jaws for ten 25Hz waypoints.
- -100 terminal on a different object lifting >4cm, or an object leaving the
  table workspace. Merely brushing another object is allowed.
- -0.01 per waypoint, plus potential shaping gamma*Phi(next)-Phi(current).
  Phi follows approach-cover, transport-cover and approach-requested-object
  geometry. Completion latches are critic state. Terminal Phi=0; a timeout
  retains Phi for bootstrapping. This prevents repeatable shaping/cover rewards.

Use float32 for Gaussian means and FM velocity, bf16 for the frozen encoders.
With a joint latent-path density over 1500 coordinates, small actor/noise LRs
are intentional. PPO clips at0.2 and stops the remaining actor updates when
estimated KL exceeds0.03. Critic fitting continues after an actor KL stop.

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
taskset -c 0-7 python -u \
  -m OATFlow.rl.train \
  --checkpoint "$BC_CHECKPOINT" \
  --output "$RL_OUTPUT" \
  --iterations 100 --rollout-steps 128 --envs 8 \
  --batch-size 256 --microbatch 8 --lr 3e-8 --eval-every 10
```

Run long jobs in tmux `huy` with `2>&1 | tee LOG`. The parent owns one CUDA
policy; eight CPU simulation workers render only at replanning boundaries.
No training video is written. Checkpoint `latest.pt` is replaced atomically
after each iteration. `best.pt` is written only after an evaluation improves
on the initial baseline; an RL result may fail to beat BC.

Baseline/evaluation uses 48 fixed fresh seeds covering all query/slot cases,
both covers closed, K5/no TE, noise seed0, 1000 waypoints. Paired target queries
share exactly the same physical reset. Report correct target success, cover
deposit, wrong-object lift and timeouts separately. This seed set is a
development evaluation set, not an untouched final test benchmark.

Resume optimizer, noise, critic and RNG state into a fresh output folder:

```bash
python -m OATFlow.rl.train --checkpoint "$BC_CHECKPOINT" --resume "$RL_OUTPUT/latest.pt" \
  --output "$RL_RESUMED_OUTPUT" --iterations 20
```

Resume resets simulator episodes; it does not restore in-flight trajectories.
Pass `--lr 3e-7` to change only the FM learning rate after restoring optimizer
state. Without `--lr`, resume keeps the saved FM LR. Noise and critic learning
rates and optimizer moments are restored unchanged.
Use the same resource variables as the full command. `config.json`,
`progress.json`, `history.json`, `last_collection.json`, and `eval_*.json`
record the actual configuration and progress.
