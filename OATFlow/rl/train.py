"""Fine-tune a pretrained FM velocity field with online ReinFlow/PPO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from OATFlow.rl.env import EXECUTE, GAMMA, LAMBDA, MAX_FRAMES, VectorGrasp, balanced_queries
from OATFlow.rl.model import Critic, FLOW_STEPS, NoiseHead, encode, load_prior, log_probability, sample
from OATFlow.task import TARGETS, TASKS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frozen_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in policy.named_parameters():
        if not tensor.requires_grad:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def restore_actor_optimizer(optimizer, state, lr=None):
    optimizer.load_state_dict(state)
    if lr is not None:
        optimizer.param_groups[0]["lr"] = lr


def gae(rewards, values, next_values, discounts, trace_discounts, terminated, truncated):
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[1])
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + discounts[t] * (~terminated[t]) * next_values[t] - values[t]
        carry = delta + trace_discounts[t] * (~(terminated[t] | truncated[t])) * carry
        advantage[t] = carry
    return advantage, advantage + values


def stack_critic(packets):
    return torch.as_tensor(np.stack([p["critic"] for p in packets]), device="cuda")


@torch.no_grad()
def evaluate(policy, vector, output, iteration, count=48):
    cases = [dict(seed=9000000 + task["task_id"] * 100 + order,
                  task_id=task["task_id"], target_id=TARGETS.index(target), slot_order=order)
             for task in TASKS for target in task["objects"] for order in range(2)][:count]
    results = []
    started = time.perf_counter()
    for begin in tqdm(range(0, len(cases), len(vector.pipes)), desc=f"eval {iteration}", mininterval=5, file=sys.stdout):
        group = cases[begin:begin + len(vector.pipes)]
        for i, case in enumerate(group):
            vector.send(i, "reset", case)
        packets = [vector.receive(i) for i in range(len(group))]
        generators = [torch.Generator(device="cuda").manual_seed(0) for _ in group]
        active = list(range(len(group)))
        progress = tqdm(total=MAX_FRAMES // EXECUTE, desc=f"eval batch {begin // len(vector.pipes) + 1}",
                        mininterval=5, leave=False, file=sys.stdout)
        while active:
            pooled, kv = encode(policy, [packets[i] for i in active])
            # A per-scene RNG makes the initial Gaussian sequence independent
            # of how quickly other vector environments finish.
            noise = torch.cat([torch.randn((1, 25, 6), device="cuda", generator=generators[i]) for i in active])
            actions = policy.flow.sample_actions(noise, kv, FLOW_STEPS).cpu().numpy()[:, :EXECUTE]
            for i, action in zip(active, actions):
                vector.send(i, "step", action)
            remaining = []
            for i in active:
                packets[i], transition = vector.receive(i)
                if transition["terminated"] or transition["truncated"]:
                    results.append(dict(**transition["episode"], slot_order=group[i]["slot_order"]))
                else:
                    remaining.append(i)
            active = remaining
            progress.update(1)
        progress.close()
    record = dict(iteration=iteration, episodes=len(results), success=sum(r["success"] for r in results),
                  cover=sum(r["cover_removed"] for r in results),
                  wrong=sum(r["wrong_object"] is not None for r in results),
                  timeout=sum(r["reason"] == "timeout" for r in results),
                  wall_s=time.perf_counter() - started, results=results)
    write_json(output / f"eval_{iteration:04d}.json", record)
    print(json.dumps(dict(event="evaluation", **{k: v for k, v in record.items() if k != "results"})), flush=True)
    return record


@torch.no_grad()
def collect(policy, head, critic, vector, packets, reset_next, generator, steps, output):
    stored = {name: [] for name in ("pooled", "chain", "sample_logprob", "critic", "values",
                                  "next_values", "rewards", "discounts", "trace_discounts", "terminated", "truncated")}
    completed, frame_count = [], 0
    kv_buffer = None
    for step in tqdm(range(steps), desc="collect K5", mininterval=5, file=sys.stdout):
        pooled, kv = encode(policy, packets)
        action, chain, sample_lp = sample(policy, head, pooled, kv, generator)
        state = stack_critic(packets)
        stored["pooled"].append(pooled.cpu())
        if kv_buffer is None:
            kv_buffer = [(torch.empty((steps * len(packets), *k.shape[1:]), dtype=torch.bfloat16),
                          torch.empty((steps * len(packets), *v.shape[1:]), dtype=torch.bfloat16)) for k, v in kv]
        begin, end = step * len(packets), (step + 1) * len(packets)
        for (dest_k, dest_v), (k, v) in zip(kv_buffer, kv):
            dest_k[begin:end].copy_(k.to(device="cpu", dtype=torch.bfloat16))
            dest_v[begin:end].copy_(v.to(device="cpu", dtype=torch.bfloat16))
        stored["chain"].append(chain.cpu())
        stored["sample_logprob"].append(sample_lp.cpu())
        stored["critic"].append(state.cpu())
        stored["values"].append(critic(state).cpu())
        for i, chunk in enumerate(action[:, :EXECUTE].cpu().numpy()):
            vector.send(i, "step", chunk)
        transitions, next_packets = [], []
        for i in range(len(packets)):
            next_packet, transition = vector.receive(i)
            transitions.append(transition)
            next_packets.append(next_packet)
        stored["next_values"].append(critic(stack_critic(next_packets)).cpu())
        for key in ("discounts", "trace_discounts", "terminated", "truncated"):
            source = {"discounts": "discount", "trace_discounts": "trace_discount"}.get(key, key)
            stored[key].append(torch.tensor([t[source] for t in transitions]))
        stored["rewards"].append(torch.tensor([t["reward"] * .01 for t in transitions]))
        frame_count += sum(t["frames"] for t in transitions)
        packets = next_packets
        reset_indices = []
        for i, transition in enumerate(transitions):
            if transition["terminated"] or transition["truncated"]:
                completed.append(transition["episode"])
                vector.send(i, "reset", reset_next())
                reset_indices.append(i)
        for i in reset_indices:
            packets[i] = vector.receive(i)
    tensor_keys = set(stored)
    buffer = {key: torch.stack(stored[key]) for key in tensor_keys}
    buffer["advantages"], buffer["returns"] = gae(*(buffer[name] for name in
        ("rewards", "values", "next_values", "discounts", "trace_discounts", "terminated", "truncated")))
    for key in tensor_keys | {"advantages", "returns"}:
        buffer[key] = buffer[key].flatten(0, 1)
    buffer["kv"] = kv_buffer
    write_json(output / "last_collection.json", dict(episodes=completed, frames=frame_count,
               transitions=len(buffer["chain"]), timeout_bootstrap=True, reward_scale=.01,
               cpu_buffer_gib=sum(t.numel() * t.element_size() for t in buffer.values() if torch.is_tensor(t)) / 2**30 +
               sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in buffer["kv"]) / 2**30))
    return buffer, packets, completed, frame_count


def batch_from(buffer, indices):
    return (buffer["pooled"][indices].cuda(),
            [(k[indices].cuda().float(), v[indices].cuda().float()) for k, v in buffer["kv"]],
            buffer["chain"][indices].cuda())


@torch.no_grad()
def old_logprobs(policy, head, buffer, microbatch):
    # Use the same flattened denoising batch as the optimizer to avoid shape-
    # dependent floating point errors in the old/new likelihood comparison.
    old = []
    for begin in tqdm(range(0, len(buffer["chain"]), microbatch), desc="likelihood check", mininterval=5, file=sys.stdout):
        idx = torch.arange(begin, min(begin + microbatch, len(buffer["chain"])))
        old.append(log_probability(policy, head, *batch_from(buffer, idx))[0].cpu())
    result = torch.cat(old)
    error = (result - buffer["sample_logprob"]).abs()
    if not torch.isfinite(result).all() or error.max() > .25:
        raise ValueError(f"sampling/recomputed log probabilities differ: {float(error.max()):.6f}")
    return result, float(error.max())


def update(policy, head, critic, actor_optimizer, critic_optimizer, buffer, batch_size, microbatch):
    old, error = old_logprobs(policy, head, buffer, microbatch)
    advantage = buffer["advantages"]
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-8)
    parameters = [p for p in policy.parameters() if p.requires_grad] + list(head.parameters())
    metrics, stop = [], False
    actor_updates = 0
    gradient_check = None
    count = len(old)
    for epoch in range(2):
        order = torch.randperm(count)
        for begin in tqdm(range(0, count, batch_size), desc=f"PPO {epoch + 1}/2", mininterval=5, file=sys.stdout):
            group = order[begin:begin + batch_size]
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            group_kl = group_clip = group_value = group_policy = group_std = 0.
            for offset in range(0, len(group), microbatch):
                idx = group[offset:offset + microbatch]
                lp, entropy, std = log_probability(policy, head, *batch_from(buffer, idx))
                difference = lp - old[idx].cuda()
                if not torch.isfinite(difference).all():
                    raise ValueError("nonfinite PPO likelihood ratio")
                if difference.abs().max() > 60:
                    stop = True
                    break
                ratio = difference.exp()
                a = advantage[idx].cuda()
                ploss = torch.maximum(-a * ratio, -a * ratio.clamp(.8, 1.2)).mean()
                vloss = .5 * (critic(buffer["critic"][idx].cuda()) - buffer["returns"][idx].cuda()).square().mean()
                weight = len(idx) / len(group)
                ((ploss - .001 * entropy.mean()) * weight).backward()
                (vloss * weight).backward()
                group_kl += float(((ratio - 1) - difference).mean().detach()) * weight
                group_clip += float(((ratio - 1).abs() > .2).float().mean()) * weight
                group_value += float(vloss.detach()) * weight
                group_policy += float(ploss.detach()) * weight
                group_std += float(std.detach()) * weight
            if stop or group_kl > .03:
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                stop = True
                metrics.append(dict(kl=group_kl, early_stop=True))
                break
            if gradient_check is None:
                fm_gradient = float(sum(p.grad.float().abs().sum() for p in policy.parameters() if p.grad is not None))
                noise_gradient = float(sum(p.grad.float().abs().sum() for p in head.parameters() if p.grad is not None))
                if fm_gradient <= 0 or noise_gradient <= 0 or any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                    raise ValueError("FM/noise gradient missing or frozen gradient present")
                gradient_check = dict(fm_abs_sum=fm_gradient, noise_abs_sum=noise_gradient, frozen_gradients=False)
            torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_updates += 1
            metrics.append(dict(kl=group_kl, clip_fraction=group_clip, value_loss=group_value,
                                policy_loss=group_policy, std=group_std))
        if stop:
            break
    # A policy KL stop must not starve the privileged value function.
    for _ in range(4):
        order = torch.randperm(count)
        for begin in range(0, count, batch_size):
            idx = order[begin:begin + batch_size]
            critic_optimizer.zero_grad(set_to_none=True)
            vloss = .5 * (critic(buffer["critic"][idx].cuda()) - buffer["returns"][idx].cuda()).square().mean()
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            critic_optimizer.step()
    return dict(sample_logprob_max_error=error, actor_updates=actor_updates, early_stop=stop,
                gradient_check=gradient_check, minibatches=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="source BC prior, also required for resume")
    parser.add_argument("--resume", type=Path, help="RL optimizer/state checkpoint; output must be fresh; envs reset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--lr", type=float,
                        help="FM LR; defaults to 3e-8 for a new run, or the saved LR when resuming")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    initial_lr = 3e-8 if args.lr is None else args.lr
    if not 1 <= args.envs <= 8 or args.batch_size % args.microbatch or args.eval_cases > 48:
        raise ValueError("at most eight envs; batch divisible by microbatch; at most 48 eval cases")
    if min(args.iterations, args.rollout_steps, args.batch_size, args.microbatch, args.eval_every, initial_lr) <= 0 or args.eval_cases < 0:
        raise ValueError("positive train settings and nonnegative eval case count required")
    if args.output.exists():
        raise FileExistsError(args.output)
    os.sched_setaffinity(0, set(range(8)))
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    policy, source = load_prior(args.checkpoint)
    head = NoiseHead().cuda()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    actor_optimizer = torch.optim.AdamW([dict(params=trainable, lr=initial_lr),
                                        dict(params=head.parameters(), lr=1e-6)], weight_decay=0.)
    digest = frozen_digest(policy)
    resumed = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True) if args.resume else None
    if resumed:
        if resumed["config"]["rl"]["frozen_digest"] != digest:
            raise ValueError("resume frozen stack differs from source prior")
        policy.load_state_dict(resumed["model"], strict=True)
        head.load_state_dict(resumed["noise"], strict=True)
        restore_actor_optimizer(actor_optimizer, resumed["actor_optimizer"], args.lr)
    args.lr = actor_optimizer.param_groups[0]["lr"]
    args.output.mkdir(parents=True)
    config = dict(vars(args), checkpoint=str(args.checkpoint), output=str(args.output),
                  resume=str(args.resume) if args.resume else None,
                  method="ReinFlow/PPO; joint Gaussian denoising path likelihood, no logprob normalization",
                  source_step=source["step"], source_epoch=source["epoch"],
                  trainable_policy=[name for name, p in policy.named_parameters() if p.requires_grad],
                  trainable_parameters=sum(p.numel() for p in trainable), frozen_digest=digest,
                  frozen="ViT, visual/proprio adapters, task/target/view embeddings, state and context projections, context decoder",
                  auxiliary="privileged critic + bounded learned Gaussian transition noise head",
                  reset="100% covered setup_scene, no context/shuffle, balanced 24 queries x two slot orders",
                  reset_randomization="same position/yaw/joint ranges and physics as prior",
                  horizon=25, execute_chunk=EXECUTE, flow_steps=FLOW_STEPS, waypoint_hz=25, servo_hz=50,
                  temporal_ensemble=False, max_frames=MAX_FRAMES, gamma_per_waypoint=GAMMA,
                  lambda_per_waypoint=LAMBDA, reward_scale=.01,
                  reward=dict(cover_deposit_once=10, success=100, wrong_lift=-100, workspace_failure=-100,
                              time_per_waypoint=-.01, potential="gamma*Phi(next)-Phi(current), terminal Phi=0"),
                  actor_precision="float32 FM likelihood/velocity; frozen image/context encoding bf16",
                  ppo_epochs=2, clip=.2, target_kl=.02, kl_stop=.03, critic_lr=3e-4,
                  noise_lr=1e-6, noise_std_bounds=[.005, .08], noise_std_initial=.02,
                  entropy_coefficient=.001, cache="CPU frozen context K/V + pooled context; no RGB buffer",
                  cache_dtype="bfloat16 K/V, lossless relative to frozen bf16 encoding", cache_preallocated=True,
                  status="starting", evaluation="48 fresh fixed scenes: task12 x target2 x slot order2, both covers closed")
    config["lr"] = actor_optimizer.param_groups[0]["lr"]
    config["noise_lr"] = actor_optimizer.param_groups[1]["lr"]
    if resumed:
        config.update(resume_iteration=resumed["epoch"], resume_step=resumed["step"],
                      saved_fm_lr=resumed["actor_optimizer"]["param_groups"][0]["lr"])
    write_json(args.output / "config.json", config)
    readme = (f"# {args.output.name}\n\nOnline ReinFlow/PPO from {args.checkpoint}, BC epoch {source['epoch']}/step {source['step']}.\n"
              "Only FM velocity modules train; the full observation/conditioning stack is frozen. Critic/noise head are RL auxiliaries.\n"
              "RL uses fresh random covered resets without shuffle.\n"
              f"H25/K5, Euler10; 25Hz waypoints/50Hz linear interpolation, no TE, 1000-frame budget. {args.envs} CPU envs, GPU0.\n"
              f"{args.iterations} PPO iterations, {args.rollout_steps} macro steps/env/iteration; batch {args.batch_size}, microbatch {args.microbatch}, "
              f"2 PPO epochs, actor LR {args.lr}, critic LR3e-4, noise LR1e-6. Reward/discount/contact evaluated at each 25Hz waypoint.\n"
              f"Baseline and evaluation every {args.eval_every} iterations use {args.eval_cases} fixed held-out resets, with identical K5/no-TE control.\n"
              "Method: https://reinflow.github.io/ ; complete settings in [config.json](config.json).\n\nStatus: running.\n")
    if resumed:
        readme += (f"\nResume from {args.resume}, iteration{resumed['epoch']}/step{resumed['step']}; "
                   f"{args.iterations} additional iterations, ending at{resumed['epoch'] + args.iterations}. "
                   "Optimizer moments, auxiliary networks and RNG are restored; simulator episodes reset.\n")
    (args.output / "README.md").write_text(readme)
    print(json.dumps(dict(event="initialized", trainable_parameters=config["trainable_parameters"],
                          frozen_digest=digest, source=str(args.checkpoint))), flush=True)
    vector = VectorGrasp(args.envs)
    query_queue = resumed["query_queue"] if resumed else []
    episode_index = resumed["episode_index"] if resumed else 0
    if resumed:
        rng.bit_generator.state = resumed["numpy_rng"]

    def reset_next():
        nonlocal query_queue, episode_index
        if not query_queue:
            query_queue = balanced_queries(rng)
        task, target, order = query_queue.pop()
        item = dict(seed=1000000 + episode_index, task_id=task, target_id=target, slot_order=order)
        episode_index += 1
        return item

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    history, total_frames = [], resumed["step"] if resumed else 0
    start_iteration = resumed["epoch"] if resumed else 0
    started = time.perf_counter()
    complete = False
    try:
        write_json(args.output / "progress.json", dict(stage="baseline", iteration=start_iteration))
        baseline = evaluate(policy, vector, args.output, start_iteration, args.eval_cases)
        best_success = baseline["success"]
        for i in range(args.envs):
            vector.send(i, "reset", reset_next())
        packets = [vector.receive(i) for i in range(args.envs)]
        critic = Critic(len(packets[0]["critic"])).cuda()
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=3e-4, weight_decay=0.)
        if resumed:
            critic.load_state_dict(resumed["critic"], strict=True)
            critic_optimizer.load_state_dict(resumed["critic_optimizer"])
            torch.set_rng_state(resumed["torch_rng_state"])
            torch.cuda.set_rng_state(resumed["cuda_rng_state"])
            generator.set_state(resumed["sampling_rng"])
        config["critic_dimension"] = len(packets[0]["critic"])
        config["status"] = "training"
        write_json(args.output / "config.json", config)
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            write_json(args.output / "progress.json", dict(stage="collection", iteration=iteration, total_waypoints=total_frames))
            iteration_started = time.perf_counter()
            buffer, packets, completed, frames = collect(policy, head, critic, vector, packets, reset_next,
                                                          generator, args.rollout_steps, args.output)
            collection_time = time.perf_counter() - iteration_started
            write_json(args.output / "progress.json", dict(stage="PPO", iteration=iteration, total_waypoints=total_frames + frames))
            metrics = update(policy, head, critic, actor_optimizer, critic_optimizer, buffer, args.batch_size, args.microbatch)
            total_frames += frames
            del buffer
            record = dict(iteration=iteration, total_waypoints=total_frames, collected_waypoints=frames,
                          collection_s=collection_time, iteration_s=time.perf_counter() - iteration_started,
                          elapsed_s=time.perf_counter() - started, episodes=len(completed),
                          successes=sum(t["success"] for t in completed),
                          wrong=sum(t["wrong_object"] is not None for t in completed),
                          peak_vram_gib=torch.cuda.max_memory_reserved() / 2**30, **metrics)
            history.append(record)
            write_json(args.output / "history.json", history)
            print(json.dumps(dict(event="iteration", **record)), flush=True)
            # A recoverable checkpoint is available after every iteration.
            saved = dict(architecture="oracle_task_target_fm", model=policy.state_dict(),
                         noise=head.state_dict(), critic=critic.state_dict(),
                         actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                         step=total_frames, epoch=iteration, config=dict(source["config"], rl=config),
                         torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
                         sampling_rng=generator.get_state(), numpy_rng=rng.bit_generator.state,
                         episode_index=episode_index, query_queue=query_queue)
            torch.save(saved, args.output / "latest.tmp")
            (args.output / "latest.tmp").replace(args.output / "latest.pt")
            if iteration % args.eval_every == 0 or iteration == start_iteration + args.iterations:
                if frozen_digest(policy) != digest:
                    raise ValueError("frozen policy parameters changed")
                result = evaluate(policy, vector, args.output, iteration, args.eval_cases)
                if result["success"] > best_success:
                    best_success = result["success"]
                    torch.save(saved, args.output / "best.tmp")
                    (args.output / "best.tmp").replace(args.output / "best.pt")
                for i in range(args.envs):
                    vector.send(i, "reset", reset_next())
                packets = [vector.receive(i) for i in range(args.envs)]
        (args.output / "final.pt").hardlink_to(args.output / "latest.pt")
        config["status"] = "complete"
        complete = True
        write_json(args.output / "config.json", config)
        write_json(args.output / "progress.json", dict(stage="complete", iteration=iteration, total_waypoints=total_frames))
        (args.output / "README.md").write_text(readme.replace("Status: running.", "Status: complete.") +
            f"\nCompleted {total_frames} waypoints. Best online evaluation success {best_success}/{args.eval_cases}; "
            "the source BC prior remains the baseline if no RL checkpoint improves it. See history.json and eval_*.json.\n")
    finally:
        if not complete:
            config["status"] = "incomplete"
            write_json(args.output / "config.json", config)
            (args.output / "README.md").write_text(readme.replace("Status: running.", "Status: incomplete; inspect the run log and latest.pt."))
        vector.close()


if __name__ == "__main__":
    main()
