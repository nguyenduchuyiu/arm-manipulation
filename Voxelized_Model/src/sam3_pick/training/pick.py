"""E131: full-data BC of GT-visible target-cloud encoder and pretrained reach FM."""

from sam3_pick.paths import PICK_DATA as DATA
from sam3_pick.utils.hashing import source_hashes
from sam3_pick.paths import PROJECT, OUTPUTS
import argparse
import json
import math
import shutil
import time
import av
import numpy as np
import torch
from tqdm import tqdm
from sam3_pick.models.pointcloud import PointCloudReach
from sam3_pick.data.pick_labels import action_labels, write
from sam3_pick.utils.hashing import sha256_file as sha
from sam3_pick.data.target_cloud import ROOT, CACHE, SOURCE, extract_cloud

UPDATES = 23520
BATCH = 128
NOTE = OUTPUTS / "notes/e131_README.md"


def fresh():
    torch.manual_seed(102)
    model = PointCloudReach().cuda()
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False, mmap=True)
    assert saved["step"] == 120000
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(True)
    return model


def optimizer(model):
    return torch.optim.AdamW(
        [
            dict(
                params=[p for n, p in model.named_parameters() if n.startswith("flow.")],
                lr=2e-05,
            ),
            dict(
                params=[p for n, p in model.named_parameters() if not n.startswith("flow.")],
                lr=0.0001,
            ),
        ],
        betas=(0.9, 0.95),
        weight_decay=0.0001,
        fused=True,
    )


def data_and_audit():
    before = time.perf_counter()
    cache_hash = sha(CACHE / "labels.npz")
    config = json.loads((CACHE / "cache_config.json").read_text())
    assert config["status"] == "complete" and config["source_manifest_sha256"] == sha(
        DATA / "manifest.jsonl"
    )
    assert config["source_checkpoint_sha256"] == sha(SOURCE)
    with np.load(CACHE / "labels.npz") as z:
        labels = {k: z[k] for k in z.files}
    assert len(labels["cloud"]) == 296688 and int(labels["is_train"].sum()) == 294640
    assert labels["cloud_valid"].all() and np.isfinite(labels["cloud"]).all()
    with np.load(CACHE / "train/indices.npz") as z:
        train_ids = z["train"].copy()
        val_ids = z["validation"].copy()
    np.testing.assert_array_equal(np.sort(train_ids), np.flatnonzero(labels["is_train"]))
    assert len(val_ids) == 512 and (not labels["is_train"][val_ids].any())
    manifests = {}
    for split, n in [("train", 2048), ("val", 256), ("test", 1408)]:
        rows = list(map(json.loads, (CACHE / f"{split}_manifest.jsonl").read_text().splitlines()))
        assert len(rows) == n
        manifests[split] = rows
        shutil.copyfile(CACHE / f"{split}_manifest.jsonl", ROOT / f"{split}_manifest.jsonl")
    sets = [{r["path"] for r in manifests[k]} for k in ("train", "val", "test")]
    assert all((not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)))
    assert {r["layout_id"] for r in manifests["train"]} == set(range(32))
    jobs = json.loads((CACHE / "cache_jobs.json").read_text())
    lookup = {r["path"]: r for rows in manifests.values() for r in rows}
    checks = []
    chosen = [
        next(
            (
                j
                for j in jobs
                if lookup[j["path"]]["split"] == split and lookup[j["path"]]["target_id"] == t
            )
        )
        for split in ("train", "val")
        for t in range(4)
    ]
    for job in tqdm(chosen, desc="Verify existing GT-cloud cache", mininterval=5):
        row = lookup[job["path"]]
        starts = np.array(job["starts"])
        indices = job["offset"] + np.arange(len(starts))
        expected = action_labels(row, starts)
        for key in ("state", "action", "mask", "target", "raw_frame", "is_train"):
            np.testing.assert_array_equal(labels[key][indices], expected[key])
        directory = DATA / row["path"]
        with np.load(directory / "masks.npz") as z:
            masks = z["instance"][:, 0] == row["target_id"] + 1
        with np.load(directory / "calibration.npz") as z:
            K = z["intrinsics"][0]
            Ts = z["camera_to_base"][:, 0]
            unit = float(z["depth_unit_m"])
        rng = np.random.default_rng(
            np.random.SeedSequence([119, row["layout_id"], row["permutation"], row["target_id"]])
        )
        wanted = {int(f): int(i) for f, i in zip(starts, indices)}
        count = 0
        with av.open(str(directory / "depth_front.mkv")) as stream:
            stream.streams.video[0].thread_count = 1
            for frame, item in enumerate(stream.decode(video=0)):
                if frame not in wanted:
                    continue
                i = wanted[frame]
                cloud, good, pixels = extract_cloud(
                    item.to_ndarray(format="gray16le"),
                    masks[frame],
                    K,
                    Ts[frame],
                    rng,
                    unit,
                )
                assert good and pixels == int(labels["visible_pixels"][i])
                np.testing.assert_array_equal(cloud, labels["cloud"][i])
                count += 1
        assert count == len(starts)
        checks.append(
            dict(
                path=row["path"],
                frames=count,
                clouds_bit_exact=True,
                action_labels_bit_exact=True,
            )
        )
    result = dict(
        status="passed",
        cache=str(CACHE / "labels.npz"),
        cache_sha256=cache_hash,
        train_frames=len(train_ids),
        validation_frames=2048,
        validation_scored=512,
        all_clouds_valid=True,
        minimum_filtered_target_pixels=int(labels["visible_pixels"].min()),
        point_cloud_bytes=labels["cloud"].nbytes,
        checks=checks,
        GT_pose_not_policy_input=True,
        elapsed_s=time.perf_counter() - before,
    )
    write(ROOT / "data_verification.json", result)
    shutil.copyfile(CACHE / "partitions.json", ROOT / "partitions.json")
    tensors = {k: torch.from_numpy(labels[k]).cuda() for k in ("cloud", "state", "action", "mask")}
    result["GPU_cache_GiB"] = sum((v.numel() * v.element_size() for v in tensors.values())) / 2**30
    write(ROOT / "data_verification.json", result)
    return (tensors, train_ids, val_ids, result)


def batch(model, cache, ids):
    ix = torch.as_tensor(ids, device="cuda", dtype=torch.long)
    return model.loss(*(cache[k][ix] for k in ("cloud", "state", "action", "mask")))


def update(model, opt, cache, ids, step):
    for group, lr in zip(opt.param_groups, (2e-05, 0.0001)):
        group["lr"] = min(step / 100, 1.0) * lr
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = batch(model, cache, ids)
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    return (float(loss.detach()), float(norm))


@torch.inference_mode()
def validate(model, cache, ids):
    model.eval()
    sums = []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(102000)
        for start in range(0, len(ids), 64):
            current = ids[start : start + 64]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = batch(model, cache, current)
            sums.append((float(loss), len(current)))
    return sum((a * b for a, b in sums)) / sum((b for _, b in sums))


def main(resume=False):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    ROOT.mkdir(exist_ok=resume)
    if not resume:
        shutil.copyfile(NOTE, ROOT / "README.md")
    cache, train_ids, val_ids, data_info = data_and_audit()
    model = fresh()
    opt = optimizer(model)
    buffers = {n: v.clone() for n, v in model.named_buffers()}
    out = ROOT / "train"
    out.mkdir(exist_ok=resume)
    if not resume:
        shutil.copyfile(NOTE, out / "README.md")
    rng = np.random.default_rng(102)
    order = rng.permutation(train_ids)
    cursor = seen = first = 0
    best = math.inf
    best_step = 0
    config = dict(
        status="running",
        experiment=ROOT.name,
        dataset=str(DATA),
        source_checkpoint=str(SOURCE),
        source_checkpoint_sha256=sha(SOURCE),
        source_step=120000,
        initialization="Entire PointCloudReach from E118best120000,including point encoder,goal projection,FM and state encoder;fresh optimizer",
        inputs="128 visible target XYZ in robot base + measured five joints and jaw;no RGB/taskID embedding/GT center/object pose",
        task_selection="taskID selects GT visible target mask outside policy",
        architecture="Unchanged E118 PointCloudReach:point MLP3/64/128/256,maxpool,Linear256/960;goal+state context to FM",
        trainable="all105102634 parameters:point encoder,goal projection,entire FM/state encoder",
        frozen="none",
        loss="masked flow-matching velocity MSE on expert H25 EE-delta actions only;no auxiliary loss",
        batch=BATCH,
        total_steps=UPDATES,
        requested_updates=UPDATES,
        steps_per_epoch=math.ceil(len(train_ids) / BATCH),
        train_chunks=len(train_ids),
        train_episodes=2048,
        val_episodes=256,
        test_episodes=1408,
        train_stride=1,
        fm_lr=2e-05,
        point_lr=0.0001,
        warmup_steps=100,
        optimizer="fusedAdamW beta(.9,.95),wd1e-4,clip1",
        precision="BF16",
        seed=102,
        normalization="All global XYZ,state,action mean/std buffers from E118 remain unchanged;no per-cloud centering",
        horizon=25,
        action_dim=10,
        action="anchor EE deltaXYZ + relative rotation6D + absolute commanded jaw",
        validation="same512fixed E122validation chunk IDs and seed102000 noise/time;best by FM loss only",
        source_labels=str(CACHE / "labels.npz"),
        source_labels_sha256=data_info["cache_sha256"],
        GPU_cache_GiB=data_info["GPU_cache_GiB"],
        missing_cloud="No missing train/val inputs (all filtered target clouds>=94pixels);online<8pixels fails explicitly",
        planned_rollout="final23520;all1408heldout,H25/K15/Euler10/seed0/300frames;then16existing XY boundary pairs;no videos",
        code_sha256=source_hashes(),
        reach_model_sha256=sha(PROJECT / "src/sam3_pick/models/pointcloud.py"),
        reach_flow_sha256=sha(PROJECT / "src/sam3_pick/models/flow.py"),
    )
    if resume:
        saved = torch.load(out / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
        assert saved["config"]["source_labels_sha256"] == data_info["cache_sha256"]
        model.load_state_dict(saved["model"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        order = saved["order"]
        cursor = saved["cursor"]
        seen = saved["samples_seen"]
        first = saved["step"]
        best = saved["best_val"]
        best_step = saved["config"]["best_step"]
        rng.bit_generator.state = saved["rng"]["numpy"]
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state(saved["rng"]["cuda"])
        del saved
    else:
        config["initial_validation_FM_loss"] = validate(model, cache, val_ids)
    write(out / "config.json", config)
    np.savez(out / "indices.npz", train=train_ids, validation=val_ids)
    started = time.perf_counter()
    history = []
    model.train()
    torch.cuda.reset_peak_memory_stats()

    def save(name, step, resumable):
        config.update(
            samples_seen=seen,
            effective_epochs=seen / len(train_ids),
            best_step=best_step,
        )
        payload = dict(model=model.state_dict(), step=step, config=dict(config), best_val=best)
        if resumable:
            payload.update(
                optimizer=opt.state_dict(),
                order=order,
                cursor=cursor,
                samples_seen=seen,
                rng=dict(
                    torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(),
                    numpy=rng.bit_generator.state,
                ),
            )
        temp = out / (name + ".tmp")
        torch.save(payload, temp)
        temp.replace(out / (name + ".pt"))

    for step in tqdm(
        range(first + 1, UPDATES + 1),
        desc="E131 GT-cloud-only BC",
        initial=first,
        total=UPDATES,
        mininterval=5,
    ):
        if cursor >= len(order):
            order = rng.permutation(train_ids)
            cursor = 0
        ids = order[cursor : cursor + BATCH]
        cursor += len(ids)
        seen += len(ids)
        loss, norm = update(model, opt, cache, ids, step)
        row = dict(
            step=step,
            epoch=seen / len(train_ids),
            BC_loss=loss,
            grad_norm=norm,
            elapsed_s=time.perf_counter() - started,
        )
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        history.append(loss)
        if step == 1 or step % 50 == 0:
            elapsed = time.perf_counter() - started
            metric = dict(
                status="training",
                **row,
                mean_BC_loss=float(np.mean(history)),
                seconds_per_update=elapsed / (step - first),
                estimated_remaining_minutes=(UPDATES - step) * elapsed / max(1, step - first) / 60,
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            )
            write(out / "progress.json", metric)
            print(json.dumps(metric), flush=True)
            history = []
        if step == 1 or step % 250 == 0 or step == UPDATES:
            score = validate(model, cache, val_ids)
            if score < best:
                best = score
                best_step = step
                save("best", step, False)
            save("latest", step, True)
            val = dict(
                step=step,
                epoch=seen / len(train_ids),
                FM_loss=score,
                best_step=best_step,
            )
            with (out / "validation.jsonl").open("a") as f:
                f.write(json.dumps(val) + "\n")
            print(json.dumps(val), flush=True)
            model.train()
            if step == 1:
                assert all((float(v["step"]) == 1 for v in opt.state.values()))
                write(
                    ROOT / "launch_verification.json",
                    dict(
                        status="passed",
                        step=1,
                        all_optimizer_counters=1,
                        source_step=120000,
                        BC_only=True,
                        all_parameters_trainable=all((p.requires_grad for p in model.parameters())),
                        parameters=sum((p.numel() for p in model.parameters())),
                        GPU_cache_GiB=data_info["GPU_cache_GiB"],
                    ),
                )
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    epochs, tail = divmod(UPDATES, config["steps_per_epoch"])
    assert seen == epochs * len(train_ids) + tail * BATCH
    config.update(
        status="complete",
        samples_seen=seen,
        effective_epochs=seen / len(train_ids),
        best_step=best_step,
        best_validation_FM_loss=best,
        elapsed_s=time.perf_counter() - started,
    )
    write(out / "config.json", config)
    write(out / "progress.json", config)
    final = out / "final.pt"
    final.hardlink_to(out / "latest.pt")
    write(
        out / "verification.json",
        dict(
            status="passed",
            normalization_unchanged=True,
            final_step=UPDATES,
            samples_seen=seen,
            effective_epochs=seen / len(train_ids),
            BC_only=True,
        ),
    )
    text = (
        NOTE.read_text().replace("Status: PLANNED.", "Status: TRAIN COMPLETE; rollout pending.")
        + "\nTraining:\n"
        + json.dumps(config, indent=2)
        + "\n"
    )
    NOTE.write_text(text)
    (ROOT / "README.md").write_text(text)
    (out / "README.md").write_text(text)
    print(json.dumps(config), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args().resume)
