"""Fine-tune TurboVLA directly on NexArm HDF5 demonstrations."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import get_worker_info

from ..data.libero_rlds import LiberoRLDSDataset as _BaseDataset
from . import trainer


class NexArmDataset(_BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.chunk_size != 12:
            raise ValueError("NexArm data uses 12-step action chunks")
        root = Path(kwargs.get("dataset_dir", args[0] if args else ""))
        episodes = []
        states, actions = [], []
        for path in sorted(root.glob("*.hdf5")):
            with h5py.File(path) as file:
                instruction = json.loads(file["data"].attrs["problem_info"])["language_instruction"]
                for index, name in enumerate(sorted(file["data"], key=lambda value: int(value[5:]))):
                    if (self.split == "train") != (index % 10 != 0):
                        continue
                    demo = file[f"data/{name}"]
                    episodes.append((path, name, instruction))
                    states.append(demo["obs/joint_states"][()])
                    mask = demo["action_chunk_mask"][:, 0].astype(bool)
                    actions.append(demo["actions"][:, 0][mask])
        state, action = np.concatenate(states), np.concatenate(actions)
        self.proprio_mean = torch.tensor(state.mean(0), dtype=torch.float32)
        self.proprio_std = torch.tensor(state.std(0), dtype=torch.float32)
        self.action_min = torch.tensor(action.min(0), dtype=torch.float32)
        self.action_max = torch.tensor(action.max(0), dtype=torch.float32)
        self.episodes = episodes

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        episodes = self.episodes[self.rank :: self.world_size][worker_id::workers]
        rng = np.random.default_rng(self.seed + self.rank * 1009 + worker_id * 9176)
        while True:
            rng.shuffle(episodes)
            for path, name, instruction in episodes:
                with h5py.File(path) as file:
                    demo = file[f"data/{name}"]
                    indices = np.arange(len(demo["actions"]))
                    if self.shuffle_steps_within_episode:
                        rng.shuffle(indices)
                    for step in indices:
                        images = (
                            self._process_image_pair(demo["obs/front_rgb"][step]),
                            self._process_image_pair(demo["obs/eye_in_hand_rgb"][step]),
                        )
                        state = self._normalize_state(torch.tensor(demo["obs/joint_states"][step], dtype=torch.float32))
                        action = self._normalize_action_chunk(torch.tensor(demo["actions"][step], dtype=torch.float32))
                        mask = torch.tensor(demo["action_chunk_mask"][step], dtype=torch.float32)
                        yield images, instruction, state, action, mask


_parse_args = trainer.parse_args


def parse_args():
    args = _parse_args()
    args.action_dim, args.state_dim, args.chunk_size = 6, 6, 12
    if args.model_init_ckpt:
        args.require_feature_enhancer_preload = False
        args.load_text_projection_from_init = False
        args.require_text_proj_preload = False
    return args


trainer.parse_args = parse_args
trainer.LiberoRLDSDataset = NexArmDataset


if __name__ == "__main__":
    trainer.train_model()
