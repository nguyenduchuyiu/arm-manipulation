"""Download a pinned SmolVLA checkpoint and export the TCOW policy weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm

from memory_occlusion.experiments.tcow_joint_flow.flow_matching import (
    SOURCE_REPO, SOURCE_REVISION, DenseFlowMatching,
)

SOURCE_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"


def export_key(key):
    expert = "model.vlm_with_expert.lm_expert."
    if key.startswith(expert):
        return key.removeprefix(expert)
    for module in ("state_proj", "action_in_proj", "action_out_proj",
                   "action_time_mlp_in", "action_time_mlp_out"):
        if key.startswith(f"model.{module}."):
            return key.removeprefix("model.")
    prefix = "model.vlm_with_expert.vlm.model.text_model.layers."
    if key.startswith(prefix):
        layer, field = key.removeprefix(prefix).split(".", 1)
        index = int(layer)
        if index < 16 and index % 2 == 1:
            fields = {"input_layernorm.weight": "norm.weight",
                      "self_attn.k_proj.weight": "k_proj.weight",
                      "self_attn.v_proj.weight": "v_proj.weight"}
            if field in fields:
                return f"context_projections.{index // 2}.{fields[field]}"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume-download", action="store_true")
    args = p.parse_args()
    if args.output.exists() and not args.resume_download:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume_download)
    if (args.output / "expert.safetensors").exists():
        raise FileExistsError(args.output / "expert.safetensors")
    url = f"https://huggingface.co/{SOURCE_REPO}/resolve/{SOURCE_REVISION}/model.safetensors"
    original = args.output / "source.safetensors"
    digest = hashlib.sha256()
    offset = original.stat().st_size if original.exists() else 0
    if offset:
        with original.open("rb") as stream:
            while block := stream.read(4 * 1024**2):
                digest.update(block)
    request = urllib.request.Request(url + f"?resume_offset={offset}",
                                     headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response, original.open("ab") as stream:
        if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise ValueError("download endpoint did not honor resume offset")
        total = offset + int(response.headers["Content-Length"])
        with tqdm(total=total, desc="download action expert", unit="B", unit_scale=True,
                  initial=offset, mininterval=5, file=sys.stdout) as progress:
            while block := response.read(4 * 1024**2):
                stream.write(block)
                digest.update(block)
                progress.update(len(block))
        if original.stat().st_size != total:
            raise ValueError("incomplete action expert download")
    if digest.hexdigest() != SOURCE_SHA256:
        raise ValueError("action expert source checksum differs from the pinned Hub checkpoint")
    state = {}
    with safe_open(str(original), framework="pt", device="cpu") as tensors:
        keys = [(key, export_key(key)) for key in tensors.keys()]
        for key, destination in tqdm(keys, desc="extract expert", unit="tensor", file=sys.stdout):
            if destination is not None:
                if destination in state:
                    raise ValueError(f"duplicate export tensor: {destination}")
                state[destination] = tensors.get_tensor(key).contiguous()
    exported = args.output / "expert.safetensors"
    save_file(state, str(exported), metadata={"repo": SOURCE_REPO, "revision": SOURCE_REVISION})
    model = DenseFlowMatching()
    report = model.load_pretrained(exported)
    report["source_sha256"] = digest.hexdigest()
    exported_digest = hashlib.sha256()
    with exported.open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            exported_digest.update(block)
    report["expert_sha256"] = exported_digest.hexdigest()
    report["export_bytes"] = exported.stat().st_size
    (args.output / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Retain the verified expert and source identity, rather than the unused VLM.
    original.unlink()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
