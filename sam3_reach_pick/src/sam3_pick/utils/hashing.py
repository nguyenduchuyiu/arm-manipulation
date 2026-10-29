"""Content hashes for source, inputs and learned parameters."""

import hashlib


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    from sam3_pick.paths import PROJECT

    root = PROJECT / "src"
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.py"))}
