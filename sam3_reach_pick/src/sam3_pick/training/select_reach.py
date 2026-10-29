"""Select the fixed 120k reach weights for pick transfer, preserving validation-best."""

import torch
from sam3_pick.data.reach_clouds import ROOT


def main():
    saved = torch.load(ROOT / "train/final.pt", map_location="cpu", weights_only=False, mmap=True)
    assert saved["step"] == 120000
    output = ROOT / "train/transfer120000.pt"
    assert not output.exists(), "Transfer weights already exist"
    torch.save(dict(model=saved["model"], step=120000, config=saved["config"]), output)
    print(
        "Selected fixed120k reach transfer; validation-best checkpoint unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
