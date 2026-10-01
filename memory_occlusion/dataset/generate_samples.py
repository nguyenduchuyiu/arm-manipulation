"""Render a few complete expert demonstrations before considering a bulk run."""
import argparse
from pathlib import Path

from memory_occlusion.dataset.episode import generate
from memory_occlusion.dataset.references import create_references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swaps", type=int, default=1)
    parser.add_argument("--targets", nargs="+", default=["Milk", "Butter", "Popcorn", "Tuna"])
    parser.add_argument("--output", type=Path, default=Path("outputs/memory_occlusion/samples"))
    args = parser.parse_args()
    create_references(args.output)
    for target in args.targets:
        result = generate(args.output, args.seed, target, args.swaps)
        print(target, result["decision_times_s"], "success", flush=True)


if __name__ == "__main__":
    main()
