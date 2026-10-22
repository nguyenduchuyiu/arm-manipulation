"""Format the running trainer's JSONL stream without restarting training."""
import json
import sys


def main():
    last_step = None
    last_phase = None
    for line in sys.stdin:
        record = json.loads(line)
        step, phase = record['step'], record['phase']
        if last_step is not None and step - last_step < 10 and phase == last_phase:
            continue
        print(f"step {step:6d} | {phase:12s} | epoch {record['train_passes']:.4f} | "
              f"loss {record['flow'] + record['xyz']:.5f} | FM {record['flow']:.5f} | "
              f"XYZ {record['xyz']:.5f} | err {record['current_xyz_mm']:.1f} mm", flush=True)
        last_step, last_phase = step, phase


if __name__ == '__main__':
    main()
