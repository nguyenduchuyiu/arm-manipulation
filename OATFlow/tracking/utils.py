"""XYZ metrics and weights-only checkpoint storage."""
import json
import numpy as np
import torch

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summary(prediction, truth, hidden, phases):
    error = np.linalg.norm(prediction - truth, axis=-1) * 1000
    def score(mask):
        e = error[mask]
        if not len(e): return dict(count=0)
        return dict(count=len(e), mean_mm=float(e.mean()), median_mm=float(np.median(e)),
                    p90_mm=float(np.quantile(e, .9)),
                    within_mm={str(mm): float((e <= mm).mean()) for mm in (1, 2, 5, 10, 20, 50)})
    return dict(all=score(np.ones(len(error), bool)), hidden=score(hidden), visible=score(~hidden),
                by_phase={phase: score(phases == phase) for phase in sorted(set(phases))},
                mae_xyz_mm=(np.abs(prediction - truth).mean(0) * 1000).tolist())


def save(path, model, optimizer, step, config, best):
    payload = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   model_config=model.config, step=step, config=config, best_val_mm=best)
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state())
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)
