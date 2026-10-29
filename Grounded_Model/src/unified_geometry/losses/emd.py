import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def emd(prediction, target):
    distances = torch.cdist(
        prediction.float(), target.float(), compute_mode="donot_use_mm_for_euclid_dist"
    )
    indices = np.stack(
        [linear_sum_assignment(matrix)[1] for matrix in distances.detach().cpu().numpy()]
    )
    order = torch.as_tensor(indices, device=prediction.device)
    return distances.gather(2, order[..., None]).squeeze(-1).mean(-1)
