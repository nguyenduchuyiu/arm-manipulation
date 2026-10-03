"""Memory-occlusion evaluation metrics over episode-level predictions."""
import numpy as np


def cover_selection_accuracy(predicted, correct) -> float:
    predicted, correct = np.asarray(predicted), np.asarray(correct)
    if predicted.shape != correct.shape or predicted.size == 0:
        raise ValueError("predicted and correct must have the same nonempty shape")
    return float(np.mean(predicted == correct))


def full_task_success(predicted, correct, cover_removed, target_grasped, target_placed) -> float:
    predicted, correct = np.asarray(predicted), np.asarray(correct)
    removed, grasped = np.asarray(cover_removed), np.asarray(target_grasped)
    placed = np.asarray(target_placed)
    if any(value.shape != predicted.shape for value in (correct, removed, grasped, placed)) or predicted.size == 0:
        raise ValueError("all metric inputs must have the same nonempty shape")
    return float(np.mean((predicted == correct) & removed & grasped & placed))
