"""Correlated actuator noise for physical expert/perturbation rollouts."""
import numpy as np


class RolloutFailure(RuntimeError):
    """A physical rollout could not complete the task within its retry budget."""


class ContinuousPerturbation:
    def __init__(self, config, limits):
        self.config = dict(config)
        self.limits = limits
        self.rng = np.random.default_rng(config["seed"])
        self.std = np.deg2rad(config["std_deg"])
        if self.std.shape != (5,) or not np.isfinite(self.std).all() or np.any(self.std < 0):
            raise ValueError("noise std must be five finite nonnegative degrees")
        if config["tau_s"] <= 0 or not 0 < config["retry_scale"] <= 1 or config["max_attempts"] < 1:
            raise ValueError("invalid noise/retry settings")
        if config["labels"] not in ("expert", "executed"):
            raise ValueError("labels must be expert or executed")
        self.rho = np.exp(-.04 / config["tau_s"])
        self.noise = np.zeros(5)
        self.scale = 1.0
        self.trace = []
        self.attempts = []

    def apply(self, stage, command, frame):
        self.noise = np.clip(self.rho * self.noise + np.sqrt(1 - self.rho**2) *
                             self.std * self.rng.normal(size=5), -2 * self.std, 2 * self.std)
        executed = command.copy()
        executed[:5] = np.clip(command[:5] + self.scale * self.noise,
                               self.limits[:5, 0], self.limits[:5, 1])
        self.trace.append({"frame": frame, "phase": stage, "expert_ctrl": command.copy(),
                           "executed_ctrl": executed.copy(), "scale": self.scale})
        return executed
