"""Timestamped temporal plans and a single background inference worker."""
from threading import Condition, Thread
import time

import numpy as np


class TemporalPlans:
    def __init__(self, coefficient, horizon):
        self.coefficient = coefficient
        self.horizon = horizon
        self.plans = []

    def add(self, source_step, actions, current_step):
        actions = np.asarray(actions)
        if actions.shape != (self.horizon, 6) or not np.isfinite(actions).all():
            raise ValueError("expected a finite Hx6 plan")
        if source_step > current_step:
            raise ValueError("plan observation is in the future")
        self.plans = [(step, plan) for step, plan in self.plans
                      if step + self.horizon > current_step]
        if source_step + self.horizon > current_step:
            self.plans.append((source_step, actions.copy()))
            self.plans.sort(key=lambda item: item[0])

    def block(self, start_step, count, previous):
        """Commit next endpoints; action[j] ends at (source_step+j+1)/25s."""
        self.plans = [(step, plan) for step, plan in self.plans
                      if step + self.horizon > start_step]
        actions, raw, counts = [], [], []
        for step in range(start_step, start_step + count):
            candidates = [plan[step - source] for source, plan in self.plans
                          if source <= step < source + self.horizon]
            if candidates:
                weights = np.exp(-self.coefficient * np.arange(len(candidates)))
                previous = np.average(candidates, axis=0, weights=weights)
            # An underrun explicitly holds the last committed waypoint.
            actions.append(np.asarray(previous).copy())
            raw.append(candidates[-1].copy() if candidates else np.asarray(previous).copy())
            counts.append(len(candidates))
        return np.asarray(actions), np.asarray(raw), np.asarray(counts)


class AsyncPlanner:
    """Predict runs only on this worker; the caller publishes immutable snapshots."""
    def __init__(self, predict, cleanup=None):
        self.predict = predict
        self.cleanup = cleanup
        self.condition = Condition()
        self.latest = None
        self.completed = []
        self.error = None
        self.stopping = False
        self.worker = Thread(target=self._run, name="FM inference")
        self.worker.start()

    def publish(self, step, snapshot):
        with self.condition:
            self.latest = (step, snapshot)
            self.condition.notify_all()

    def take(self, wait=False):
        with self.condition:
            if wait:
                self.condition.wait_for(lambda: self.completed or self.error)
            if self.error is not None:
                raise self.error
            results, self.completed = self.completed, []
            return results

    def _run(self):
        last_step = -1
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or
                                            (self.latest is not None and self.latest[0] > last_step))
                    if self.stopping:
                        return
                    step, snapshot = self.latest
                started = time.perf_counter()
                actions = self.predict(snapshot)
                finished = time.perf_counter()
                with self.condition:
                    self.completed.append((step, actions, finished - started, finished))
                    self.condition.notify_all()
                last_step = step
        except BaseException as error:
            # Propagate inference failures to the simulator thread; never silently hold on error.
            with self.condition:
                self.error = error
                self.condition.notify_all()
        finally:
            if self.cleanup is not None:
                try:
                    self.cleanup()
                except BaseException as error:
                    with self.condition:
                        self.error = error
                        self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.worker.join()
        if self.error is not None:
            raise self.error
