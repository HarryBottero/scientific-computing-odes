from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable
from typing import SupportsFloat

import numpy as np


# Data classes for results and benchmarking
@dataclass
class SolverResult:
    method: str
    t: np.ndarray
    y: np.ndarray
    h: np.ndarray | None = None
    nfev: int = 0
    accepted_steps: int = 0
    rejected_steps: int = 0
    runtime_sec: float = 0.0

@dataclass
class BenchmarkRecord:
    method: str
    setting_name: str
    setting_value: float
    final_error: float
    max_path_error: float
    rms_path_error: float
    runtime_sec: float
    nfev: int
    accepted_steps: int
    rejected_steps: int
    invariant_drift: float | None = None


# ensure state is 1D array
def as_1d_state(y0: Iterable[float]) -> np.ndarray:
    arr = np.asarray(y0, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"State must be 1D; got shape {arr.shape}.")
    return arr

# clip step size to bounds
def clip_step(h: SupportsFloat, hmin: SupportsFloat, hmax: SupportsFloat) -> float:
    h_f = float(h)
    hmin_f = float(hmin)
    hmax_f = float(hmax)
    return max(hmin_f, min(hmax_f, h_f))


# simple timer context manager
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
        return False