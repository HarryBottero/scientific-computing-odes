import numpy as np

from src.ftle import estimate_ftle



def linear_expansion_rhs(t, y, a=0.3):
    return np.array([a * y[0], 0.0])



def test_ftle_matches_known_linear_expansion_rate():
    est = estimate_ftle(
        linear_expansion_rhs,
        t_span=(0.0, 2.0),
        y0=np.array([1.0, 0.0]),
        delta0=(1e-6, 0.0),
        solver="rk4",
        solver_kwargs={"dt": 0.01},
        rhs_kwargs={"a": 0.3},
    )
    assert np.isclose(est, 0.3, atol=1e-2)
