import numpy as np

from src.solvers import euler_integrate, rk4_integrate, rkf45_integrate



def decay_rhs(t, y):
    return -y



def exact_solution(t):
    return np.exp(-t)



def test_rk4_is_more_accurate_than_euler_on_decay_problem():
    y0 = np.array([1.0])
    t_span = (0.0, 1.0)
    dt = 0.1

    euler = euler_integrate(decay_rhs, t_span, y0, dt=dt)
    rk4 = rk4_integrate(decay_rhs, t_span, y0, dt=dt)

    euler_err = abs(euler.y[-1, 0] - exact_solution(1.0))
    rk4_err = abs(rk4.y[-1, 0] - exact_solution(1.0))
    assert rk4_err < euler_err



def test_rkf45_reaches_final_time_and_is_accurate():
    y0 = np.array([1.0])
    result = rkf45_integrate(decay_rhs, (0.0, 1.0), y0, h0=0.1, hmin=1e-6, hmax=0.2, tol=1e-8)
    assert np.isclose(result.t[-1], 1.0)
    assert abs(result.y[-1, 0] - exact_solution(1.0)) < 1e-5
