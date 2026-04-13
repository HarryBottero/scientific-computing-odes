import numpy as np

from src.flows import steady_dipole_streamfunction, steady_dipole_velocity



def test_velocity_shapes_match_input_shapes():
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    u, v = steady_dipole_velocity(X, Y)
    assert u.shape == X.shape
    assert v.shape == Y.shape



def test_streamfunction_matches_velocity_by_finite_difference():
    x0, y0 = 0.2, 0.3
    h = 1e-6
    psi_x_plus = steady_dipole_streamfunction(x0 + h, y0)
    psi_x_minus = steady_dipole_streamfunction(x0 - h, y0)
    psi_y_plus = steady_dipole_streamfunction(x0, y0 + h)
    psi_y_minus = steady_dipole_streamfunction(x0, y0 - h)

    dpsi_dx = (psi_x_plus - psi_x_minus) / (2 * h)
    dpsi_dy = (psi_y_plus - psi_y_minus) / (2 * h)
    u, v = steady_dipole_velocity(x0, y0)

    assert np.isclose(u, -dpsi_dy, rtol=1e-5, atol=1e-7)
    assert np.isclose(v, dpsi_dx, rtol=1e-5, atol=1e-7)
