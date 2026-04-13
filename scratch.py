# %%
import numpy as np
from src.flows import steady_dipole_rhs
from src.solvers import rk4_integrate

# %%
y0 = np.array([0.2, 0.3])
res = rk4_integrate(steady_dipole_rhs, (0.0, 5.0), y0, dt=0.01)

print(res.method)
print(res.t.shape)
print(res.y.shape)
print(res.y[-1])
# %%
