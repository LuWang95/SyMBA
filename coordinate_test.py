import numpy as np
from SyMBA import cartesian_to_dh, dh_to_cartesian



rng = np.random.default_rng(42)

m = np.array([1.0, 1e-3, 2e-3])
q = rng.normal(size=(3, 3))
v = rng.normal(size=(3, 3))
p = m[:, None] * v

Q, P = cartesian_to_dh(q, p, m)
q_back, p_back = dh_to_cartesian(Q, P, m)

np.testing.assert_allclose(q_back, q, rtol=0, atol=1e-14)
np.testing.assert_allclose(p_back, p, rtol=0, atol=1e-14)

print("q_error:", np.max(np.abs(q_back - q)))
print("p_error:", np.max(np.abs(p_back - p)))
