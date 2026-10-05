import numpy as np
from SyMBA import *


def coordinate_test():
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


def test_sun_drift():
    m_test = np.array([2.0, 0.1, 0.2])

    Q_test = np.array([
        [5.0, 6.0, 7.0],  # barycenter
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    P_test = np.array([
        [10.0, 20.0, 30.0],  # total_momenta
        [0.2, 0.0, 0.0],
        [0.0, 0.4, 0.0],
    ])

    Q_before = Q_test.copy()
    P_before = P_test.copy()

    sun_drift(Q_test, P_test, m_test, dt=0.5)

    # displacement = 0.5 * [0.2, 0.4, 0.0] / 2
    #              = [0.05, 0.10, 0.0]
    Q_expected = np.array([
        [5.0, 6.0, 7.0],
        [1.05, 0.10, 0.0],
        [0.05, 2.10, 0.0],
    ])

    np.testing.assert_allclose(Q_test, Q_expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(P_test, P_before)

    # relative position stay the same
    np.testing.assert_allclose(
        Q_test[2] - Q_test[1],
        Q_before[2] - Q_before[1],
        rtol=0, atol=1e-14,
    )

    # reverse the step,should get recover the original location
    sun_drift(Q_test, P_test, m_test, dt=-0.5)
    np.testing.assert_allclose(Q_test, Q_before, rtol=0, atol=1e-14)

    print("sun_drift test passed")


def test_interaction_kick():
    m_test = np.array([1.0, 0.1, 0.2])
    Q_test = np.array([
        [0.0, 0.0, 0.0],  # barycenter
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    P_test = np.array([
        [0.3, -0.2, 0.1],  # total
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    Q_before = Q_test.copy()
    P_before = P_test.copy()

    interaction_kick(Q_test, P_test, m_test, dt=0.5)

    P_expected = P_before.copy()
    P_expected[1] = [0.0025, 0.0, 0.0]
    P_expected[2] = [-0.0025, 0.0, 0.0]

    np.testing.assert_allclose(
        P_test, P_expected, rtol=0, atol=1e-14
    )
    np.testing.assert_array_equal(Q_test, Q_before)
    np.testing.assert_array_equal(P_test[0], P_before[0])

    # the total momenta should stay the same
    np.testing.assert_allclose(
        np.sum(P_test[1:], axis=0),
        np.sum(P_before[1:], axis=0),
        rtol=0, atol=1e-14,
    )

    # reverse the step should recover the momenta
    interaction_kick(Q_test, P_test, m_test, dt=-0.5)
    np.testing.assert_allclose(
        P_test, P_before, rtol=0, atol=1e-14
    )

    print("interaction_kick test passed")


def test_kepler_single():
    m_test = np.array([1.0, 1e-3])

    Q_test = np.array([
        [2.0, 3.0, 4.0], # barycenter
        [1.0, 0.0, 0.0],
    ])

    P_test = np.array([
        [0.3, 0.2, 0.1],  # total momenta
        [0.0, 1e-3, 0.0],
    ])

    Q_before = Q_test.copy()
    P_before = P_test.copy()

    # 1/4 period
    kepler_drift(Q_test, P_test, m_test, dt=np.pi / 2)

    np.testing.assert_allclose(
        Q_test[1], [0.0, 1.0, 0.0],
        rtol=0, atol=1e-14,
    )
    np.testing.assert_allclose(
        P_test[1] / m_test[1], [-1.0, 0.0, 0.0],
        rtol=0, atol=1e-14,
    )

    np.testing.assert_array_equal(Q_test[0], Q_before[0])
    np.testing.assert_array_equal(P_test[0], P_before[0])

    # reverse the step
    kepler_drift(Q_test, P_test, m_test, dt=-np.pi / 2)

    np.testing.assert_allclose(
        Q_test, Q_before, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        P_test, P_before, rtol=0, atol=1e-14
    )

    print("REBOUND kepler_drift_single passed")

def test_dh_step():
    G = 1.0
    m = np.array([1.0, 1e-3, 2e-3])

    q = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-np.sqrt(0.5), 0.0, 0.0],
    ])

    p = m[:, None] * v

    Q, P = cartesian_to_dh(q, p, m)

    Q_initial = Q.copy()
    P_initial = P.copy()

    E_initial = total_energy(Q, P, m, G)

    dt = 0.02
    n_steps = 1000

    energy_rel_errors = []

    for _ in range(n_steps):
        dh_step(Q, P, m, dt, G)

        E = total_energy(Q, P, m, G)
        energy_rel_errors.append((E - E_initial) / abs(E_initial))

    print(
        "maximum_energy_rel_errors:",
        np.max(np.abs(energy_rel_errors)),
    )

    # Check the independently known center-of-mass motion.
    Q_com_expected = (Q_initial[0] + n_steps * dt * P_initial[0] / np.sum(m))

    np.testing.assert_allclose(
        Q[0], Q_com_expected, rtol=0, atol=1e-12,
    )
    np.testing.assert_array_equal(P[0], P_initial[0])

    # Reverse all steps.
    for _ in range(n_steps):
        dh_step(Q, P, m, -dt, G)

    print("position error:", np.max(np.abs(Q - Q_initial)))
    print("momenta error:", np.max(np.abs(P - P_initial)))

def test_dh_step_ratio():
    G = 1.0
    m = np.array([1.0, 1e-3, 2e-3])

    q = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])

    v = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-np.sqrt(0.5), 0.0, 0.0],
    ])

    p = m[:, None] * v
    T = 20.0

    previous_error = None

    for dt in [0.04, 0.02, 0.01, 0.005]:
        # each stepping will change Q,P in place, so we have to reset Q and P each time
        Q, P = cartesian_to_dh(q, p, m)
        E_initial = total_energy(Q, P, m, G)

        n_steps = round(T / dt)
        max_error = 0.0

        for _ in range(n_steps):
            dh_step(Q, P, m, dt, G)

            E = total_energy(Q, P, m, G)
            error = abs(E - E_initial) / abs(E_initial)
            max_error = max(max_error, error)

        if previous_error is None:
            print(f"dt={dt:.3f}, max energy error={max_error:.6e}")
        else:
            ratio = previous_error / max_error
            print(
                f"dt={dt:.3f}, max energy error={max_error:.6e}, "
                f"ratio={ratio:.3f}"
            )

        previous_error = max_error

def test_force_weight():
    radii = np.array([1.0, 0.5, 0.25])

    test_cases = [
        (2.0, [1.0, 0.0, 0.0]),
        (0.75, [0.5, 0.5, 0.0]),
        (0.375, [0.0, 0.5, 0.5]),
        (0.10, [0.0, 0.0, 1.0]),
    ]

    for r, expected in test_cases:
        weights = level_weights(r, radii)

        np.testing.assert_allclose(
            weights, expected, rtol=0, atol=1e-14
        )

    # Check that the decomposition preserves the complete force.
    for r in np.geomspace(0.01, 10.0, 200):
        weights = level_weights(r, radii)

        np.testing.assert_allclose(
            np.sum(weights), 1.0, rtol=0, atol=1e-14
        )
        assert np.all(weights >= -1e-14)

    print("force_weights test passed")



if __name__ == "__main__":
    coordinate_test()
    test_sun_drift()
    test_interaction_kick()
    test_kepler_single()
    test_dh_step()
    test_dh_step_ratio()