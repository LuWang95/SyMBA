import numpy as np
from SyMBA import sun_drift,interaction_kick,kepler_drift



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

if __name__ == "__main__":
    test_sun_drift()
    test_interaction_kick()
    test_kepler_single()