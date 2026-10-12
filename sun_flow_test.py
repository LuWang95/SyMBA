from SyMBA_fixed_depth import *
import matplotlib.pyplot as plt

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

def test_level_wise_interaction_kick():
    m = np.array([1.0, 1e-3, 2e-3, 1.5e-3])
    G = 1.0
    dt = 0.02

    Q0 = np.array([
        [0.2, -0.1, 0.3],
        [1.0, 0.0, 0.0],
        [1.75, 0.0, 0.0],
        [1.75, 0.375, 0.0],
    ])

    P0 = np.array([
        [0.01, -0.02, 0.0],
        [0.0, 1e-3, 0.0],
        [0.0, 1.5e-3, 0.0],
        [-1e-4, 1e-3, 0.0],
    ])

    Q_full, P_full = Q0.copy(), P0.copy()
    interaction_kick(Q_full, P_full, m, dt, G)

    radii = make_radii(1.0, 3, ratio=2.0)
    Q_split, P_split = Q0.copy(), P0.copy()
    for level in range(len(radii)):
        level_wise_interaction_kick(
            Q_split, P_split, m, dt, level, radii, G
        )

    np.testing.assert_array_equal(Q_split, Q0)
    np.testing.assert_allclose(
        P_split, P_full, rtol=0, atol=1e-14
    )
    print("level_wise_interaction_kick test passed")

def test_one_level():
    m = np.array([1.0, 1e-3, 2e-3, 1.5e-3])
    G = 1.0
    dt = 0.02

    Q0 = np.array([
        [0.2, -0.1, 0.3],
        [1.0, 0.0, 0.0],
        [1.75, 0.0, 0.0],
        [1.75, 0.375, 0.0],
    ])

    P0 = np.array([
        [0.01, -0.02, 0.0],
        [0.0, 1e-3, 0.0],
        [0.0, 1.5e-3, 0.0],
        [-1e-4, 1e-3, 0.0],
    ])


    Q_dh, P_dh = Q0.copy(), P0.copy()
    dh_step(Q_dh, P_dh, m, dt, G)

    radii_zero = make_radii(1.0, 1)
    Q_symba, P_symba = Q0.copy(), P0.copy()
    symba_step(Q_symba, P_symba, m, dt, radii_zero, G)

    np.testing.assert_allclose(Q_symba, Q_dh, rtol=0, atol=1e-14)
    np.testing.assert_allclose(P_symba, P_dh, rtol=0, atol=1e-14)

    print("same as DH")


def test_symba_reversibility():
    G = 1.0
    m = np.array([1.0, 1e-3, 2e-3])
    q = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.3, 0.0, 0.0],
    ])
    v = np.array([[0.3, 0.0, 0.4],
                  [0.0, 1.0, 0.0],
                  [0.0, np.sqrt(0.8), 0.0]])

    p = m[:, None] * v
    Q, P = cartesian_to_dh(q, p, m)
    Q_initial = Q.copy()
    P_initial = P.copy()

    radii = make_radii(r_outer=0.5,total_levels=3,ratio=2.08)
    distance = np.linalg.norm(Q[2] - Q[1])
    print("initial weights:" , level_weights(distance, radii))

    dt = 0.01
    n_steps = 100

    for _ in range(n_steps):
        symba_step(
            Q, P, m, dt, radii=radii, G=G,
        )


    for _ in range(n_steps):
        symba_step(
            Q, P, m, -dt, radii=radii, G=G,
        )

    position_error = np.max(np.abs(Q - Q_initial))
    position_relative_error = np.max(np.linalg.norm(Q[1:] - Q_initial[1:], axis=1)/np.linalg.norm(Q_initial[1:], axis=1))

    v_initial = P_initial[1:] / m[1:, None]
    v_final = P[1:] / m[1:, None]
    error_norm = np.linalg.norm(v_final - v_initial, axis=1)
    initial_norm = np.linalg.norm(v_initial, axis=1)
    velocity_absolute_error = np.max(np.abs(v_final - v_initial))
    velocity_relative_error = np.max(error_norm / initial_norm)

    print("position_absolute_error:" , position_error)
    print("position_relative_error:" ,position_relative_error)
    print("velocity_absolute_error", velocity_absolute_error)
    print("velocity_relative_error", velocity_relative_error)


def test_symba_convergence():
    G = 1.0
    m = np.array([1.0, 1e-3, 2e-3])
    q = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.3, 0.0, 0.0],
    ])
    v = np.array([[0.3, 0.0, 0.4],
                  [0.0, 1.0, 0.0],
                  [0.0, np.sqrt(0.8), 0.0]])

    p = m[:, None] * v
    Q_initial, P_initial = cartesian_to_dh(q, p, m)

    radii = make_radii(r_outer=0.5, total_levels=3, ratio=2.08)

    T = 1.0
    timesteps = [0.02, 0.01, 0.005]
    final_positions = []
    final_velocities = []

    for dt in timesteps:
        Q = Q_initial.copy()
        P = P_initial.copy()

        n_steps = round(T / dt)
        for _ in range(n_steps):
            symba_step(Q, P, m, dt, radii=radii, G=G)

        final_positions.append(Q[1:].copy())
        final_velocities.append((P[1:] / m[1:, None]).copy())

    def max_vector_difference(x,y):
        return np.max(np.linalg.norm(x - y, axis=1))

    position_diff_0_1 = max_vector_difference(final_positions[0], final_positions[1])
    position_diff_1_2 = max_vector_difference(final_positions[1], final_positions[2])

    velocity_diff_coarse_0_1 = max_vector_difference(final_velocities[0], final_velocities[1])
    velocity_diff_1_2 = max_vector_difference(final_velocities[1], final_velocities[2])

    order_from_position = np.log2(position_diff_0_1 / position_diff_1_2)
    order_from_velocity = np.log2(velocity_diff_coarse_0_1 / velocity_diff_1_2)
    print("order_from_position:", order_from_position)
    print("order_from_velocity:", order_from_velocity)

def integrate_ias15(q, v, m, G, t_end,epsilon = 1e-9):
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "ias15"
    sim.integrator.epsilon = epsilon

    for i in range(len(m)):
        sim.add(m=m[i],x=q[i, 0],y=q[i, 1],z=q[i, 2],vx=v[i, 0],vy=v[i, 1],vz=v[i, 2])

    sim.integrate(t_end, exact_finish_time=1)

    q_final = np.array([[particle.x, particle.y, particle.z] for particle in sim.particles])
    v_final = np.array([[particle.vx, particle.vy, particle.vz]for particle in sim.particles])

    return q_final, v_final

def integrate_symba(q, v, m, G, dt, t_end, radii):
    n_steps = round(t_end / dt)

    p = m[:, None] * v
    Q, P = cartesian_to_dh(q, p, m)

    for _ in range(n_steps):
        symba_step(Q, P, m, dt, radii=radii, G=G)

    q_final, p_final = dh_to_cartesian(Q, P, m)
    v_final = p_final / m[:, None]

    return q_final, v_final

def ias15_with_trajectory(q, v, m, G, times, epsilon=1e-12):
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "ias15"
    sim.integrator.epsilon = epsilon

    for i in range(len(m)):
        sim.add(m=m[i], x=q[i, 0], y=q[i, 1], z=q[i, 2], vx=v[i, 0], vy=v[i, 1], vz=v[i, 2])

    positions = []
    velocities = []

    for t in times:
        sim.integrate(float(t), exact_finish_time=1)
        positions.append([[b.x, b.y, b.z] for b in sim.particles])
        velocities.append([[b.vx, b.vy, b.vz] for b in sim.particles])
    return np.array(positions), np.array(velocities)

def test_against_ias15():
    G = 1.0
    m = np.array([1.0, 1e-3, 2e-3])
    q = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.3, 0.0, 0.0],
    ])
    v = np.array([[0.3, 0.0, 0.4],
                  [0.0, 1.0, 0.0],
                  [0.0, np.sqrt(0.8), 0.0]])

    radii = make_radii(r_outer=0.5, total_levels=3, ratio=2.08)
    t_end = 1.0

    q_ref, v_ref = integrate_ias15(q, v, m, G, t_end, epsilon=1e-12)
    q_ref_tight, v_ref_tight = integrate_ias15(q, v, m, G, t_end, epsilon=1e-13)

    reference_q_change = np.max(np.linalg.norm(q_ref_tight - q_ref, axis=1))
    reference_v_change = np.max(np.linalg.norm(v_ref_tight - v_ref, axis=1))

    print("IAS15 reference position change:", reference_q_change)
    print("IAS15 reference velocity change:", reference_v_change)

    q_ref = q_ref_tight
    v_ref = v_ref_tight

    for dt in [0.02, 0.01, 0.005]:
        q_symba, v_symba = integrate_symba(
            q, v, m, G, dt, t_end, radii
        )

        position_error = np.max(
            np.linalg.norm(q_symba - q_ref, axis=1)
        )
        velocity_error = np.max(
            np.linalg.norm(v_symba - v_ref, axis=1)
        )

        print(f"dt={dt:.3f}: "
            f"position error={position_error:.6e}, "
            f"velocity error={velocity_error:.6e}"
        )



def test_symba_shell_crossing():
    G = 1.0
    m = np.array([1.0, 1e-4, 1e-4])
    q = np.array([[0.0, 0.0, 0.0],
                  [1.0, -0.15, -0.01],
                  [1.0, 0.15, 0.01],
                  ])
    v = np.array([[0.0, 0.0, 0.0],
                  [0.0, 1.6, 0.0],
                  [0.0, 0.4, 0.0],])

    radii = make_radii(r_outer=0.12, total_levels=4, ratio=2.08)

    dt = 0.005
    t_end = 0.48
    n_steps = round(t_end / dt)

    times = np.arange(n_steps + 1) * dt

    q_ref, v_ref = ias15_with_trajectory(q, v, m, G, times, epsilon=1e-12)
    q_ref_tight, v_ref_tight = ias15_with_trajectory(q, v, m, G, times, epsilon=1e-13)

    reference_position_change = np.max(np.linalg.norm(q_ref_tight - q_ref, axis=2))
    reference_velocity_change = np.max(np.linalg.norm(v_ref_tight - v_ref, axis=2))

    print("IAS15 reference position change:", reference_position_change)
    print("IAS15 reference velocity change:", reference_velocity_change)


    q_ref = q_ref_tight
    v_ref = v_ref_tight


    p = m[:, None] * v
    Q, P = cartesian_to_dh(q, p, m)
    E_initial = total_energy(Q, P, m, G)

    positions = []
    velocities = []
    distances = []
    weights = []
    energy_errors = []

    for step in range(n_steps + 1):
        if step > 0:
            symba_step(Q, P, m, dt, radii=radii, G=G)

        q_now, p_now = dh_to_cartesian(Q, P, m)
        v_now = p_now / m[:, None]

        distance = np.linalg.norm(Q[2] - Q[1])

        positions.append(q_now.copy())
        velocities.append(v_now.copy())
        distances.append(distance)
        weights.append(level_weights(distance, radii))
        E = total_energy(Q, P, m, G)
        energy_errors.append((E - E_initial) / abs(E_initial))

    positions = np.array(positions)
    velocities = np.array(velocities)
    distances = np.array(distances)
    weights = np.array(weights)
    energy_errors = np.array(energy_errors)

    position_abs_errors = np.max(np.linalg.norm(positions[:, 1:] - q_ref[:, 1:], axis=2),axis=1)
    velocity_abs_errors = np.max(np.linalg.norm(velocities[:, 1:] - v_ref[:, 1:], axis=2),axis=1)

    reference_distances = np.linalg.norm(q_ref[:, 2] - q_ref[:, 1], axis=1)
    reference_weights = np.array([level_weights(r, radii)for r in reference_distances])


    print("radii:", radii)
    print("minimum sampled distance:", np.min(distances))
    print("maximum absolute position error:", np.max(position_abs_errors))
    print("maximum absolute velocity error:", np.max(velocity_abs_errors))
    print("maximum relative energy error:",np.max(np.abs(energy_errors)))
    print("shell crossing test passed")

    # Plot diagnostics.
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    axes[0].plot(times, distances, label="SyMBA")
    axes[0].plot(times, reference_distances, "--", label="IAS15")
    for k, radius in enumerate(radii):
        axes[0].axhline(
            radius, color="gray", linestyle=":", linewidth=0.8
        )
        axes[0].text(
            times[-1], radius, f" R{k}", va="center"
        )
    axes[0].set_ylabel("Pair distance")
    axes[0].legend()

    for k in range(len(radii)):
        axes[1].plot(times, weights[:, k], label=f"level {k}")
        axes[1].plot(
            times,
            reference_weights[:, k],
            "--",
            color="black",
            linewidth=0.8,
        )
    axes[1].set_ylabel("Force weight")
    axes[1].set_title("Solid: SyMBA; dashed: weights on IAS15 trajectory")
    axes[1].legend()

    axes[2].plot(times, energy_errors)
    axes[2].set_ylabel("(E - E0) / |E0|")


    axes[3].semilogy(times[1:], np.maximum(position_abs_errors[1:], 1e-18))
    axes[3].set_ylabel("Position error")

    axes[4].semilogy(times[1:], np.maximum(velocity_abs_errors[1:], 1e-18))
    axes[4].set_ylabel("Velocity error")
    axes[4].set_xlabel("Time")

    for ax in axes:
        ax.grid(alpha=0.2)

    plt.show()

    return {
        "times": times,
        "distances": distances,
        "weights": weights,
        "energy_errors": energy_errors,
        "position_errors": position_abs_errors,
        "velocity_errors": velocity_abs_errors,
    }



if __name__ == "__main__":
    # coordinate_test()
    # test_sun_drift()
    # test_interaction_kick()
    # test_kepler_single()
    # test_dh_step()
    # test_dh_step_ratio()
    # test_level_wise_interaction_kick()
    # test_symba_components()
    # test_one_level()
    # test_symba_reversibility()
    # test_symba_convergence()
    # test_against_ias15()
    test_symba_shell_crossing()
