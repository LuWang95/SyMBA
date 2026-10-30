import numpy as np
from unittest.mock import patch

import SyMBA_fixed_depth as fixed
import SyMBA_adaptive_depth as adaptive
import SyMBA_adaptive_depth_optimized as optimized

from symba_validation import (
    initial_state,
    integrate,
    reference,
    diagnostics,
    errors,
)


def test_single_level_matches_dh():
    q, v, m = initial_state("spatial")
    Q, P = fixed.cartesian_to_dh(q, m[:, None] * v, m)
    radii = [0.15]

    # With one level, all three versions should reduce to the DH step.
    for dt in [0.002, -0.002, 0.0]:
        Q_dh, P_dh = Q.copy(), P.copy()
        fixed.dh_step(Q_dh, P_dh, m, dt)

        for module in [fixed, adaptive, optimized]:
            Q_test, P_test = Q.copy(), P.copy()
            module.symba_step(Q_test, P_test, m, dt, radii)

            print(f"\n{module.__name__}, dt={dt}")
            print("Maximum position difference:", np.max(np.abs(Q_test - Q_dh)))
            print("Maximum momentum difference:", np.max(np.abs(P_test - P_dh)))

            np.testing.assert_allclose(Q_test, Q_dh, rtol=0, atol=2e-14)
            np.testing.assert_allclose(P_test, P_dh, rtol=0, atol=2e-18)

    print("Single-level DH equivalence test passed")


def test_prediction_boundaries():
    m = np.ones(3)
    radii = [1.0, 0.5]

    # Name, relative position, relative velocity, dt, expected decision.
    cases = [
        ("On boundary", [1, 0, 0], [0, 0, 0], 1, True),
        ("Just outside", [1 + 1e-10, 0, 0], [0, 0, 0], 1, False),
        ("Just inside", [1 - 1e-10, 0, 0], [0, 0, 0], 1, True),
        ("Cross during step", [-2, 0, 0], [4, 0, 0], 1, True),
        ("Enter at step end", [2, 0, 0], [-1, 0, 0], 1, True),
        ("Enter after step end", [2, 0, 0], [-0.99, 0, 0], 1, False),
        ("Backward encounter", [2, 0, 0], [1, 0, 0], -1, True),
        ("Tangent", [-2, 1, 0], [4, 0, 0], 1, True),
        ("Just miss tangent", [-2, 1 + 1e-10, 0], [4, 0, 0], 1, False),
    ]

    for name, position, velocity, dt, expected in cases:
        Q = np.zeros((3, 3))
        P = np.zeros((3, 3))
        Q[2] = position
        P[2] = velocity

        particles, pairs = optimized.get_deeper_particles_pairs(
            Q, P, m, [(1, 2)], dt, 0, radii
        )
        decision = adaptive.need_deeper_level(Q, P, m, dt, 0, radii)

        print(f"{name}: adaptive={decision}, optimized pairs={pairs}")
        assert pairs == ([(1, 2)] if expected else []), name
        assert particles == ({1, 2} if expected else set()), name
        assert decision == expected, name

    print("Encounter prediction boundary test passed")


def test_recursive_particle_time_budget():
    Q = np.array([
        [0.0, 0.0, 0.0],  # Center of mass
        [1.0, 0.0, 0.0],  # A
        [1.02, 0.0, 0.0],  # B
        [1.12, 0.0, 0.0],  # C
        [3.0, 0.0, 0.0],  # D
    ])
    P = np.zeros_like(Q)
    m = np.ones(5)
    dt = 0.027
    radii = [0.15, 0.075, 0.0375]
    pairs = optimized.make_planet_pairs(m)
    durations = {i: [] for i in range(1, 5)}

    # Freeze physical flows to check only the recursive scheduling.
    # AB reaches the deepest level, C stops earlier, D stays at the root.
    def recorded_drift(Q, P, m, particles, drift_dt, G=1.0):
        for i in particles:
            durations[i].append(drift_dt)

    with (
        patch.object(optimized, "kepler_drift_particles", new=recorded_drift),
        patch.object(optimized, "level_wise_interaction_kick"),
    ):
        optimized.evolve_level(
            Q, P, m, set(range(1, 5)), pairs, dt, 0, radii
        )

    expected_calls = {1: 9, 2: 9, 3: 3, 4: 1}
    for i in durations:
        total_time = sum(durations[i])
        print(f"Particle {i}: drift calls={len(durations[i])}, total time={total_time}")
        np.testing.assert_allclose(total_time, dt, rtol=0, atol=5e-16)
        assert len(durations[i]) == expected_calls[i]

    print("Recursive particle time-budget test passed")


def test_permutation_and_moving_barycenter():
    q, v, m = initial_state("spatial")
    q += [2.0, -1.0, 0.4]
    v += [0.2, -0.3, 0.1]
    Q, P = fixed.cartesian_to_dh(q, m[:, None] * v, m)

    dt = 0.002
    n_steps = 20
    radii = [0.15, 0.075, 0.0375]
    order = np.array([0, 3, 1, 4, 2])
    inverse_order = np.argsort(order)

    Q_original, P_original = Q.copy(), P.copy()
    Q_permuted, P_permuted = Q[order].copy(), P[order].copy()

    for _ in range(n_steps):
        optimized.symba_step(Q_original, P_original, m, dt, radii)
        optimized.symba_step(Q_permuted, P_permuted, m[order], dt, radii)

    # Restore the original particle order before comparing.
    Q_restored = Q_permuted[inverse_order]
    P_restored = P_permuted[inverse_order]
    print("Permutation position difference:", np.max(np.abs(Q_restored - Q_original)))
    print("Permutation momentum difference:", np.max(np.abs(P_restored - P_original)))
    np.testing.assert_allclose(Q_restored, Q_original, rtol=0, atol=2e-13)
    np.testing.assert_allclose(P_restored, P_original, rtol=0, atol=2e-17)

    Q_center_expected = Q[0] + n_steps * dt * P[0] / m.sum()
    print("Barycenter position error:", np.max(np.abs(Q_original[0] - Q_center_expected)))
    np.testing.assert_array_equal(P_original[0], P[0])
    np.testing.assert_allclose(Q_original[0], Q_center_expected, rtol=0, atol=1e-14)

    print("Permutation and moving-barycenter test passed")


def test_forward_backward_without_switching():
    q, v, m = initial_state("smooth")
    Q_initial, P_initial = fixed.cartesian_to_dh(q, m[:, None] * v, m)
    Q, P = Q_initial.copy(), P_initial.copy()
    dt = 0.01
    n_steps = 20
    radii = [0.15, 0.075, 0.0375]

    # This checks a smooth orbit without adaptive shell switching.
    for _ in range(n_steps):
        optimized.symba_step(Q, P, m, dt, radii)
    for _ in range(n_steps):
        optimized.symba_step(Q, P, m, -dt, radii)

    print("Reverse position error:", np.max(np.abs(Q - Q_initial)))
    print("Reverse momentum error:", np.max(np.abs(P - P_initial)))
    np.testing.assert_allclose(Q, Q_initial, rtol=0, atol=2e-13)
    np.testing.assert_allclose(P, P_initial, rtol=0, atol=2e-17)

    print("Forward/backward test without switching passed")


def test_encounters_against_ias15():
    dt = 0.00125
    t_end = 0.5
    output_interval = 0.005
    n_outputs = round(t_end / output_interval)
    times = np.arange(n_outputs + 1) * output_interval

    for case in ["shared", "independent", "spatial"]:
        print(f"\nEncounter case: {case}", flush=True)
        q, v, m = initial_state(case)
        q_ref, v_ref = reference(q, v, m, times)
        q_test, v_test = integrate(q, v, m, times, dt)
        metrics = diagnostics(q_test, v_test, m)

        position_error = errors(q_test, q_ref).max()
        velocity_error = errors(v_test, v_ref).max()
        energy_error = metrics["energy"].max()
        angular_error = metrics["angular"].max()
        min_distance = diagnostics(q_ref, v_ref, m)["distance"].min()

        print("Maximum position error against IAS15:", position_error)
        print("Maximum velocity error against IAS15:", velocity_error)
        print("Maximum relative energy error:", energy_error)
        print("Maximum relative angular momentum error:", angular_error)
        print("Minimum sampled reference pair distance:", min_distance)

        # Fixture-specific budgets; measurements are at output times.
        assert np.isfinite(q_test).all()
        assert np.isfinite(v_test).all()
        assert position_error < 1e-8, case
        assert velocity_error < 1e-7, case
        assert energy_error < 1e-7, case
        assert angular_error < 2e-12, case
        assert min_distance < 0.075, case

    print("Encounter comparison against IAS15 passed")


def test_smooth_second_order():
    q, v, m = initial_state("smooth")
    times = np.array([0.0, 1.0])
    q_ref, v_ref = reference(q, v, m, times)
    residuals = []

    for dt in [0.04, 0.02, 0.01]:
        q_test, v_test = integrate(q, v, m, times, dt)
        position_error = errors(q_test, q_ref)[-1]
        velocity_error = errors(v_test, v_ref)[-1]
        error = max(position_error, velocity_error)
        residuals.append(error)

        print(f"dt={dt:.3f}, final position error={position_error:.6e}, "
              f"final velocity error={velocity_error:.6e}")

    ratios = np.array(residuals[:-1]) / residuals[1:]
    print("Step-halving error ratios:", ratios)
    assert np.all((ratios > 3.5) & (ratios < 4.5)), (residuals, ratios)

    print("Smooth second-order convergence test passed")


if __name__ == "__main__":
    test_single_level_matches_dh()
    test_prediction_boundaries()
    test_recursive_particle_time_budget()
    test_permutation_and_moving_barycenter()
    test_forward_backward_without_switching()
    test_encounters_against_ias15()
    test_smooth_second_order()
