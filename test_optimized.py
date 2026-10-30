"""Optimized SyMBA component tests and full-step comparison."""
import numpy as np

import SyMBA_fixed_depth as fixed
import SyMBA_adaptive_depth_optimized as optimized
from close_encounter_covergence import (
    make_initial_state,
    reference_trajectory,
    max_planet_error,
)
from test_adaptive import run_forward



def test_pair_kick():
    m = np.array([1.0, 1e-3, 2e-3, 1.5e-3])

    Q = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [1.1, 0.2, 0.0],
    ])

    P = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1e-3, 0.0],
        [0.0, 2e-3, 0.0],
        [0.0, 1.5e-3, 0.0],
    ])

    pairs = optimized.make_planet_pairs(m)
    radii = fixed.make_radii(0.3, 4)
    dt = 0.005

    assert pairs == [(1, 2), (1, 3), (2, 3)]

    for level in range(len(radii)):
        Q_fixed, P_fixed = Q.copy(), P.copy()
        Q_pairs, P_pairs = Q.copy(), P.copy()

        fixed.level_wise_interaction_kick(
            Q_fixed, P_fixed, m, dt, level, radii
        )

        active_pairs = optimized.get_active_pairs(
            Q_pairs, pairs, level, radii
        )

        optimized.level_wise_interaction_kick(
            Q_pairs, P_pairs, m,
            active_pairs, dt, level, radii,
        )

        print(f"level={level}, active pairs={active_pairs}")

        np.testing.assert_array_equal(Q_pairs, Q)
        np.testing.assert_allclose(
            P_pairs, P_fixed, rtol=0, atol=1e-15
        )

        print(
            f"level={level}: "
            f"maximum momentum difference="
            f"{np.max(np.abs(P_pairs - P_fixed)):.6e}"
        )

    print("Pair-list kick test passed")


def test_selected_kepler_drift():
    m = np.array([1.0, 1e-3, 2e-3, 1.5e-3])

    Q = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [-2.0, 0.0, 0.0],
    ])

    P = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1e-3, 0.0],
        [-2e-3 / np.sqrt(1.5), 0.0, 0.0],
        [0.0, -1.5e-3 / np.sqrt(2.0), 0.0],
    ])

    dt = 0.01

    # Reference: advance all planets.
    Q_full, P_full = Q.copy(), P.copy()
    fixed.kepler_drift(Q_full, P_full, m, dt)

    # Advance only particles 1 and 3.
    Q_selected, P_selected = Q.copy(), P.copy()
    optimized.kepler_drift_particles(Q_selected, P_selected, m, {1, 3}, dt)

    np.testing.assert_allclose(Q_selected[[1, 3]], Q_full[[1, 3]],rtol=0, atol=1e-14)
    np.testing.assert_allclose(P_selected[[1, 3]], P_full[[1, 3]],rtol=0, atol=1e-14)

    # Particle 2 and the center-of-mass state must remain unchanged.
    np.testing.assert_array_equal(Q_selected[[0, 2]], Q[[0, 2]])
    np.testing.assert_array_equal(P_selected[[0, 2]], P[[0, 2]])

    # Advance particle 2; the final state should match advancing all planets.
    optimized.kepler_drift_particles(Q_selected, P_selected, m, {2}, dt)
    np.testing.assert_allclose(Q_selected, Q_full, rtol=0, atol=1e-14)
    np.testing.assert_allclose(P_selected, P_full, rtol=0, atol=1e-14)

    print("Selected-particle Kepler drift test passed")


def test_deeper_particles_pairs():
    m = np.array([1.0, 1e-3, 1e-3, 1e-3, 1e-3])

    Q = np.array([
        [0.0, 0.0, 0.0],  # Center of mass
        [1.0, 0.0, 0.0],  # A
        [1.2, 0.0, 0.0],  # B
        [1.4, 0.0, 0.0],  # C
        [3.0, 0.0, 0.0],  # D
    ])

    P = np.zeros_like(Q)
    radii = fixed.make_radii(0.3, 4)
    pairs = optimized.make_planet_pairs(m)

    # 1. AB and BC are inside R0; AC and all pairs involving D are outside.
    deeper_particles, deeper_pairs = (
        optimized.get_deeper_particles_pairs(
            Q, P, m, pairs,
            dt=0.01, level=0, radii=radii,
        )
    )

    assert set(deeper_pairs) == {(1, 2), (2, 3)}
    assert len(deeper_pairs) == 2  # No duplicates allowed
    assert deeper_particles == {1, 2, 3}

    # B belongs to two pairs but appears only once in the particle set.
    current_particles = set(range(1, len(m)))
    drift_here = current_particles - deeper_particles

    assert drift_here == {4}
    assert drift_here.isdisjoint(deeper_particles)
    assert drift_here | deeper_particles == current_particles

    print("Shared-particle selection passed")

    # 2. Check whether level 1 requires deeper recursion.
    # AB and BC both have separation 0.2 > R1 and zero relative velocity.
    particles_next, pairs_next = (
        optimized.get_deeper_particles_pairs(
            Q, P, m, deeper_pairs,
            dt=0.01 / 3, level=1, radii=radii,
        )
    )

    assert particles_next == set()
    assert pairs_next == []

    print("Deeper-shell selection passed")

    # 3. Start outside R0 but enter it during this step.
    Q_moving = Q.copy()
    P_moving = P.copy()

    Q_moving[2] = [1.4, 0.0, 0.0]
    P_moving[2] = m[2] * np.array([-0.2, 0.0, 0.0])

    particles_forward, pairs_forward = (
        optimized.get_deeper_particles_pairs(
            Q_moving, P_moving, m, [(1, 2)],
            dt=1.0, level=0, radii=radii,
        )
    )

    assert particles_forward == {1, 2}
    assert pairs_forward == [(1, 2)]

    # Starting from the same state, backward motion increases the separation.
    particles_backward, pairs_backward = (
        optimized.get_deeper_particles_pairs(
            Q_moving, P_moving, m, [(1, 2)],
            dt=-1.0, level=0, radii=radii,
        )
    )

    assert particles_backward == set()
    assert pairs_backward == []

    print("Forward/backward prediction passed")

    # 4. Stop recursion at the maximum level.
    particles_max, pairs_max = (
        optimized.get_deeper_particles_pairs(
            Q, P, m, pairs,
            dt=0.01, level=len(radii) - 1, radii=radii,
        )
    )

    assert particles_max == set()
    assert pairs_max == []

    print("Maximum-level stopping passed")


def test_symba_step(dt=0.005, t_end=32.0, output_interval=0.02):
    q, v, m, G = make_initial_state()


    radii = fixed.make_radii(
        r_outer=0.3,
        total_levels=4,
        ratio=2.08,
    )

    n_outputs = round(t_end / output_interval)
    times = np.arange(n_outputs + 1) * output_interval

    print("Computing IAS15 reference...", flush=True)
    q_ref, v_ref = reference_trajectory(
        q, v, m, G, times
    )

    results = {}

    for name, module in [
        ("fixed", fixed),
        ("optimized", optimized),
    ]:
        print(f"\nRunning {name}...", flush=True)

        result = run_forward(
            module, q, v, m, G, times, dt, radii
        )
        results[name] = result

        print(
            "Maximum position error against IAS15:",
            max_planet_error(result["q"], q_ref),
        )
        print(
            "Maximum velocity error against IAS15:",
            max_planet_error(result["v"], v_ref),
        )
        print(
            "Maximum relative energy error:",
            result["energy_error"],
        )
        print(
            "Maximum relative angular momentum error:",
            result["angular_error"],
        )
        print(
            "Minimum distance at main-step endpoints:",
            result["min_distance"],
        )

    print("\nOPTIMIZED VS FIXED")

    print(
        "Maximum position difference:",
        max_planet_error(
            results["optimized"]["q"],
            results["fixed"]["q"],
        ),
    )
    print(
        "Maximum velocity difference:",
        max_planet_error(
            results["optimized"]["v"],
            results["fixed"]["v"],
        ),
    )


def main():
    # Comment out individual tests when only one example is needed.
    test_pair_kick()
    test_selected_kepler_drift()
    test_deeper_particles_pairs()
    test_symba_step()


if __name__ == "__main__":
    main()
