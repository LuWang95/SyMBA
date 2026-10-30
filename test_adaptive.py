"""Adaptive SyMBA tests, encounter plots, and shared integration helpers."""
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

import SyMBA_fixed_depth as fixed
import SyMBA_adaptive_depth as adaptive
import SyMBA_adaptive_depth_optimized as optimized
from close_encounter_covergence import (
    make_initial_state,
    reference_trajectory,
    max_planet_error,
)

# Settings for the multiple-encounter examples.
CASES_TO_RUN = ["shared", "independent", "spatial"]
DT = 0.0025
T_END = 1.0
OUTPUT_INTERVAL = 0.005
TOTAL_LEVELS = 4
R_OUTER = 0.15
RUN_REVERSE = True



# Shared forward and reverse integration helpers.


def run_forward(module, q, v, m, G, times, dt, radii):
    output_interval = times[1] - times[0]
    steps_per_output = round(output_interval / dt)

    assert dt > 0.0
    assert steps_per_output >= 1
    np.testing.assert_allclose(
        np.diff(times),
        steps_per_output * dt,
        rtol=0,
        atol=1e-12,
    )

    Q, P = fixed.cartesian_to_dh(q, m[:, None] * v, m)

    energy_initial = fixed.total_energy(Q, P, m, G)
    angular_initial = np.sum(
        np.cross(q, m[:, None] * v),
        axis=0,
    )
    angular_scale = np.linalg.norm(angular_initial)
    assert angular_scale > 0.0

    positions = [q.copy()]
    velocities = [v.copy()]

    max_energy_error = 0.0
    max_angular_error = 0.0
    min_distance = np.linalg.norm(Q[2] - Q[1])

    kepler_calls = 0
    original_drift = getattr(module, "kepler_drift")

    def counted_drift(*args, **kwargs):
        nonlocal kepler_calls
        kepler_calls += 1
        return original_drift(*args, **kwargs)

    # noinspection PyUnresolvedReferences
    with patch.object(module, "kepler_drift", new=counted_drift):
        for _ in times[1:]:
            for _ in range(steps_per_output):
                module.symba_step(Q, P, m, dt, radii, G)

                q_now, p_now = fixed.dh_to_cartesian(Q, P, m)

                energy = fixed.total_energy(Q, P, m, G)
                max_energy_error = max(max_energy_error,abs(energy - energy_initial) / abs(energy_initial),)
                angular = np.sum(np.cross(q_now, p_now),axis=0,)
                max_angular_error = max(max_angular_error,np.linalg.norm(angular - angular_initial)/ angular_scale,)
                min_distance = min(min_distance,np.linalg.norm(Q[2] - Q[1]),)
            positions.append(q_now.copy())
            velocities.append(p_now / m[:, None])

    positions = np.array(positions)
    velocities = np.array(velocities)

    assert np.isfinite(positions).all()
    assert np.isfinite(velocities).all()

    return {
        "q": positions,
        "v": velocities,
        "Q_final": Q.copy(),
        "P_final": P.copy(),
        "energy_error": max_energy_error,
        "angular_error": max_angular_error,
        "min_distance": min_distance,
        "kepler_calls": kepler_calls,
    }


def run_reverse(module, result, q, v, m, G, dt, n_steps, radii):
    Q = result["Q_final"].copy()
    P = result["P_final"].copy()

    Q_initial, P_initial = fixed.cartesian_to_dh(
        q, m[:, None] * v, m
    )

    for _ in range(n_steps):
        module.symba_step(Q, P, m, -dt, radii, G)

    assert np.isfinite(Q).all()
    assert np.isfinite(P).all()

    position_error = np.max(np.linalg.norm(Q[1:] - Q_initial[1:], axis=1))

    velocity_error = np.max(np.linalg.norm((P[1:] - P_initial[1:]) / m[1:, None],axis=1,))

    return position_error, velocity_error


# Single-encounter comparison against fixed SyMBA and IAS15.


def test_adaptive_step(dt=0.0025, t_end=32.0, output_interval=0.02):
    q, v, m, G = make_initial_state()

    total_levels = 4

    radii = fixed.make_radii(
        r_outer=0.3,
        total_levels=total_levels,
        ratio=2.08,
    )

    n_steps = round(t_end / dt)
    n_outputs = round(t_end / output_interval)

    assert np.isclose(n_steps * dt, t_end)
    assert np.isclose(n_outputs * output_interval, t_end)

    times = np.arange(n_outputs + 1) * output_interval

    print("dt:", dt)
    print("total_levels:", total_levels)
    print("radii:", radii)

    print("\nComputing IAS15 reference...", flush=True)
    q_ref, v_ref = reference_trajectory(q, v, m, G, times)

    reference_distances = np.linalg.norm(q_ref[:, 2] - q_ref[:, 1],axis=1,)

    print("IAS15 minimum distance at output times:",np.min(reference_distances),)

    results = {}

    for name, module in [("fixed", fixed),("adaptive", adaptive), ]:
        print(f"\nRunning {name} forward...", flush=True)
        result = run_forward( module, q, v, m, G, times, dt, radii)
        results[name] = result

        print("Maximum position error against IAS15:",max_planet_error(result["q"], q_ref))
        print("Maximum velocity error against IAS15:",max_planet_error(result["v"], v_ref))
        print("Maximum relative energy error:",result["energy_error"])
        print("Maximum relative angular momentum error:",result["angular_error"])
        print("Minimum distance at main-step endpoints:",result["min_distance"])
        print("Kepler drift calls:", result["kepler_calls"])

        print(f"Running {name} backward...", flush=True)

        reverse_q_error, reverse_v_error = run_reverse(module, result, q, v, m, G,
                                                       dt, n_steps, radii,)

        print("Reverse position error:", reverse_q_error)
        print("Reverse barycentric velocity error:", reverse_v_error)

    fixed_result = results["fixed"]
    adaptive_result = results["adaptive"]

    print("\nADAPTIVE VS FIXED")

    position_differences = np.max(np.linalg.norm(adaptive_result["q"][:, 1:]- fixed_result["q"][:, 1:],axis=2,),axis=1,)

    velocity_differences = np.max(np.linalg.norm( adaptive_result["v"][:, 1:]- fixed_result["v"][:, 1:],axis=2,),axis=1,)

    iq = np.argmax(position_differences)
    iv = np.argmax(velocity_differences)

    print(f"Maximum position difference: "
          f"{position_differences[iq]:.6e} at t={times[iq]:.4f}"
    )
    print(
        f"Maximum velocity difference: "
        f"{velocity_differences[iv]:.6e} at t={times[iv]:.4f}"
    )

    expected_fixed_calls = n_steps * 3 ** (total_levels - 1)
    assert fixed_result["kepler_calls"] == expected_fixed_calls

    reduction = (1.0- adaptive_result["kepler_calls"]/ fixed_result["kepler_calls"])

    print(f"Kepler drift call reduction: {reduction:.2%}")


# Multiple-encounter fixtures, execution recording, and visualization.


def make_case(name):
    """Controlled encounter initial conditions with positive masses and G=1."""
    if name == "shared":
        title = "Shared-particle encounters: 4 bodies"

        planet_q = [
            [1.00, -0.15, 0.0],
            [1.03,  0.00, 0.0],
            [0.98,  0.15, 0.0],
        ]
        planet_v = [
            [0.0, 1.4, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.6, 0.0],
        ]

    elif name == "independent":
        title = "Two encounter groups: 5 bodies"

        planet_q = [
            [ 1.00, -0.12, 0.0],
            [ 1.03,  0.12, 0.0],
            [-1.00, -0.12, 0.0],
            [-1.03,  0.12, 0.0],
        ]
        planet_v = [
            [0.0,  1.4, 0.0],
            [0.0,  0.6, 0.0],
            [0.0, -0.6, 0.0],
            [0.0, -1.4, 0.0],
        ]

    elif name == "spatial":
        title = "Three-dimensional encounters: 5 bodies"

        planet_q = [
            [0.88, -0.04,  0.03],
            [1.10,  0.02, -0.02],
            [1.02, -0.15, -0.03],
            [0.97,  0.13,  0.04],
        ]
        planet_v = [
            [ 0.40, 1.05, -0.02],
            [-0.35, 0.95,  0.04],
            [ 0.03, 1.40,  0.02],
            [-0.02, 0.60, -0.03],
        ]

    else:
        raise ValueError(f"Unknown case: {name}")

    n_planets = len(planet_q)
    m = np.array([1.0] + [2e-5] * n_planets)

    q = np.vstack(([0.0, 0.0, 0.0], planet_q))
    v = np.vstack(([0.0, 0.0, 0.0], planet_v))

    # Transform to the center-of-mass frame.
    q -= np.sum(m[:, None] * q, axis=0) / np.sum(m)
    v -= np.sum(m[:, None] * v, axis=0) / np.sum(m)

    return title, q, v, m, 1.0


def pair_distances(positions, pairs):
    """Return an array with shape (number of output times, number of pairs)."""
    return np.column_stack([
        np.linalg.norm(
            positions[:, j] - positions[:, i],
            axis=1,
        )
        for i, j in pairs
    ])


def energy_series(positions, velocities, m, G):
    """Compute total energy from saved outputs without integrating again."""
    kinetic = 0.5 * np.sum(
        m[None, :, None] * velocities**2,
        axis=(1, 2),
    )

    potential = np.zeros(len(positions))

    for i, j in combinations(range(len(m)), 2):
        distance = np.linalg.norm(
            positions[:, j] - positions[:, i],
            axis=1,
        )
        potential -= G * m[i] * m[j] / distance

    return kinetic + potential


def error_series(actual, reference):
    """Return the maximum planet vector error at each output time."""
    return np.max(
        np.linalg.norm(
            actual[:, 1:] - reference[:, 1:],
            axis=2,
        ),
        axis=1,
    )


def run_forward_with_levels(module, q, v, m, G, times, dt, radii):
    """Record executed kick levels during the original forward integration.

    Each row summarizes one main step, including all recursive substeps.
    Record the maximum visited depth; optimized may use different particle depths.
    A pair level of -1 means no nonzero-weight kick was executed for that pair.
    """
    pairs = list(combinations(range(1, len(m)), 2))
    recursion_depths = []
    pair_levels = []
    kick_counts = []
    original_step = module.symba_step
    original_kick = module.level_wise_interaction_kick
    current_depth = -1
    current_counts = None
    pair_indices = {pair: index for index, pair in enumerate(pairs)}
    particle_kepler_calls = 0
    original_drift = module.kepler_drift

    def recorded_drift(Q, P, masses, drift_dt, G=1.0):
        nonlocal particle_kepler_calls
        # Count individual Kepler solves, not Python calls on different batch sizes.
        if drift_dt != 0.0:
            particle_kepler_calls += len(masses) - 1
        return original_drift(Q, P, masses, drift_dt, G)

    def recorded_kick(Q, P, masses, *args):
        nonlocal current_depth
        if module is optimized:
            active_pairs, kick_dt, level, shell_radii, *extra = args
        else:
            kick_dt, level, shell_radii, *extra = args
            active_pairs = pairs
        current_depth = max(current_depth, level)
        for i, j in active_pairs:
            distance = np.linalg.norm(Q[j] - Q[i])
            weight = fixed.level_weights(distance, shell_radii)[level]
            if kick_dt != 0.0 and weight != 0.0:
                current_counts[pair_indices[(i, j)], level] += 1
        return original_kick(Q, P, masses, *args)

    def recorded_step(Q, P, masses, step_dt, shell_radii, G=1.0):
        nonlocal current_depth, current_counts
        current_depth = -1
        current_counts = np.zeros((len(pairs), len(radii)), dtype=int)
        original_step(Q, P, masses, step_dt, shell_radii, G)
        recursion_depths.append(current_depth)
        deepest = np.max(
            np.where(current_counts > 0, np.arange(len(radii)), -1), axis=1
        )
        pair_levels.append(deepest)
        kick_counts.append(current_counts)

    # Patch the module's own binding, including the imported adaptive kick.
    # The original flows still run unchanged; recording does not advance time.
    with (
        patch.object(module, "symba_step", new=recorded_step),
        patch.object(module, "kepler_drift", new=recorded_drift),
        patch.object(module, "level_wise_interaction_kick", new=recorded_kick),
    ):
        result = run_forward(module, q, v, m, G, times, dt, radii)

    result["particle_kepler_calls"] = particle_kepler_calls
    result["recursion_depths"] = np.array(recursion_depths)
    result["pair_levels"] = np.array(pair_levels)
    result["pair_kick_counts"] = np.array(kick_counts)
    result["step_edges"] = times[0] + np.arange(len(pair_levels) + 1) * dt
    np.testing.assert_allclose(result["step_edges"][-1], times[-1], atol=1e-12)
    assert np.all(result["pair_levels"] <= result["recursion_depths"][:, None])
    return result


def test_recorded_levels():
    """Check recorded depths and ensure instrumentation preserves the solution."""
    m = np.array([1.0, 2e-5, 2e-5])
    v = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.0]])
    dt = 1e-6
    times = np.array([0.0, dt, 2 * dt])
    radii = fixed.make_radii(0.15, 4)

    for separation in [1.0, 0.02]:
        q = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 + separation, 0.0, 0.0],
        ])
        for module in [fixed, adaptive, optimized]:
            baseline = run_forward(module, q, v, m, 1.0, times, dt, radii)
            recorded = run_forward_with_levels(
                module, q, v, m, 1.0, times, dt, radii
            )
            np.testing.assert_array_equal(recorded["q"], baseline["q"])
            np.testing.assert_array_equal(recorded["v"], baseline["v"])
            assert recorded["kepler_calls"] == baseline["kepler_calls"]
            batch_size = 1 if module is optimized else len(m) - 1
            assert recorded["particle_kepler_calls"] == baseline["kepler_calls"] * batch_size

            expected_depth = 3 if module is fixed or separation == 0.02 else 0
            expected_pair_level = 3 if separation == 0.02 else 0
            np.testing.assert_array_equal(recorded["recursion_depths"], [expected_depth] * 2)
            np.testing.assert_array_equal(recorded["pair_levels"], [[expected_pair_level]] * 2)
            # There are two kicks per invocation at the active level.
            expected_kicks = 54 if separation == 0.02 else 2
            np.testing.assert_array_equal(
                recorded["pair_kick_counts"][:, 0, expected_pair_level],
                [expected_kicks] * 2,
            )
            np.testing.assert_allclose(recorded["step_edges"], times, rtol=0, atol=1e-15)

    # Only AB enters deep recursion; the distant particle stays at level zero.
    # This catches accidental recording of pairs excluded by the optimized path.
    m = np.array([1.0, 2e-5, 2e-5, 2e-5])
    q = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [1.02, 0.0, 0.0], [3.0, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                  [0.0, 0.9, 0.0], [0.0, 0.6, 0.0]])
    baseline = run_forward(optimized, q, v, m, 1.0, times, dt, radii)
    recorded = run_forward_with_levels(optimized, q, v, m, 1.0, times, dt, radii)
    np.testing.assert_array_equal(recorded["q"], baseline["q"])
    np.testing.assert_array_equal(recorded["v"], baseline["v"])
    np.testing.assert_array_equal(recorded["pair_levels"], [[3, 0, 0]] * 2)
    assert not recorded["pair_kick_counts"][:, 1:, 1:].any()
    assert recorded["particle_kepler_calls"] == 2 * (27 + 27 + 1)
    print("Executed-level recording test passed")


def plot_case(
    name, title, times, q, v, m, G,
    radii, q_ref, v_ref, pairs, results,
):
    """Use the same six-panel layout for any of the three SyMBA methods."""
    colors = {"fixed": "#D97706", "adaptive": "#2563EB", "optimized": "#059669"}
    styles = {"fixed": "--", "adaptive": "-.", "optimized": ":"}
    names = list(results)
    planet_colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
    is_3d = name in ["spatial", "smooth"]
    ax_orbit = fig.add_subplot(grid[0, 0], projection="3d" if is_3d else None)
    ax_distance = fig.add_subplot(grid[0, 1])
    ax_error = fig.add_subplot(grid[0, 2])
    ax_energy = fig.add_subplot(grid[1, 0])
    ax_levels = fig.add_subplot(grid[1, 1])
    ax_calls = fig.add_subplot(grid[1, 2])
    dt = results[names[0]]["step_edges"][1] - times[0]
    fig.suptitle(f"{title}\ndt={dt:g}, levels={len(radii)}, G={G:g}", fontsize=15)

    # 1. Planet colors identify trajectories; line styles identify methods.
    ref_helio = q_ref - q_ref[:, 0:1]
    for i in range(1, len(m)):
        color = planet_colors[(i - 1) % len(planet_colors)]
        ref = ref_helio[:, i]
        coordinates = [ref[:, k] for k in range(3 if is_3d else 2)]
        ax_orbit.plot(*coordinates, color=color, label=f"P{i}")
        for method, result in results.items():
            track = result["q"][:, i] - result["q"][:, 0]
            coordinates = [track[:, k] for k in range(3 if is_3d else 2)]
            ax_orbit.plot(*coordinates, styles[method], color=color, linewidth=1)
    method_handles = [Line2D([], [], color="black", label="IAS15")]
    method_handles += [Line2D([], [], color="black", ls=styles[n], label=n) for n in names]
    handles, labels = ax_orbit.get_legend_handles_labels()
    ax_orbit.legend(handles=handles + method_handles, fontsize=7, ncol=2)
    ax_orbit.set(title="Heliocentric tracks (curves may overlap)", xlabel="x", ylabel="y")
    if is_3d:
        ax_orbit.set_zlabel("z")
        extent = np.ptp(ref_helio[:, 1:].reshape(-1, 3), axis=0)
        ax_orbit.set_box_aspect(np.maximum(extent, 1e-8))
    else:
        ax_orbit.set_aspect("equal", adjustable="datalim")

    # 2. Compare the same closest pairs for every method.
    d_ref = pair_distances(q_ref, pairs)
    selected = np.argsort(np.min(d_ref, axis=0))[:3]
    if name == "many_particles":
        # Show the closest pair in each group instead of three from one group.
        selected = [
            min(
                [index for index, (i, j) in enumerate(pairs) if i in group and j in group],
                key=lambda index: d_ref[:, index].min(),
            )
            for group in [(1, 2, 3), (4, 5, 6), (7, 8)]
        ]
    for index in selected:
        i, j = pairs[index]
        line, = ax_distance.semilogy(times, d_ref[:, index], label=f"P{i}-P{j}")
        for method, result in results.items():
            distances = pair_distances(result["q"], pairs)
            ax_distance.semilogy(times, distances[:, index], styles[method],
                                 color=line.get_color(), linewidth=1)
    for level, radius in enumerate(radii):
        ax_distance.axhline(radius, color="gray", linestyle=":", linewidth=0.8)
        ax_distance.text(times[-1], radius, f" R{level}", fontsize=8, va="center")
    ax_distance.set(title="Closest pairs · line styles as in orbit panel", xlabel="Time", ylabel="Pair distance")
    if name == "many_particles":
        ax_distance.set_title("Closest pair in each group · styles as in orbit panel")
    ax_distance.legend(fontsize=8)

    # 3. Normalize errors by initial heliocentric position and velocity scales.
    q_scale = np.max(np.linalg.norm(q[1:] - q[0], axis=1))
    v_scale = np.max(np.linalg.norm(v[1:] - v[0], axis=1))
    for method, result in results.items():
        for data, reference, scale, style, label in [
            (result["q"], q_ref, q_scale, "-", "position"),
            (result["v"], v_ref, v_scale, "--", "velocity"),
        ]:
            error = error_series(data, reference) / scale
            ax_error.semilogy(times[1:], np.maximum(error[1:], 1e-18), style,
                              color=colors[method], label=f"{method}: {label}")
    ax_error.set(title="Errors against IAS15", xlabel="Time", ylabel="Error / initial characteristic scale")
    ax_error.legend(fontsize=8)

    # 4. Signed energy errors at output times.
    initial_energy = energy_series(q[None], v[None], m, G)[0]
    for method, result in results.items():
        energy = energy_series(result["q"], result["v"], m, G)
        ax_energy.plot(times, (energy - initial_energy) / abs(initial_energy),
                       color=colors[method], label=method)
    ax_energy.axhline(0.0, color="gray", linewidth=0.8)
    ax_energy.set(title="Energy error at output times", xlabel="Time", ylabel="(E - E0) / |E0|")
    ax_energy.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_energy.legend(fontsize=8)

    # 5. Actual execution, summarized over all substeps in each main step.
    # Optimized recursion depth is the maximum across its particle groups.
    rows, labels = [], []
    shown_pairs = list(range(len(pairs)))
    if len(pairs) > 12:
        # Keep large-system plots readable; all pairs remain in the saved arrays.
        shown_pairs = [
            index for index in range(len(pairs))
            if any(np.any(result["pair_levels"][:, index] > 0) for result in results.values())
        ]
    for method, result in results.items():
        rows.append(result["recursion_depths"])
        labels.append(f"{method}: max recursion")
        for index in shown_pairs:
            i, j = pairs[index]
            rows.append(result["pair_levels"][:, index])
            labels.append(f"{method}: P{i}-P{j}")
    cmap = plt.get_cmap("viridis", len(radii)).copy()
    cmap.set_under("#eeeeee")
    norm = BoundaryNorm(np.arange(len(radii) + 1) - 0.5, cmap.N)
    mesh = ax_levels.pcolormesh(results[names[0]]["step_edges"],
                                np.arange(len(rows) + 1) - 0.5, np.array(rows),
                                shading="flat", cmap=cmap, norm=norm)
    ax_levels.set_yticks(np.arange(len(rows)))
    ax_levels.set_yticklabels(labels, fontsize=7)
    ax_levels.invert_yaxis()
    for block in range(1, len(names)):
        ax_levels.axhline(block * (len(shown_pairs) + 1) - 0.5, color="white", linewidth=2)
    level_title = "Actual execution: max level per main step\nPair rows: nonzero kicks; gray: none"
    if len(shown_pairs) != len(pairs):
        level_title += f"\nShowing {len(shown_pairs)}/{len(pairs)} pairs with kicks above level 0"
    ax_levels.set(title=level_title,
                  xlabel="Time (main-step intervals)")
    fig.colorbar(mesh, ax=ax_levels, ticks=np.arange(len(radii)), label="Level")

    # 6. Count individual particle solves for a fair comparison across methods.
    counts = [results[n]["particle_kepler_calls"] for n in names]
    bars = ax_calls.bar(names, counts, color=[colors[n] for n in names])
    ax_calls.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=4)
    ax_calls.set_ylim(0, max(counts) * 1.2)
    ax_calls.set(title="Individual Kepler solves\nOne particle advanced = one solve", ylabel="Solves")
    for ax in [ax_distance, ax_error, ax_energy]:
        ax.grid(alpha=0.2)
        ax.set_xlim(times[0], times[-1])
    return fig


def run_case(name):
    title, q, v, m, G = make_case(name)

    radii = fixed.make_radii(
        R_OUTER, TOTAL_LEVELS, ratio=2.08
    )

    n_steps = round(T_END / DT)
    n_outputs = round(T_END / OUTPUT_INTERVAL)

    assert np.isclose(n_steps * DT, T_END)
    assert np.isclose(n_outputs * OUTPUT_INTERVAL, T_END)

    times = np.arange(n_outputs + 1) * OUTPUT_INTERVAL
    pairs = list(combinations(range(1, len(m)), 2))

    print(f"\n{'=' * 60}")
    print(title)
    print("radii:", radii)
    print("Computing IAS15 reference...", flush=True)

    q_ref, v_ref = reference_trajectory(
        q, v, m, G, times
    )

    d_ref = pair_distances(q_ref, pairs)

    print("\nMinimum sampled reference distance for every pair:")
    for pair, distance in zip(pairs, np.min(d_ref, axis=0)):
        print(f"  {pair}: {distance:.6e}")

    print("\nIntegrating fixed and adaptive...", flush=True)
    results = {}

    for method, module in [
        ("fixed", fixed),
        ("adaptive", adaptive),
    ]:
        print(f"\nRunning {method}...", flush=True)

        result = run_forward_with_levels(
            module, q, v, m, G, times, DT, radii
        )
        results[method] = result

        distances = pair_distances(result["q"], pairs)

        print(
            "  Maximum position error:",
            max_planet_error(result["q"], q_ref),
        )
        print(
            "  Maximum velocity error:",
            max_planet_error(result["v"], v_ref),
        )
        print("  Maximum relative energy error:", result["energy_error"])
        print("  Maximum relative angular error:", result["angular_error"])
        print("  Minimum sampled pair distance:", np.min(distances))
        print("  Kepler drift calls:", result["kepler_calls"])

        if RUN_REVERSE:
            reverse_q, reverse_v = run_reverse(
                module, result, q, v, m, G,
                DT, n_steps, radii,
            )
            print("  Reverse position error:", reverse_q)
            print("  Reverse velocity error:", reverse_v)

    expected_calls = n_steps * 3 ** (TOTAL_LEVELS - 1)
    assert results["fixed"]["kepler_calls"] == expected_calls

    print("\nAdaptive vs fixed:")
    print(
        "  Maximum position difference:",
        max_planet_error(
            results["adaptive"]["q"],
            results["fixed"]["q"],
        ),
    )
    print(
        "  Maximum velocity difference:",
        max_planet_error(
            results["adaptive"]["v"],
            results["fixed"]["v"],
        ),
    )

    fig = plot_case(
        name, title, times, q, v, m, G,
        radii, q_ref, v_ref, pairs, results,
    )

    output_dir = Path(__file__).resolve().parent / "validation_output"
    output_dir.mkdir(exist_ok=True)
    figure_path = output_dir / f"multiple_encounter_{name}.png"
    fig.savefig(figure_path, dpi=160)
    # Save the measured arrays so the figure can be audited without rerunning.
    np.savez_compressed(
        output_dir / f"multiple_encounter_{name}_levels.npz",
        pairs=np.array(pairs),
        **{
            f"{method}_{key}": result[key]
            for method, result in results.items()
            for key in ["step_edges", "recursion_depths", "pair_levels", "pair_kick_counts"]
        },
    )
    print("Saved figure:", figure_path)
    return results


def test_multiple_encounters():
    test_recorded_levels()
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "lines.linewidth": 1.5,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    for name in CASES_TO_RUN:
        run_case(name)




# Audit forces omitted when adaptive recursion stops early.


def test_skipped_kicks(dt=0.005, t_end=32.0):
    q, v, m, G = make_initial_state()

    total_levels = 4

    radii = fixed.make_radii(
        r_outer=0.3,
        total_levels=total_levels,
        ratio=2.08,
    )

    n_steps = round(t_end / dt)
    assert np.isclose(n_steps * dt, t_end)

    Q, P = fixed.cartesian_to_dh(
        q, m[:, None] * v, m
    )

    original_need = adaptive.need_deeper_level
    original_kick = fixed.level_wise_interaction_kick

    stats = {
        "checked_stops": 0,
        "stops_with_nonzero_kicks": 0,
        "checked_kicks": 0,
        "nonzero_pair_impulses": 0,
        "max_pair_delta_v": 0.0,
    }

    examples = []
    current_step = 0

    def checked_need(Q, P, m, dt, level, radii, G=1.0):
        decision = original_need(
            Q, P, m, dt, level, radii, G
        )

        if decision:
            return True

        if level >= len(radii) - 1:
            return False

        stats["checked_stops"] += 1

        Q_check = Q.copy()
        P_check = P.copy()

        local = {
            "nonzero_pairs": 0,
            "first_event": None,
        }

        def inspected_kick(
            Qk, Pk, masses, kick_dt, kick_level, shell_radii, G=1.0
        ):
            stats["checked_kicks"] += 1

            for i in range(1, len(masses)):
                for j in range(i + 1, len(masses)):
                    r = Qk[j] - Qk[i]
                    r2 = np.dot(r, r)

                    if r2 == 0.0:
                        raise ValueError(
                            "Two planets occupy the same position."
                        )

                    distance = np.sqrt(r2)
                    weight = fixed.level_weights(
                        distance, shell_radii
                    )[kick_level]

                    force = (
                        G * masses[i] * masses[j]
                        * r / (r2 * distance)
                    )
                    impulse = kick_dt * force * weight

                    if np.any(impulse != 0.0):
                        local["nonzero_pairs"] += 1
                        stats["nonzero_pair_impulses"] += 1

                        delta_v = max(
                            np.linalg.norm(impulse / masses[i]),
                            np.linalg.norm(impulse / masses[j]),
                        )

                        stats["max_pair_delta_v"] = max(
                            stats["max_pair_delta_v"],
                            delta_v,
                        )

                        if local["first_event"] is None:
                            local["first_event"] = {
                                "kick_level": kick_level,
                                "pair": (i, j),
                                "distance": distance,
                                "weight": weight,
                                "delta_v": delta_v,
                            }

            original_kick(
                Qk, Pk, masses, kick_dt,
                kick_level, shell_radii, G,
            )

        with patch.object(
            fixed,
            "level_wise_interaction_kick",
            new=inspected_kick,
        ):
            for _ in range(3):
                fixed.evolve_level(
                    Q_check,
                    P_check,
                    m,
                    dt / 3,
                    level + 1,
                    radii,
                    G,
                )

        if local["nonzero_pairs"] > 0:
            stats["stops_with_nonzero_kicks"] += 1

            if len(examples) < 5:
                examples.append({
                    "main_step": current_step,
                    "stop_level": level,
                    "interval": dt,
                    **local["first_event"],
                })
        return False

    print("dt:", dt)
    print("radii:", radii)
    print("Auditing skipped kicks...", flush=True)

    with patch.object(
        adaptive,
        "need_deeper_level",
        new=checked_need,
    ):
        for step in range(n_steps):
            current_step = step

            adaptive.symba_step(
                Q, P, m, dt, radii, G
            )

            if (step + 1) % 1000 == 0:
                print(
                    f"Completed {step + 1}/{n_steps} main steps",
                    flush=True,
                )

    assert np.isfinite(Q).all()
    assert np.isfinite(P).all()

    print("\nAUDIT RESULTS")
    print("Checked early stops:", stats["checked_stops"])
    print("Checked deeper kicks:", stats["checked_kicks"])
    print(
        "Early stops containing nonzero deeper kicks:",
        stats["stops_with_nonzero_kicks"],
    )
    print(
        "Nonzero pair impulses in diagnostic branches:",
        stats["nonzero_pair_impulses"],
    )
    print(
        "Maximum single-pair velocity increment:",
        stats["max_pair_delta_v"],
    )

    if stats["checked_stops"] == 0:
        print("No early stops occurred; this run did not test skipping.")
    elif stats["stops_with_nonzero_kicks"] == 0:
        print(
            "All inspected deeper pair impulses were zero "
            "for this forward trajectory."
        )
    else:
        print("\nFirst affected early stops:")

        for event in examples:
            step = event["main_step"]
            print(
                f"\nMain-step interval: "
                f"[{step * dt:.6f}, {(step + 1) * dt:.6f}]"
            )
            print("Stopped at level:", event["stop_level"])
            print("Skipped interval length:", event["interval"])
            print("First nonzero kick level:", event["kick_level"])
            print("Particle pair:", event["pair"])
            print("Distance at that kick:", event["distance"])
            print("Force weight:", event["weight"])
            print("Pair velocity increment:", event["delta_v"])


def main():
    # Comment out individual tests when only one example is needed.
    test_adaptive_step()
    test_multiple_encounters()
    test_skipped_kicks()
    plt.show()


if __name__ == "__main__":
    main()
