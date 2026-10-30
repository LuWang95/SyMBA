"""Run each validation example and plot fixed, adaptive, and optimized together."""
from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import SyMBA_fixed_depth as fixed
import SyMBA_adaptive_depth as adaptive
import SyMBA_adaptive_depth_optimized as optimized
from symba_validation import initial_state, reference, integrate, diagnostics, errors
from test_adaptive import (
    plot_case,
    run_forward_with_levels,
    test_recorded_levels,
)


DT = 0.00125
T_END = 0.5
OUTPUT_INTERVAL = 0.005
TOTAL_LEVELS = 4
R_OUTER = 0.15
OUTPUT_DIR = Path(__file__).resolve().parent / "validation_output"
METHODS = [("fixed", fixed), ("adaptive", adaptive), ("optimized", optimized)]


def run_case(name, title):
    q, v, m = initial_state(name)
    G = 1.0
    radii = fixed.make_radii(R_OUTER, TOTAL_LEVELS)
    n_outputs = round(T_END / OUTPUT_INTERVAL)
    times = np.arange(n_outputs + 1) * OUTPUT_INTERVAL
    pairs = list(combinations(range(1, len(m)), 2))
    np.testing.assert_allclose(times[-1], T_END, rtol=0, atol=1e-12)

    print(f"\n{title}", flush=True)
    print("Computing IAS15 reference...", flush=True)
    q_ref, v_ref = reference(q, v, m, times)
    if name == "many_particles":
        # Verify that all three intended groups really encounter on the reference.
        assert len(m) == 11
        for group in [(1, 2, 3), (4, 5, 6), (7, 8)]:
            minimum = min(
                np.linalg.norm(q_ref[:, i] - q_ref[:, j], axis=1).min()
                for i, j in combinations(group, 2)
            )
            print(f"Group {group}: minimum sampled distance={minimum:.6e}")
            assert minimum < 0.075, group
        for i in [9, 10]:
            minimum = min(
                np.linalg.norm(q_ref[:, i] - q_ref[:, j], axis=1).min()
                for j in range(1, len(m)) if i != j
            )
            assert minimum > R_OUTER, ("Spectator entered an encounter", i)
    results, report = {}, {}

    for method, module in METHODS:
        print(f"Running {method}...", flush=True)
        result = run_forward_with_levels(module, q, v, m, G, times, DT, radii)
        results[method] = result
        metrics = diagnostics(result["q"], result["v"], m)
        position_error = float(errors(result["q"], q_ref).max())
        velocity_error = float(errors(result["v"], v_ref).max())
        report[method] = {
            "max_position_error": position_error,
            "max_velocity_error": velocity_error,
            "max_relative_energy_error_at_outputs": float(metrics["energy"].max()),
            "max_relative_angular_error_at_outputs": float(metrics["angular"].max()),
            "minimum_sampled_pair_distance": float(metrics["distance"].min()),
            "individual_kepler_solves": result["particle_kepler_calls"],
        }
        print("Maximum position error against IAS15:", position_error)
        print("Maximum velocity error against IAS15:", velocity_error)
        print("Maximum relative energy error at outputs:", metrics["energy"].max())
        print("Maximum relative angular error at outputs:", metrics["angular"].max())
        print("Individual Kepler solves:", result["particle_kepler_calls"])

        # These are short-time fixture-specific budgets, not universal guarantees.
        assert position_error < 1e-8, (name, method, position_error)
        assert velocity_error < 1e-7, (name, method, velocity_error)
        assert metrics["energy"].max() < 1e-7, (name, method)
        assert metrics["angular"].max() < 2e-12, (name, method)

    expected_solves = round(T_END / DT) * 3 ** (TOTAL_LEVELS - 1) * (len(m) - 1)
    assert results["fixed"]["particle_kepler_calls"] == expected_solves
    if name != "smooth":
        assert diagnostics(q_ref, v_ref, m)["distance"].min() < 0.075

    fig = plot_case(name, title, times, q, v, m, G, radii, q_ref, v_ref, pairs, results)
    OUTPUT_DIR.mkdir(exist_ok=True)
    figure_path = OUTPUT_DIR / f"validation_{name}.png"
    fig.savefig(figure_path, dpi=160)
    (OUTPUT_DIR / f"validation_{name}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        OUTPUT_DIR / f"validation_{name}_levels.npz",
        pairs=np.array(pairs),
        **{
            f"{method}_{key}": result[key]
            for method, result in results.items()
            for key in ["step_edges", "recursion_depths", "pair_levels", "pair_kick_counts"]
        },
    )
    print("Saved figure:", figure_path)
    print(f"{name} validation passed")
    return report


def test_shared_encounters():
    # Three interacting planets and one distant spectator.
    return run_case("shared", "Shared-particle encounters with a distant planet")


def test_independent_encounters():
    # Two encounter groups on opposite sides of the central star.
    return run_case("independent", "Two independent encounter groups")


def test_spatial_encounters():
    # Encounters with nonzero position and velocity components in all axes.
    return run_case("spatial", "Three-dimensional encounters")


def test_smooth_orbits():
    # A no-encounter control, shown with the same six panels.
    return run_case("smooth", "Smooth orbits without close encounters")


def test_many_particle_encounters():
    # P1-P3 and P4-P6 are shared encounters; P7-P8 form a separate pair.
    # P9 and P10 stay far away and should not require deep optimized drifts.
    return run_case("many_particles", "10 planets + star: three encounter groups and two distant planets")


def test_smooth_convergence():
    # Order is checked only on the smooth fixture, not across close encounters.
    q, v, m = initial_state("smooth")
    times = np.array([0.0, 1.0])
    q_ref, v_ref = reference(q, v, m, times)
    steps = np.array([0.04, 0.02, 0.01])
    colors = {"fixed": "#D97706", "adaptive": "#2563EB", "optimized": "#059669"}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    report = {}
    for method, module in METHODS:
        residuals = []
        for dt in steps:
            q_test, v_test = integrate(q, v, m, times, dt, module=module)
            residuals.append(float(max(errors(q_test, q_ref)[-1], errors(v_test, v_ref)[-1])))
        ratios = np.array(residuals[:-1]) / residuals[1:]
        print(f"{method} step-halving error ratios:", ratios)
        assert np.all((ratios > 3.5) & (ratios < 4.5)), (method, residuals, ratios)
        report[method] = {"dt": steps.tolist(), "errors": residuals, "halving_ratios": ratios.tolist()}
        ax.loglog(steps, residuals, "o-", color=colors[method], label=method)
    scale = report["optimized"]["errors"][0]
    ax.loglog(steps, scale * (steps / steps[0]) ** 2, "k--", label="Second order")
    ax.set(title="Smooth-orbit convergence", xlabel="dt",
           ylabel="Final max(position error, velocity error)")
    ax.legend()
    ax.grid(alpha=0.2)
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / "validation_smooth_convergence.png", dpi=160)
    (OUTPUT_DIR / "validation_smooth_convergence_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Smooth convergence validation passed")
    return report


def main():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "lines.linewidth": 1.5,
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    test_recorded_levels()
    # Comment out individual examples to run only the cases you need.
    test_shared_encounters()
    test_independent_encounters()
    test_spatial_encounters()
    test_many_particle_encounters()
    test_smooth_orbits()
    test_smooth_convergence()
    plt.show()


if __name__ == "__main__":
    main()
