import numpy as np
import rebound

import SyMBA_fixed_depth as symba


def make_initial_state():
    sim = rebound.Simulation()
    sim.G = 1.0

    sim.add(m=1.0)
    sim.add(m=1e-3, a=1.0)
    sim.add(m=5e-3, a=1.25)
    sim.move_to_com()

    m = np.array([p.m for p in sim.particles])
    q = np.array([[p.x, p.y, p.z] for p in sim.particles])
    v = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])

    return q, v, m, sim.G


def reference_trajectory(q, v, m, G, times):
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "ias15"

    for i in range(len(m)):
        sim.add(m=m[i],x=q[i, 0],y=q[i, 1],z=q[i, 2],vx=v[i, 0],vy=v[i, 1],vz=v[i, 2],)

    positions = []
    velocities = []

    for t in times:
        sim.integrate(float(t), exact_finish_time=1)

        positions.append([[p.x, p.y, p.z] for p in sim.particles])
        velocities.append([[p.vx, p.vy, p.vz] for p in sim.particles])

    return np.array(positions), np.array(velocities)


def run_symba(q, v, m, G, times, dt, total_levels):
    radii = symba.make_radii(r_outer=0.3,total_levels=total_levels,ratio=2.08,)

    output_interval = times[1] - times[0]
    steps_per_output = round(output_interval / dt)
    Q, P = symba.cartesian_to_dh(q, m[:, None] * v, m)

    energy_initial = symba.total_energy(Q, P, m, G)

    q_initial, p_initial = symba.dh_to_cartesian(Q, P, m)
    angular_initial = np.sum(np.cross(q_initial, p_initial), axis=0)
    angular_scale = np.linalg.norm(angular_initial)
    assert angular_scale > 0.0

    positions = [q_initial.copy()]
    velocities = [p_initial / m[:, None]]

    max_energy_error = 0.0
    max_angular_error = 0.0
    min_distance = np.linalg.norm(Q[2] - Q[1])

    for _ in times[1:]:
        for _ in range(steps_per_output):
            symba.symba_step(Q, P, m, dt, radii, G)
            energy = symba.total_energy(Q, P, m, G)
            max_energy_error = max(
                max_energy_error,
                abs(energy - energy_initial) / abs(energy_initial),
            )

            q_now, p_now = symba.dh_to_cartesian(Q, P, m)
            angular = np.sum(np.cross(q_now, p_now), axis=0)

            max_angular_error = max(max_angular_error,np.linalg.norm(angular - angular_initial)/ angular_scale,)

            min_distance = min(min_distance,np.linalg.norm(Q[2] - Q[1]),)

        positions.append(q_now.copy())
        velocities.append(p_now / m[:, None])

    return {
        "q": np.array(positions),
        "v": np.array(velocities),
        "energy_error": max_energy_error,
        "angular_error": max_angular_error,
        "min_distance": min_distance,
        "radii": radii,
    }


def max_planet_error(actual, reference):
    return np.max(np.linalg.norm(actual[:, 1:] - reference[:, 1:],axis=2,))


def main():
    q, v, m, G = make_initial_state()

    t_end = 32.0
    output_interval = 0.02
    n_outputs = round(t_end / output_interval)
    times = np.arange(n_outputs + 1) * output_interval

    print("Computing IAS15 reference with default settings...", flush=True)

    q_ref, v_ref = reference_trajectory(q, v, m, G, times)

    reference_distances = np.linalg.norm(q_ref[:, 2] - q_ref[:, 1],axis=1,)
    print("IAS15 minimum sampled distance:",np.min(reference_distances),)


    results_cached = {}
    def evaluate(dt, levels):
        key = (dt, levels)

        if key not in results_cached:
            print(
                f"Running dt={dt}, levels={levels}...",
                flush=True,
            )

            result = run_symba(q, v, m, G, times, dt, levels)
            result["q_error"] = max_planet_error(result["q"], q_ref)
            result["v_error"] = max_planet_error(result["v"], v_ref)
            results_cached[key] = result

        return results_cached[key]

    print("\nSTEP-SIZE CONVERGENCE: fixed levels=4")

    previous = None

    for dt in [0.02, 0.01, 0.005, 0.0025]:
        result = evaluate(dt, levels=4)

        print(
            f"dt={dt:.4f}  "
            f"q_err={result['q_error']:.6e}  "
            f"v_err={result['v_error']:.6e}  "
            f"dE={result['energy_error']:.6e}  "
            f"dL={result['angular_error']:.6e}  "
            f"r_min={result['min_distance']:.6e}"
        )

        if previous is not None:
            for key in ["q_error", "v_error"]:
                if result[key] > 0 and previous[key] > 0:
                    ratio = previous[key] / result[key]
                    order = np.log2(ratio)
                    print(
                        f"  {key}: ratio={ratio:.3f}, "
                        f"observed order={order:.3f}"
                    )

        previous = result

    print("\nDEPTH CONVERGENCE: fixed dt=0.005")

    for levels in [3, 4, 5]:
        result = evaluate(dt=0.005, levels=levels)

        deepest_dt = 0.005 / 3 ** (levels - 1)

        print(
            f"levels={levels}  "
            f"deepest_dt={deepest_dt:.6e}  "
            f"q_err={result['q_error']:.6e}  "
            f"v_err={result['v_error']:.6e}  "
            f"dE={result['energy_error']:.6e}  "
            f"dL={result['angular_error']:.6e}  "
            f"r_min={result['min_distance']:.6e}"
        )
        print("  radii:", result["radii"])


if __name__ == "__main__":
    main()