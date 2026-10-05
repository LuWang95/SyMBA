import time
import rebound
import numpy as np
import matplotlib.pyplot as plt

def make_simulation():
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")

    G = sim.G

    m_star = 1.0
    m1 = 1e-3
    m2 = 1e-3
    m_binary = m1 + m2

    a_outer = 1.0
    a_binary = 0.0125
    e_binary = 0.6

    total_mass = m_star + m_binary

    x_star = -m_binary / total_mass * a_outer
    x_binary_com = m_star / total_mass * a_outer

    outer_relative_velocity = np.sqrt(
        G * total_mass / a_outer
    )

    vy_star = -m_binary / total_mass * outer_relative_velocity
    vy_binary_com = m_star / total_mass * outer_relative_velocity

    binary_distance = a_binary * (1.0 - e_binary)

    binary_relative_velocity = np.sqrt(G * m_binary * (1.0 + e_binary) / (a_binary * (1.0 - e_binary)))

    x1 = x_binary_com - binary_distance / 2.0
    x2 = x_binary_com + binary_distance / 2.0

    vy1 = vy_binary_com - binary_relative_velocity / 2.0
    vy2 = vy_binary_com + binary_relative_velocity / 2.0

    sim.add(
        m=m_star,
        x=x_star,
        vy=vy_star,
        name="star",
    )

    sim.add(
        m=m1,
        x=x1,
        vy=vy1,
        name="planet1",
    )

    sim.add(
        m=m2,
        x=x2,
        vy=vy2,
        name="planet2",
    )

    sim.move_to_com()
    return sim

def run_simulation(
    initial_simulation,
    integrator,
    end_time=100.0,
    num_outputs=10001,
    dt=None,
):
    sim = initial_simulation.copy()
    sim.integrator = integrator

    if dt is not None:
        sim.dt = dt

    initial_energy = sim.energy()

    initial_L = sim.angular_momentum()
    initial_L_array = np.array([
        initial_L.x,
        initial_L.y,
        initial_L.z,
    ])

    requested_times = np.linspace(
        0.0,
        end_time,
        num_outputs,
    )

    data = {
        "time": [],
        "binary_a": [],
        "binary_e": [],
        "binary_distance": [],
        "relative_x": [],
        "relative_y": [],
        "energy_error": [],
        "angular_momentum_error": [],
        "runtime": 0.0,
    }

    start_time = time.perf_counter()

    for requested_time in requested_times:
        if integrator.lower() == "whfast":
            sim.integrate(
                requested_time,
                exact_finish_time=0,
            )
        else:
            sim.integrate(requested_time)

        planet1 = sim.particles["planet1"]
        planet2 = sim.particles["planet2"]

        binary_orbit = planet2.orbit(
            primary=planet1
        )

        dx = planet2.x - planet1.x
        dy = planet2.y - planet1.y
        dz = planet2.z - planet1.z

        distance = np.sqrt(
            dx**2 + dy**2 + dz**2
        )

        current_energy = sim.energy()
        energy_error = (
            current_energy - initial_energy
        ) / abs(initial_energy)

        current_L = sim.angular_momentum()
        current_L_array = np.array([
            current_L.x,
            current_L.y,
            current_L.z,
        ])

        angular_momentum_error = (np.linalg.norm(current_L_array - initial_L_array) / np.linalg.norm(initial_L_array))

        data["time"].append(sim.t)
        data["binary_a"].append(binary_orbit.a)
        data["binary_e"].append(binary_orbit.e)
        data["binary_distance"].append(distance)
        data["relative_x"].append(dx)
        data["relative_y"].append(dy)
        data["energy_error"].append(energy_error)
        data["angular_momentum_error"].append(
            angular_momentum_error
        )

    data["runtime"] = (
        time.perf_counter() - start_time
    )

    for key in data:
        if isinstance(data[key], list):
            data[key] = np.asarray(data[key])

    return data

def print_result_summary(name, result):
    print(name)
    print(
        f"runtime: "
        f"{result['runtime']:.3f} seconds"
    )
    print(
        f"minimum distance: "
        f"{np.min(result['binary_distance']):.8f} AU"
    )
    print(
        f"maximum distance: "
        f"{np.max(result['binary_distance']):.8f} AU"
    )
    print(
        f"binary a range: "
        f"{np.min(result['binary_a']):.8f} to "
        f"{np.max(result['binary_a']):.8f} AU"
    )
    print(
        f"binary e range: "
        f"{np.min(result['binary_e']):.8f} to "
        f"{np.max(result['binary_e']):.8f}"
    )
    print(
        f"maximum energy error: "
        f"{np.max(np.abs(result['energy_error'])):.3e}"
    )
    print(
        f"maximum angular momentum error: "
        f"{np.max(result['angular_momentum_error']):.3e}"
    )
    print()

def plot_results(results):
    colors = {
        "IAS15": "tab:blue",
        "WHFast": "tab:orange",
    }
    fig1, ax1 = plt.subplots(
        figsize=(7, 7)
    )

    for name, result in results.items():
        ax1.plot(
            result["relative_x"],
            result["relative_y"],
            label=name,
            color=colors.get(name),
            linewidth=0.6,
            rasterized=True,
        )

    ax1.set_xlabel(
        r"$x_2-x_1$ [AU]"
    )

    ax1.set_ylabel(
        r"$y_2-y_1$ [AU]"
    )

    ax1.set_title(
        "Binary relative orbit"
    )

    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(alpha=0.2)
    fig1.tight_layout()

    fig2, axes = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True,
    )

    for name, result in results.items():
        time_values = result["time"]

        axes[0].plot(
            time_values,
            result["binary_a"],
            label=name,
            color=colors.get(name),
            linewidth=0.35,
            rasterized=True,
        )

        axes[1].plot(
            time_values,
            result["binary_e"],
            label=name,
            color=colors.get(name),
            linewidth=0.35,
            rasterized=True,
        )

        axes[2].plot(
            time_values,
            result["energy_error"],
            label=name,
            color=colors.get(name),
            linewidth=0.6,
            rasterized=True,
        )

    axes[0].set_ylim(
        0.0,
        0.020,
    )

    axes[1].set_ylim(
        0.0,
        1.0,
    )

    axes[0].axhline(
        0.0125,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="initial value",
    )

    axes[1].axhline(
        0.6,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="initial value",
    )

    axes[0].set_ylabel(
        "Binary Semi-major axis (AU)"
    )

    axes[1].set_ylabel(
        "Binary Eccentricity"
    )

    axes[2].set_ylabel(
        r"$(E-E_0)/|E_0|$"
    )

    axes[2].set_xlabel(
        "Time (yrs)"
    )

    all_end_times = [
        result["time"][-1]
        for result in results.values()
    ]

    axes[2].set_xlim(
        0.0,
        max(all_end_times),
    )

    for axis in axes:
        axis.legend()
        axis.grid(False)

    fig2.tight_layout()

    fig3, axes3 = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    for name, result in results.items():
        axes3[0].plot(
            result["time"],
            result["binary_distance"],
            label=name,
            color=colors.get(name),
            linewidth=0.5,
            rasterized=True,
        )

        angular_error = np.maximum(
            result["angular_momentum_error"],
            1e-18,
        )

        axes3[1].plot(
            result["time"],
            angular_error,
            label=name,
            color=colors.get(name),
            linewidth=0.6,
            rasterized=True,
        )

    initial_periapsis = (
        0.0125 * (1.0 - 0.6)
    )

    initial_apoapsis = (
        0.0125 * (1.0 + 0.6)
    )

    axes3[0].axhline(
        initial_periapsis,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="initial periapsis",
    )

    axes3[0].axhline(
        initial_apoapsis,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="initial apoapsis",
    )

    axes3[0].set_ylabel(
        "Planet separation (AU)"
    )

    axes3[1].set_ylabel(
        r"$|L-L_0|/|L_0|$"
    )

    axes3[1].set_xlabel(
        "Time (yrs)"
    )

    axes3[1].set_yscale("log")

    for axis in axes3:
        axis.legend()
        axis.grid(alpha=0.2)

    fig3.tight_layout()

    return fig1, fig2, fig3


if __name__ == "__main__":
    initial_simulation = make_simulation()

    ias15_result = run_simulation(
        initial_simulation=initial_simulation,
        integrator="ias15",
        end_time=100.0,
        num_outputs=10001,
    )

    whfast_result = run_simulation(
        initial_simulation=initial_simulation,
        integrator="whfast",
        end_time=100.0,
        num_outputs=10001,
        dt=0.01,
    )

    print_result_summary(
        "IAS15",
        ias15_result,
    )

    print_result_summary("whfast", whfast_result)

    plot_results({
        "IAS15": ias15_result,
    })
    plt.show()

    plot_results({
        "whfast": whfast_result,
    })
    plt.show()