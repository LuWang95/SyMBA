import time
import rebound
import numpy as np
import matplotlib.pyplot as plt


def make_simulation():
    sim = rebound.Simulation()
    sim.add(m=1, name="star")
    star = sim.particles["star"]
    sim.add(
        m=1e-3,
        a=1.0,
        e=0.2,
        primary=star,
        name="planet1",
    )
    sim.add(
        m=2e-3,
        a=1.8,
        e=0.1,
        primary=star,
        name="planet2",
    )
    sim.move_to_com()
    return sim

def run_simulation(sim_cpy, integrator, end_time, num_intervals, dt=None):
    start_time = time.perf_counter()
    sim_cpy.integrator = integrator
    if dt is not None:
        sim_cpy.dt = dt
    times = np.linspace(0, end_time, num_intervals)
    E0 = sim_cpy.energy()
    L0 = sim_cpy.angular_momentum()
    L0_array = np.array([L0.x,L0.y,L0.z])
    data = {"planet1_x":[],"planet1_y":[],"planet2_x":[],"planet2_y":[],"star_x":[],"star_y":[],"planet1_a":[],
            "planet1_e":[],"planet2_a":[],"planet2_e":[],"energy_error":[],"actual_time":[], "planet_distance":[],"angular_momentum_error":[],"run_time":0}
    for t in times:
        if integrator == "whfast":
            sim_cpy.integrate(t,exact_finish_time=0)
        else:
            sim_cpy.integrate(t)
        data["actual_time"].append(sim_cpy.t)
        star = sim_cpy.particles["star"]
        planet1 = sim_cpy.particles["planet1"]
        planet2 = sim_cpy.particles["planet2"]
        orbit1 = planet1.orbit(primary=star)
        orbit2 = planet2.orbit(primary=star)
        data["planet1_x"].append(planet1.x)
        data["planet1_y"].append(planet1.y)
        data["planet2_x"].append(planet2.x)
        data["planet2_y"].append(planet2.y)
        data["star_x"].append(star.x)
        data["star_y"].append(star.y)
        data["planet1_a"].append(orbit1.a)
        data["planet1_e"].append(orbit1.e)
        data["planet2_a"].append(orbit2.a)
        data["planet2_e"].append(orbit2.e)
        energy = sim_cpy.energy()
        data["energy_error"].append((energy-E0)/abs(E0))
        distance = np.sqrt((planet2.x - planet1.x)**2 + (planet2.y - planet1.y)**2 + (planet2.z - planet1.z)**2)
        data["planet_distance"].append(distance)
        L_array = np.array([sim_cpy.angular_momentum().x,sim_cpy.angular_momentum().y,sim_cpy.angular_momentum().z])
        data["angular_momentum_error"].append(np.linalg.norm(L_array - L0_array)/np.linalg.norm(L0_array))
    data["run_time"] = time.perf_counter() - start_time
    for key in data.keys():
        if isinstance(data[key], list):
            data[key] = np.array(data[key])
    return data

def plot(sim_result):
    trajectory_fig, trajectory_ax = plt.subplots()
    trajectory_ax.plot(sim_result["planet1_x"], sim_result["planet1_y"], label="planet1")
    trajectory_ax.plot(sim_result["planet2_x"], sim_result["planet2_y"], label="planet2")
    trajectory_ax.plot(sim_result["star_x"], sim_result["star_y"], label="star", color="red")
    trajectory_ax.scatter(
        [sim_result["planet1_x"][-1], sim_result["planet2_x"][-1]],
        [sim_result["planet1_y"][-1], sim_result["planet2_y"][-1]],
        color="blue",
    )
    trajectory_ax.set_xlabel("x")
    trajectory_ax.set_ylabel("y")
    trajectory_ax.set_aspect("equal")
    trajectory_ax.legend()

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(sim_result["actual_time"], sim_result["planet1_a"], label="planet1")
    axes[0].plot(sim_result["actual_time"], sim_result["planet2_a"], label="planet2")
    axes[1].plot(sim_result["actual_time"], sim_result["planet1_e"], label="planet1")
    axes[1].plot(sim_result["actual_time"], sim_result["planet2_e"], label="planet2")
    axes[0].set_ylabel("semimajor axis a")
    axes[0].legend()
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("eccentricity e")
    axes[1].legend()

    energy_fig, energy_ax = plt.subplots()
    energy_ax.plot(sim_result["actual_time"], sim_result["energy_error"])
    energy_ax.set_xlabel("time")
    energy_ax.set_ylabel("energy relative error")

    return trajectory_fig, fig, energy_fig

if __name__ == "__main__":
    sim_init = make_simulation()
    sim1 = sim_init.copy()
    sim2 = sim_init.copy()

    sim1_result = run_simulation(sim1, "ias15", 20 * np.pi, 1000)
    sim2_result = run_simulation(sim2, "whfast", 20 * np.pi, 1000, dt=0.01)

    plot(sim1_result)
    plot(sim2_result)
    plt.show()
    print(f"sim1 time: {sim1_result['run_time']}")
    print(f"sim2 time: {sim2_result['run_time']}")
