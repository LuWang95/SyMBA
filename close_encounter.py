import matplotlib.pyplot as plt
import numpy as np
import rebound

def make_sim():
    sim = rebound.Simulation()

    sim.G = 1.0

    # Sun
    sim.add(m=1.0)

    # Jupiter-like planet
    sim.add(
        m=1e-3,
        a=1.0,
        e=0.05,
        name = "Jupiter",
    )

    # Test particle
    sim.add(
        m=1e-5,
        a=1.15,
        e=0.18,
        f=np.pi,
        name = "test_particle",
    )

    sim.move_to_com()
    sim.integrator = "whfast"
    return sim

def run_close_encounter(dt, t_end=100.0, n_outputs=5000):
    sim = make_sim()
    sim.dt = dt

    times = np.linspace(0.0, t_end, n_outputs)

    distances = np.zeros(n_outputs)
    energy_errors = np.zeros(n_outputs)

    E0 = sim.energy()

    for i, t in enumerate(times):
        sim.integrate(t, exact_finish_time=0)

        jupiter = sim.particles["Jupiter"]
        particle = sim.particles["test_particle"]

        r_j = np.array([
            jupiter.x,
            jupiter.y,
            jupiter.z
        ])

        r_p = np.array([
            particle.x,
            particle.y,
            particle.z
        ])

        distances[i] = np.linalg.norm(r_p - r_j)

        E = sim.energy()
        energy_errors[i] = abs((E - E0) / E0)

    return times, distances, energy_errors


if __name__ == "__main__":
    P = 2 * np.pi
    dts = [P / 10, P / 20, P / 40, P / 80, P/640, P/2560]
    results = {}
    for dt in dts:
        results[dt] = run_close_encounter(
            dt,
            t_end=100.0,
            n_outputs=5000
        )

    plt.figure(figsize=(9, 6))

    dt_fine = dts[-1]

    times, distances, errors = results[dt_fine]

    i_enc = np.argmin(distances)
    t_enc = times[i_enc]

    print(t_enc)
    plt.figure(figsize=(9, 6))

    window = 2.0

    for dt, (times, distances, energy_errors) in results.items():
        mask = ((times >= t_enc - window) &(times <= t_enc + window))

        plt.semilogy(
            times[mask],
            energy_errors[mask],
            label=f"dt = {dt:.4f}"
        )

    plt.axvline(t_enc, linestyle="--")

    plt.xlabel("Time")
    plt.ylabel(r"$|\Delta E / E_0|$")
    plt.title("Energy error near encounter")
    plt.legend()

    plt.show()