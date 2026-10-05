import rebound
import numpy as np
import matplotlib.pyplot as plt


def make_simulation():
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    sim.add(m=1.0, name='Sun')

    sim.add(
        m=9.545e-4,
        a=5.203,
        e=0.0484,
        f=0.0,
        name='Jupiter'
    )

    sim.add(
        m=2.858e-4,
        a=9.54,
        e=0.054,
        f=np.pi / 2,
        name='Jupiter'
    )

    sim.move_to_com()
    sim.integrator = "whfast"
    return sim

def run_whfast(dt,end_time=1000,n_outputs = 2000):
    sim = make_simulation()
    sim.dt = dt

    output_times = np.linspace(0.0,end_time,n_outputs)

    E0 = sim.energy()

    actual_times = []
    energy_error = []
    for output_time in output_times:
        sim.integrate(output_time,exact_finish_time=0)
        actual_times.append(output_time)
        energy = sim.energy()
        energy_error.append((E0-energy)/E0)

    return np.array(actual_times), np.array(energy_error)

if __name__ == "__main__":
    sim = make_simulation()
    P = sim.particles["Jupiter"].P
    dts = [P/10, P/20, P/40, P/80, P/160]

    results = {}
    max_errors = []
    for dt in dts:
        times,rel_errors = run_whfast(dt)
        max_errors.append(np.max(np.abs(rel_errors)))
        results[dt] = (times,rel_errors)

    plt.figure(figsize=(8, 5))

    for i in range(len(max_errors) - 1):
        ratio = max_errors[i] / max_errors[i + 1]
        print(
            f"h ratio = {dts[i] / dts[i + 1]:.1f}, "
            f"error ratio = {ratio:.3f}"
        )

    for dt, (times, rel_errors) in results.items():
        plt.plot(
            times,
            rel_errors,
            label=fr"$h=P/{P / dt:.0f}$"
        )

    plt.xlabel("Time [yr]")
    plt.ylabel(r"$(E(t)-E_0)/E_0$")
    plt.legend()
    plt.grid()
    plt.show()

