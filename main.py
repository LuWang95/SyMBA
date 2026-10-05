import rebound
import numpy as np
import matplotlib.pyplot as plt

sim = rebound.Simulation()

sim.add(m=1.0, hash="star")
star = sim.particles["star"]

sim.add(
    m=1e-3,
    a=1.0,
    e=0.2,
    primary=star,
    hash="planet1"
)

sim.add(
    m=2e-3,
    a=1.8,
    e=0.1,
    primary=star,
    hash="planet2"
)
sim.move_to_com()
sim.integrator = "ias15"
initial_sim = sim.copy()
times = np.linspace(0.0, 20.0 * np.pi, 1000)
E0 = sim.energy()
x1, y1 = [], []
x2, y2 = [], []
xs, ys = [], []
a1s, e1s = [], []
a2s, e2s = [], []
energy_errors = []

for t in times:
    sim.integrate(t)

    star = sim.particles["star"]
    planet1 = sim.particles["planet1"]
    planet2 = sim.particles["planet2"]

    orbit1 = planet1.orbit(primary=star)
    orbit2 = planet2.orbit(primary=star)

    a1s.append(orbit1.a)
    e1s.append(orbit1.e)

    a2s.append(orbit2.a)
    e2s.append(orbit2.e)


    xs.append(star.x)
    ys.append(star.y)

    x1.append(planet1.x)
    y1.append(planet1.y)

    x2.append(planet2.x)
    y2.append(planet2.y)
    energy_errors.append((sim.energy() - E0)/abs(E0))

plt.plot(x1, y1, label="planet 1")
plt.plot(x2, y2, label="planet 2")
plt.plot(xs, ys, color="orange", label="star")

plt.scatter(
    [xs[-1], x1[-1], x2[-1]],
    [ys[-1], y1[-1], y2[-1]]
)

plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(times, a1s, label="planet 1")
axes[0].plot(times, a2s, label="planet 2")
axes[0].set_ylabel("semimajor axis a")
axes[0].legend()

axes[1].plot(times, e1s, label="planet 1")
axes[1].plot(times, e2s, label="planet 2")
axes[1].set_xlabel("time")
axes[1].set_ylabel("eccentricity e")
axes[1].legend()

plt.figure()
plt.plot(times, energy_errors)
plt.xlabel("time")
plt.ylabel("energy error")
plt.show()

sim2 = initial_sim.copy()

sim2.integrator = "whfast"
sim2.dt = 0.01

target_times = np.linspace(0.0, 20.0 * np.pi, 1000)
actual_times = []

E0 = sim2.energy()

x1, y1 = [], []
x2, y2 = [], []
xs, ys = [], []

a1s, e1s = [], []
a2s, e2s = [], []

energy_errors = []

for t in target_times:
    sim2.integrate(t, exact_finish_time=0)

    star = sim2.particles["star"]
    planet1 = sim2.particles["planet1"]
    planet2 = sim2.particles["planet2"]

    orbit1 = planet1.orbit(primary=star)
    orbit2 = planet2.orbit(primary=star)

    actual_times.append(sim2.t)

    a1s.append(orbit1.a)
    e1s.append(orbit1.e)

    a2s.append(orbit2.a)
    e2s.append(orbit2.e)

    xs.append(star.x)
    ys.append(star.y)

    x1.append(planet1.x)
    y1.append(planet1.y)

    x2.append(planet2.x)
    y2.append(planet2.y)

    error = (sim2.energy() - E0) / abs(E0)
    energy_errors.append(error)

plt.figure()
plt.plot(x1, y1, label="planet 1")
plt.plot(x2, y2, label="planet 2")
plt.plot(xs, ys, color="orange", label="star")

plt.scatter(
    [xs[-1], x1[-1], x2[-1]],
    [ys[-1], y1[-1], y2[-1]]
)

plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("WHFast trajectories")
plt.legend()
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].plot(actual_times, a1s, label="planet 1")
axes[0].plot(actual_times, a2s, label="planet 2")
axes[0].set_ylabel("semimajor axis a")
axes[0].legend()

axes[1].plot(actual_times, e1s, label="planet 1")
axes[1].plot(actual_times, e2s, label="planet 2")
axes[1].set_xlabel("time")
axes[1].set_ylabel("eccentricity e")
axes[1].legend()

fig.suptitle("WHFast orbital elements")

plt.figure()
plt.plot(actual_times, energy_errors)
plt.xlabel("time")
plt.ylabel(r"$(E(t)-E_0)/|E_0|$")
plt.title("WHFast energy error")

plt.show()