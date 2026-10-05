import rebound
import numpy as np
import matplotlib.pyplot as plt

sim = rebound.Simulation()
sim.add(m=1.0)
sim.add(m=1e-3, a=1.0, e=0.2)
sim.move_to_com()

sim.integrator = "ias15"

E0 = sim.energy()
times = np.linspace(0.0, 20.0 * np.pi, 1000)
x = []
y = []
energy_error = []

for t in times:
    sim.integrate(t)
    planet = sim.particles[1]
    x.append(planet.x)
    y.append(planet.y)
    energy_error.append((sim.energy() - E0)/E0)

plt.plot(x, y)
plt.scatter([0], [0], color="orange", label="star")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

plt.figure()
plt.plot(times, energy_error)
plt.xlabel("time")
plt.ylabel(r"relative error")
plt.show()