"""Deterministic numerical fixtures shared by tests and the validation figure."""
from itertools import combinations

import numpy as np
import rebound

import SyMBA_fixed_depth as fixed
import SyMBA_adaptive_depth_optimized as optimized


def initial_state(case):
    if case == "shared":
        xyz = [[1, -.15, 0], [1.03, 0, 0], [.98, .15, 0], [2.5, 0, 0]]
        vel = [[0, 1.4, 0], [0, 1, 0], [0, .6, 0], [0, .63, 0]]
    elif case == "independent":
        xyz = [[1, -.12, 0], [1.03, .12, 0], [-1, -.12, 0], [-1.03, .12, 0]]
        vel = [[0, 1.4, 0], [0, .6, 0], [0, -.6, 0], [0, -1.4, 0]]
    elif case == "spatial":
        xyz = [[.88, -.04, .03], [1.10, .02, -.02], [1.02, -.15, -.03], [.97, .13, .04]]
        vel = [[.4, 1.05, -.02], [-.35, .95, .04], [.03, 1.4, .02], [-.02, .6, -.03]]
    elif case == "many_particles":
        # Ten planets: two shared-particle triples, one pair, two spectators.
        # Rotate the encounter groups around the star to keep them separated.
        xyz, vel = [], []
        groups = [
            (0.0, [[1, -.15, 0], [1.03, 0, 0], [.98, .15, 0]],
             [[0, 1.4, 0], [0, 1, 0], [0, .6, 0]]),
            (2 * np.pi / 3, [[1, -.15, 0], [1.03, 0, 0], [.98, .15, 0]],
             [[0, 1.32, 0], [0, 1, 0], [0, .68, 0]]),
            (4 * np.pi / 3, [[1, -.12, 0], [1.03, .12, 0]],
             [[0, 1.4, 0], [0, .6, 0]]),
        ]
        for angle, positions, velocities in groups:
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            xyz.extend(np.array(positions) @ rotation.T)
            vel.extend(np.array(velocities) @ rotation.T)
        xyz.extend([[2.5, 0, 0], [-2.8, 0, 0]])
        vel.extend([[0, np.sqrt(1 / 2.5), 0], [0, -np.sqrt(1 / 2.8), 0]])
    elif case == "smooth":
        xyz = [[1, 0, 0], [0, 1.8, .1]]
        vel = [[0, 1, .02], [-.74, 0, 0]]
    else:
        raise ValueError(case)
    m = np.array([1.] + [2e-5] * len(xyz))
    q = np.vstack(([0., 0., 0.], xyz))
    v = np.vstack(([0., 0., 0.], vel))
    q -= np.sum(m[:, None] * q, axis=0) / m.sum()
    v -= np.sum(m[:, None] * v, axis=0) / m.sum()
    return q, v, m


def reference(q, v, m, times):
    sim = rebound.Simulation()
    sim.G = 1.
    sim.integrator = "ias15"
    sim.integrator.epsilon = 1e-12
    for qi, vi, mi in zip(q, v, m):
        sim.add(m=mi, x=qi[0], y=qi[1], z=qi[2], vx=vi[0], vy=vi[1], vz=vi[2])
    positions, velocities = [], []
    for t in times:
        sim.integrate(float(t), exact_finish_time=1)
        positions.append([[p.x, p.y, p.z] for p in sim.particles])
        velocities.append([[p.vx, p.vy, p.vz] for p in sim.particles])
    return np.array(positions), np.array(velocities)


def integrate(q, v, m, times, dt, module=optimized):
    Q, P = fixed.cartesian_to_dh(q, m[:, None] * v, m)
    radii = fixed.make_radii(.15, 4)
    positions, velocities = [q.copy()], [v.copy()]
    for start, end in zip(times[:-1], times[1:]):
        steps = round((end - start) / dt)
        if steps < 1 or not np.isclose(steps * dt, end - start, rtol=0, atol=1e-12):
            raise ValueError("Output intervals must be positive integer multiples of dt")
        for _ in range(steps):
            module.symba_step(Q, P, m, dt, radii)
        qi, pi = fixed.dh_to_cartesian(Q, P, m)
        positions.append(qi)
        velocities.append(pi / m[:, None])
    return np.array(positions), np.array(velocities)


def diagnostics(q, v, m):
    """Independent Cartesian diagnostics at output times, including every pair."""
    energy = .5 * np.sum(m[None, :, None] * v**2, axis=(1, 2))
    for i, j in combinations(range(len(m)), 2):
        energy -= m[i] * m[j] / np.linalg.norm(q[:, i] - q[:, j], axis=1)
    angular = np.sum(np.cross(q, m[None, :, None] * v), axis=1)
    distances = np.array([np.linalg.norm(q[:, i] - q[:, j], axis=1)
                          for i, j in combinations(range(1, len(m)), 2)])
    return {
        "energy": np.abs((energy - energy[0]) / energy[0]),
        "angular": np.linalg.norm(angular - angular[0], axis=1) / np.linalg.norm(angular[0]),
        "distance": distances.min(axis=0),
    }


def errors(actual, expected):
    return np.max(np.linalg.norm(actual[:, 1:] - expected[:, 1:], axis=2), axis=1)
