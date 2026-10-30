import numpy as np
from SyMBA_fixed_depth import level_weights,kepler_drift,sun_drift


def make_planet_pairs(m):
    pairs = []
    for i in range(1, len(m)):
        for j in range(i+1,len(m)):
            pairs.append((i,j))
    return pairs


def get_deeper_particles_pairs(Q,P,m,pairs,dt,level,radii, G=1.0):
    deeper_particles = set()
    deeper_pairs = []
    max_level = len(radii) - 1
    h = abs(dt)
    direction = 1.0 if dt >= 0.0 else -1.0

    if level >= max_level:
        return deeper_particles, deeper_pairs

    for i,j in pairs:
        r = Q[i] - Q[j]
        v = (P[i] / m[i] - P[j] / m[j]) * direction

        if np.dot(r, r) <= radii[level] ** 2:
            deeper_pairs.append((i, j))
            deeper_particles.add(i)
            deeper_particles.add(j)
            continue

        if np.dot(r, v) >= 0:
            continue

        t = min(-np.dot(r, v) / np.dot(v, v), h)
        r_near = r + v * t
        r_min_2 = np.dot(r_near, r_near)
        if r_min_2 <= radii[level] ** 2:
            deeper_pairs.append((i, j))
            deeper_particles.add(i)
            deeper_particles.add(j)
    return deeper_particles, deeper_pairs


def get_active_pairs(Q, pairs, level, radii):
    active_pairs = []

    for i, j in pairs:
        r = Q[j] - Q[i]
        distance = np.linalg.norm(r)

        weight = level_weights(distance, radii)[level]

        if weight != 0.0:
            active_pairs.append((i, j))

    return active_pairs

def level_wise_interaction_kick(Q, P, m,
                                active_pairs,dt,level,radii, G=1.0):
    for i,j in active_pairs:
        r = Q[j] - Q[i]
        r2 = np.dot(r,r)

        distance = np.sqrt(r2)
        weight = level_weights(distance,radii)[level]
        if weight == 0.0:
            continue
        force = G * m[i] * m[j] * r / (r2 * np.sqrt(r2))
        impulse = dt * force * weight
        P[i] += impulse
        P[j] -= impulse

def kepler_drift_particles(Q, P, m, particles, dt, G=1.0):
    for i in sorted(set(particles)):
        indices = [0, i]  # Can be optimized when implemented in C, here is for testing the logic only

        Q_local = Q[indices].copy()
        P_local = P[indices].copy()
        m_local = m[indices]

        kepler_drift(Q_local, P_local, m_local, dt, G)

        Q[i] = Q_local[1]
        P[i] = P_local[1]


def evolve_level(Q, P, m, particles, pairs,dt, level,radii, G=1.0):
    particles = set(particles)
    max_level = len(radii) - 1

    level_wise_interaction_kick(Q, P, m,pairs,dt/2,level,radii, G)
    if level >= max_level:
        kepler_drift_particles(Q,P,m,particles,dt,G)
    else:
        deeper_particles, deeper_pairs = get_deeper_particles_pairs(Q,P,m,pairs,dt,level,radii, G)
        drift_here = particles - deeper_particles
        kepler_drift_particles(Q,P,m,drift_here,dt,G)
        if deeper_pairs:
            for _ in range(3):
                evolve_level(Q, P, m, deeper_particles, deeper_pairs, dt/3, level + 1,
                             radii, G)
    level_wise_interaction_kick(Q, P, m,pairs,dt/2,level,radii, G)

def symba_step(Q, P, m, dt, radii, G=1.0):
    if dt == 0.0:
        return

    particles = set(range(1, len(m)))
    pairs = make_planet_pairs(m)

    sun_drift(Q, P, m, dt / 2)

    evolve_level(Q, P, m, particles, pairs,dt, 0, radii, G,)

    sun_drift(Q, P, m, dt / 2)

    Q[0] += dt * P[0] / np.sum(m)





