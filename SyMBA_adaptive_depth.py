import numpy as np
from SyMBA_fixed_depth import (
    sun_drift,
    kepler_drift,
    level_wise_interaction_kick,
)


def need_deeper_level(Q,P,m,dt,level,radii, G=1.0):
    max_level = len(radii) - 1
    if level >= max_level:
        return False

    h = abs(dt)
    direction = 1.0 if dt >= 0.0 else -1.0

    n = len(m)
    for i in range(1,n):
        for j in range(i+1,n):

            r = Q[i] - Q[j]
            v = (P[i]/m[i] - P[j]/m[j]) * direction

            if np.dot(r,r) <= radii[level]**2:
                return True

            if np.dot(r,v) >= 0:
                continue

            t = min(-np.dot(r, v) / np.dot(v, v), h)
            r_near = r + v * t
            r_min_2 = np.dot(r_near, r_near)
            if r_min_2 <= radii[level]**2:
                return True

    return False


def evolve_level(Q, P, m, dt, level,radii, G=1.0):
    level_wise_interaction_kick(Q, P, m, dt/2,level,radii, G)
    if not need_deeper_level(Q,P,m,dt,level,radii, G):
        kepler_drift(Q, P, m, dt, G)
    else:
        next_dt = dt / 3
        for _ in range(3):
            evolve_level(Q, P, m, next_dt, level+1, radii, G)
    level_wise_interaction_kick(Q, P, m, dt/2,level,radii, G)

def symba_step(Q, P, m, dt,radii, G=1.0):
    sun_drift(Q,P,m,dt/2)

    evolve_level(Q, P, m, dt, 0, radii, G)

    sun_drift(Q,P,m,dt/2)

    Q[0] += dt * P[0] / np.sum(m)
