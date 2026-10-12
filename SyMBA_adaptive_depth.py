from SyMBA_fixed_depth import *

def need_deeper_level():
    return True

def evolve_level(Q, P, m, dt, level,radii, G=1.0):
    max_level = len(radii) - 1
    level_wise_interaction_kick(Q, P, m, dt/2,level,radii, G)
    if level == max_level or not need_deeper_level():
        kepler_drift(Q, P, m, dt, G)
    else:
        next_dt = dt / 3
        for _ in range(3):
            evolve_level(Q, P, m, next_dt, level+1, radii, G)
    level_wise_interaction_kick(Q, P, m, dt/2,level,radii, G)