from SyMBA_adaptive_depth import *



def get_deeper_particles_pairs():
    return None,None

def level_wise_interaction_kick(Q, P, m, particles, pairs,dt,level,radii, G=1.0):
    n = len(m)
    for i in range(1, n):
        for j in range(i + 1, n):
            r = Q[j] - Q[i]
            r2 = np.dot(r, r)
            weights = level_weights(np.sqrt(r2), radii)
            force = G * m[i] * m[j] * r / (r2 * np.sqrt(r2))
            impulse = dt * force * weights[level]
            P[i] += impulse
            P[j] -= impulse


def evolve_level(Q, P, m, particles, pairs,
                 dt, level,radii, G=1.0):
    max_level = len(radii) - 1
    level_wise_interaction_kick(Q, P, m,particles,pairs,dt/2,level,radii, G)
    deeper_particles, deeper_pairs = get_deeper_particles_pairs()
