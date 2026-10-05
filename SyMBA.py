import numpy as np
import rebound
import ctypes as ct

_kepler_solver = (
    rebound.clibrebound.reb_integrator_whfast_kepler_solver
)

_kepler_solver.argtypes = [
    ct.POINTER(rebound.Particle),
    ct.c_double,
    ct.c_double,
    ct.POINTER(rebound.Simulation),
]

_kepler_solver.restype = None


def cartesian_to_dh(q,p,m):
    """
    Convert inertial Cartesian canonical coordinates (q, p)
    to Democratic Heliocentric canonical coordinates (Q, P).
    :param q: ndarray,shape(N,3) Cartesian position q[0] denotes the sun
    :param p: ndarray,shape(N,3) Cartesian momenta. p[i] = m[i] * v[i]
    :param m: ndarray,shape(N,)  mass, m[0] denote the sun mass
    :return:
    Q: ndarray, shape (N, 3)
        DH positions.
        Q[0] = barycenter position.
        Q[i] = q[i] - q[0] for i != 0. （Relative to Sun position.)
    P : ndarray, shape (N, 3)
        DH canonical momenta.
        P[0] = total momentum.
        P[i] = barycentric momentum for i != 0.
    """

    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.asarray(m, dtype=float)

    m_total = np.sum(m)
    p_total = np.sum(p,axis = 0)

    Q = np.zeros_like(q)
    P = np.zeros_like(p)

    Q[0] = np.sum(m[:,None] * q,axis = 0) / m_total  # m[:,None] broadcasting m from (N,) to (N,1)
    Q[1:] = q[1:] - q[0]

    P[0] = p_total
    P[1:] = p[1:] - (m[1:,None] * p_total) / m_total
    return Q, P

def dh_to_cartesian(Q,P,m):
    """
        Convert DH canonical coordinates to inertial Cartesian coordinates.

        Parameters
        ----------
        Q : (N, 3)
            Q[0]  : barycenter position.
            Q[1:] : heliocentric positions.
        P : (N, 3)
            P[0]  : total momentum.
            P[1:] : barycentric momenta.
        m : (N, )
            Masses; index 0 denotes the Sun.
        Returns
        -------
        q : (N, 3)
            Inertial Cartesian positions.
        p : (N, 3)
            Inertial Cartesian momenta.
    """
    Q = np.asarray(Q, dtype=float)
    P = np.asarray(P, dtype=float)
    m = np.asarray(m, dtype=float)

    m_total = np.sum(m)

    q = np.zeros_like(Q)
    p = np.zeros_like(P)

    q[0] = Q[0] - np.sum(m[1:,None] * Q[1:],axis = 0) / m_total
    q[1:] = Q[1:] + q[0] #  same as Q[1:] + Q[0] - np.sum(m[1:,None] * Q[1:],axis = 0) / m_total

    p[1:] = P[1:] + m[1:, None] * P[0] / m_total
    p[0] =P[0] - np.sum(p[1:],axis = 0)

    return q, p


def sun_drift(Q,P,m,dt):
    """
    Apply the H_Sun flow in place.

    Update Q[1:] only.
    Q[0] and all momenta P remain unchanged.
    """
    displacement = dt * np.sum(P[1:], axis=0) / m[0]
    Q[1:] += displacement


def interaction_kick(Q, P, m, dt, G=1.0):
    """
    Apply the full planet-planet interaction flow in place.

    Update P[1:] only.
    All positions Q and total momentum P[0] remain unchanged.
    """
    n = len(m)

    for i in range(1, n):
        for j in range(i + 1, n): #deal with all pairs of particles
            r = Q[j] - Q[i]
            r2 = np.dot(r, r)

            if r2 == 0.0:
                raise ValueError("Two planets occupy the same position.")

            force = G * m[i] * m[j] * r / (r2 * np.sqrt(r2))
            impulse = dt * force

            P[i] += impulse
            P[j] -= impulse

def kepler_drift(Q, P, m, dt, G=1.0):
    """
    Apply the H_Kep flow in place using REBOUND's solver.

    Update Q[1:] and P[1:].

    Assumes floating-point arrays and positive masses.
    """
    if dt == 0.0:
        return

    mu = G * m[0]

    for i in range(1, len(m)):
        velocity = P[i] / m[i]

        # create a particle in REBOUND that represent the current particle
        particle = rebound.Particle(
            x=Q[i, 0],
            y=Q[i, 1],
            z=Q[i, 2],
            vx=velocity[0],
            vy=velocity[1],
            vz=velocity[2],
        )

        _kepler_solver(
            ct.byref(particle),
            mu,
            dt,
            None,
        )

        Q[i] = [particle.x, particle.y, particle.z]

        P[i] = m[i] * np.array([
            particle.vx,
            particle.vy,
            particle.vz,
        ])


def dh_step(Q,P,m,dt,G = 1.0):
    sun_drift(Q,P,m,dt/2)

    interaction_kick(Q,P,m,dt/2,G)

    kepler_drift(Q,P,m,dt, G)

    interaction_kick(Q,P,m,dt/2, G)

    sun_drift(Q,P,m,dt/2)

    Q[0] += dt * P[0] / np.sum(m) #important, easy to miss


def total_energy(Q, P, m, G=1.0):
    """Compute the full Newtonian energy; all masses must be positive."""
    q, p = dh_to_cartesian(Q, P, m)

    kinetic = np.sum(
        np.sum(p * p, axis=1) / (2.0 * m)
    )

    potential = 0.0

    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            distance = np.linalg.norm(q[j] - q[i])
            potential -= G * m[i] * m[j] / distance

    return kinetic + potential

def weight_changing_function(r,r_outer,r_inner):
    """smooth function is chosen as 2*x**3 - 3*x**2 + 1. as in the paper"""
    if not (r_outer > r_inner > 0.0):
        raise ValueError("Require r_outer > r_inner > 0.")

    if r < 0.0:
        raise ValueError("Distance must be nonnegative.")

    if r <= r_inner:
        return 0.0

    if r >= r_outer:
        return 1.0

    x = (r_outer - r) / (r_outer - r_inner)

    return 2*x**3 - 3*x**2 + 1


def level_weights(r, radii):
    """
    Return weights for shell 0 ... L.

    radii = [R_0, ..., R_L], the radius for each shell L strictly decreasing.
    The deepest level receives the remaining force.
    """
    radii = np.asarray(radii, dtype=float)

    if radii.ndim != 1 or len(radii) == 0:
        raise ValueError("radii must be a nonempty 1D array.")

    if not np.all(radii > 0.0):
        raise ValueError("Radii must be positive.")

    if not np.all(np.diff(radii) < 0.0):
        raise ValueError("Radii must be strictly decreasing.")

    if r < 0.0:
        raise ValueError("Distance must be nonnegative.")

    L = len(radii) - 1

    if L == 0:
        return np.array([1.0])

    cumulative = np.array([weight_changing_function(r,radii[k],radii[k+1]) for k in range(L)])

    weights = np.empty(len(radii))
    weights[0] = cumulative[0]
    weights[1:L] = np.diff(cumulative)
    weights[L] = 1.0 - cumulative[-1]

    np.testing.assert_allclose(np.sum(weights), 1, rtol=0, atol=1e-14)

    return weights

def make_radii():
    pass

def level_wise_interaction_kick():
    pass

def evolve_level():
    pass



