import numpy as np
import matplotlib.pyplot as plt

def energy(q,p,k,m):
    return 1/2 * k * q**2 + p**2/(2*m)

def euler(k, m, q0, p0, h, n):
    q = np.zeros(n + 1)
    p = np.zeros(n + 1)

    q[0] = q0
    p[0] = p0

    for i in range(1, n + 1):
        q[i] = q[i - 1] + h * p[i - 1] / m
        p[i] = p[i - 1] - h * k * q[i - 1]

    return q, p

def symplectic_euler(k,m,q0,p0,h,n):
    q = np.zeros(n+1)
    p = np.zeros(n+1)
    q[0] = q0
    p[0] = p0

    for i in range(1,n+1):
        p[i] = p[i-1] - k * q[i-1] * h
        q[i] = q[i-1] + h * (p[i]/m)

    return q, p

def leapfrog(k,m,q0,p0,h,n):
    q = np.zeros(n+1)
    p = np.zeros(n+1)
    q[0] = q0
    p[0] = p0

    for i in range(1,n+1):
        p_half = p[i-1] - k * q[i-1] * h / 2
        q[i] = q[i-1] + h * (p_half/m)
        p[i] = p_half - k * q[i] * h / 2

    return q, p

def RK4(k, m, q0, p0, h, n):
    q = np.zeros(n + 1)
    p = np.zeros(n + 1)

    q[0] = q0
    p[0] = p0

    for i in range(1, n + 1):
        qn = q[i - 1]
        pn = p[i - 1]

        k1_q = pn / m
        k1_p = -k * qn

        k2_q = (pn + h * k1_p / 2) / m
        k2_p = -k * (qn + h * k1_q / 2)

        k3_q = (pn + h * k2_p / 2) / m
        k3_p = -k * (qn + h * k2_q / 2)

        k4_q = (pn + h * k3_p) / m
        k4_p = -k * (qn + h * k3_q)

        q[i] = qn + h/6 * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        p[i] = pn + h/6 * (k1_p + 2*k2_p + 2*k3_p + k4_p)

    return q, p


k = 1.0
m = 1.0
q0 = 1.0
p0 = 1.0
T = 1000
h = 0.01
n = int(T/h)
t = np.arange(n+1) * h

q_e, p_e = euler(k, m, q0, p0, h, n)
q_se, p_se = symplectic_euler(k, m, q0, p0, h, n)
q_lf, p_lf = leapfrog(k, m, q0, p0, h, n)
q_rk, p_rk = RK4(k, m, q0, p0, h, n)

H_e = energy(q_e, p_e, k, m)
H_se = energy(q_se, p_se, k, m)
H_lf = energy(q_lf, p_lf, k, m)
H_rk = energy(q_rk, p_rk, k, m)

H0 = energy(q0, p0, k, m)
err_e =  H_e - H0
err_se = H_se - H0
err_lf = H_lf - H0
err_rk = H_rk - H0


plt.figure(figsize=(10, 6))
plt.plot(t, err_e, label="Euler")
plt.plot(t, err_se, label="Symplectic Euler")
plt.plot(t, err_lf, label="Leapfrog")
plt.plot(t, err_rk, label="RK4")
plt.xlabel("Time")
plt.ylabel("H(t) - H(0)")
plt.title("Energy Error for Harmonic Oscillator")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(t, err_se, label="Symplectic Euler")
plt.plot(t, err_lf, label="Leapfrog")
plt.plot(t, err_rk, label="RK4")
plt.xlabel("Time")
plt.ylabel("H(t) - H(0)")
plt.title("Energy Error: Symplectic Methods vs RK4")
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(t[t<=20], err_se[t<=20], label="Symplectic Euler")
plt.plot(t[t<=20], err_lf[t<=20], label="Leapfrog")
plt.plot(t[t<=20], err_rk[t<=20], label="RK4")
plt.xlabel("Time")
plt.ylabel("H(t) - H(0)")
plt.title("Energy Error: Symplectic Methods vs RK4")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(q_se, p_se, label="Symplectic Euler")
plt.plot(q_lf, p_lf, label="Leapfrog")
plt.plot(q_rk, p_rk, label="RK4")
plt.plot(q_e, p_e, label="Euler")
plt.xlabel("q")
plt.ylabel("p")
plt.title("Phase Space")
plt.legend()
plt.grid()
plt.show()


print("Euler", np.min(err_e), np.max(err_e))

print("Symplectic Euler:",
      np.min(err_se), np.max(err_se))

print("Leapfrog:",
      np.min(err_lf), np.max(err_lf))

print("RK4:",
      np.min(err_rk), np.max(err_rk))