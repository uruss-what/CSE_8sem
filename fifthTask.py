import numpy as np
import matplotlib.pyplot as plt


a = 1.0
L = 25.0
T = 17.0
dx = 0.05
mu = 0.8
dt = mu * dx / a
nx = int(L/dx) + 1
nt = int(T/dt) + 1
x = np.linspace(0, L, nx)


def u0(x):
    return np.exp(-20*(x-2)**2) + np.exp(-(x-5)**2)


def exact_solution(x, t):
    return np.where(x - a*t > 0, u0(x - a*t), 0)


def upwind_scheme():
    u = np.zeros((nt, nx))
    u[0] = u0(x)
    
    for n in range(nt-1):
        for i in range(1, nx):
            u[n+1,i] = u[n,i] - a*dt/dx * (u[n,i] - u[n,i-1])

        u[n+1,0] = 0
        u[n+1,-1] = u[n+1,-2]
    
    return u


def lax_wendroff_scheme():
    u = np.zeros((nt, nx))
    u[0] = u0(x)
    
    for n in range(nt-1):
        for i in range(1, nx-1):
            u[n+1,i] = u[n,i] - a*dt/(2*dx)*(u[n,i+1] - u[n,i-1]) \
                      + (a*dt)**2/(2*dx**2)*(u[n,i+1] - 2*u[n,i] + u[n,i-1])

        u[n+1,0] = 0
        u[n+1,-1] = u[n+1,-2]
    
    return u


def beam_warming_scheme():
    u = np.zeros((nt, nx))
    u[0] = u0(x)
    
    for n in range(nt-1):
        for i in range(2, nx):
            u[n+1,i] = u[n,i] - a*dt/(2*dx)*(3*u[n,i] - 4*u[n,i-1] + u[n,i-2]) \
                      + (a*dt)**2/(2*dx**2)*(u[n,i] - 2*u[n,i-1] + u[n,i-2])

        u[n+1,0] = 0
        u[n+1,1] = u[n+1,2]
        u[n+1,-1] = u[n+1,-2]
    
    return u


def fromm_scheme():
    u = np.zeros((nt, nx))
    u[0] = u0(x)
    
    for n in range(nt-1):
        for i in range(2, nx-1):

            lw = u[n,i] - a*dt/(2*dx)*(u[n,i+1] - u[n,i-1]) \
                 + (a*dt)**2/(2*dx**2)*(u[n,i+1] - 2*u[n,i] + u[n,i-1])

            bw = u[n,i] - a*dt/(2*dx)*(3*u[n,i] - 4*u[n,i-1] + u[n,i-2]) \
                 + (a*dt)**2/(2*dx**2)*(u[n,i] - 2*u[n,i-1] + u[n,i-2])

            u[n+1,i] = 0.5 * (lw + bw)

        u[n+1,0] = 0
        u[n+1,1] = u[n+1,2]
        u[n+1,-1] = u[n+1,-2]
    
    return u


u_upwind = upwind_scheme()
u_lw = lax_wendroff_scheme()
u_bw = beam_warming_scheme()
u_fromm = fromm_scheme()


start_idx = int(15/dx)
end_idx = int(25/dx)
x_interval = x[start_idx:end_idx]

plt.figure(figsize=(12, 8))


exact = exact_solution(x_interval, T)
plt.plot(x_interval, exact, 'k--', linewidth=2, label='Точное решение')

plt.plot(x_interval, u_upwind[-1, start_idx:end_idx], 'b-', label='Upwind')
plt.plot(x_interval, u_lw[-1, start_idx:end_idx], 'g-', label='Lax-Wendroff')
plt.plot(x_interval, u_bw[-1, start_idx:end_idx], 'r-', label='Beam-Warming')
plt.plot(x_interval, u_fromm[-1, start_idx:end_idx], 'm-', label='Fromm')

plt.title(f'Сравнение схем при t={T:.1f} (μ={mu:.2f})')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()