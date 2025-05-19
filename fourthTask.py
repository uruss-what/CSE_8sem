import numpy as np
import matplotlib.pyplot as plt

a = 0.1
L = 25.0
T = 10.0
dx = 0.05
mu = 0.1
dt = mu*dx/a
nx = int(L/dx) + 1
nt = int(T/dt) + 1
x = np.linspace(0, L, nx)


def u0(x):
    return np.exp(-20*(x-2)**2) + np.exp(-(x-5)**2)


def solve():
    u = np.zeros((nt, nx))
    u[0] = u0(x)
    

    epsilon = 0.5 * a * dx * (1 - mu)
    
    for n in range(nt-1):
        for i in range(1, nx-1):
            u[n+1,i] = u[n,i] - (a*dt)/(2*dx)*(u[n,i+1] - u[n,i-1]) \
                        + epsilon*dt/dx**2 * (u[n,i+1] - 2*u[n,i] + u[n,i-1])
        
        u[n+1,0] = 0
        u[n+1,-1] = u[n+1,-2]
    
    return u

u = solve()


t_final = nt-1
plt.figure(figsize=(10,6))
plt.plot(x, u0(x - a*T), 'k--', label='Точное решение переноса')
plt.plot(x, u[t_final], 'r-', label=f'Численное (ε={0.5*a*dx*(1-mu):.4f})')
plt.title(f'Сравнение при t={T:.1f}\n(μ={mu}, ε={0.5*a*dx*(1-mu):.4f})')
plt.xlabel('x'); plt.ylabel('u(x,t)')
plt.legend(); plt.grid()
plt.show()