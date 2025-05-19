import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


L = 10.0
T = 5.0
dx = 0.1
dt = 0.01
nx = int(L/dx) + 1
nt = int(T/dt) + 1
x = np.linspace(0, L, nx)


def initial_condition(x):
    return np.where((x >= 4.9) & (x <= 5.1), 1.0, 0.0)


cases = [
    {'alpha': 0.0, 'beta': 2.0, 'label': 'α=0, β=2'},
    {'alpha': 0.1, 'beta': 0.0, 'label': 'α=0.1, β=0'},
    {'alpha': 0.1, 'beta': 2.0, 'label': 'α=0.1, β=2'}
]


approximations = [
    {'type': 'forward', 'name': 'Вперед', 'color': 'blue'},
    {'type': 'backward', 'name': 'Назад', 'color': 'green'},
    {'type': 'central', 'name': 'Центральная', 'color': 'red'}
]


def solve_case(alpha, beta, approx_type):
    u = np.zeros((nt, nx))
    u[0] = initial_condition(x)
    
    for n in range(nt-1):
        for i in range(1, nx-1):

            diffusion = alpha * (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx**2
            

            if approx_type == 'forward':
                adv_term = beta * (u[n, i+1] - u[n, i]) / dx
            elif approx_type == 'backward':
                adv_term = beta * (u[n, i] - u[n, i-1]) / dx
            else:
                adv_term = beta * (u[n, i+1] - u[n, i-1]) / (2*dx)
            
            u[n+1, i] = u[n, i] + dt * (diffusion - adv_term)
        

        u[n+1, 0] = 0
        u[n+1, -1] = 0
    
    return u


solutions = {}
for case in cases:
    for approx in approximations:
        key = f"{case['label']}, {approx['name']}"
        solutions[key] = {
            'solution': solve_case(case['alpha'], case['beta'], approx['type']),
            'color': approx['color'],
            'case': case['label'],
            'approx': approx['name']
        }


fig, axes = plt.subplots(3, 3, figsize=(18, 12))
lines = {}


for i, case in enumerate(cases):
    for j, approx in enumerate(approximations):
        ax = axes[i, j]
        key = f"{case['label']}, {approx['name']}"
        sol_data = solutions[key]
        
        line, = ax.plot(x, sol_data['solution'][0], 
                       color=approx['color'],
                       label=f"{case['label']}, {approx['name']}")
        
        ax.set_xlim(0, L)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f"{case['label']}, {approx['name']}")
        ax.grid(True)
        
        lines[key] = line

fig.suptitle('Решение уравнения диффузии с переносом\nРазличные параметры и аппроксимации', fontsize=16)
plt.tight_layout()

def update(frame):
    for key in solutions:
        lines[key].set_ydata(solutions[key]['solution'][frame])
    
    fig.suptitle(f'Решение уравнения диффузии с переносом\nРазличные параметры и аппроксимации', 
                fontsize=16)
    return list(lines.values())

ani = FuncAnimation(
    fig, update,
    frames=range(0, nt, 5),
    interval=100,
    blit=True
)

plt.tight_layout()
plt.show()
