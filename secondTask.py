import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


A = np.array([[2.0, 0.0], [-0.5, 3.0]])
I = np.eye(2)
x_min, x_max = -5.0, 5.0
nx = 201
x = np.linspace(x_min, x_max, nx)
dx = x[1] - x[0]
T = 4.0
sigma = 0.3
dt = sigma * dx
nt = int(T / dt)


def initial_condition(x_point):
    return np.array([4.0, 1.0]) if x_point < 0 else np.array([2.0, 2.0])

u0 = np.zeros((nx, 2))
for i in range(nx):
    u0[i,:] = initial_condition(x[i])


def implicit_scheme(u0, A, dt, dx, nt):
    nx = u0.shape[0]
    solutions = np.zeros((nt + 1, nx, 2))
    solutions[0] = u0
    sigma = dt / dx
    
    M_left  = I - sigma*A  
    M_right = sigma*A      

    u = u0.copy()

    for n in range(nt):
        u_old = u.copy()
        for i in range(1, nx):
            u[i] = M_left @ u_old[i] + M_right @ u_old[i-1]
        u[0] = u_old[0]

        solutions[n+1] = u

    return solutions

def lax_wendroff(u, dt, dx, A):
    u_new = np.zeros_like(u)
    for i in range(1, nx-1):
        du = (u[i+1] - u[i-1]) / (2*dx)
        d2u = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
        u_new[i] = u[i] - dt * A @ du + 0.5 * dt**2 * A @ A @ d2u
    

    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    return u_new

def solve_explicit(u0, A, dt, dx, nt):
    solutions = np.zeros((nt + 1, nx, 2))
    solutions[0] = u0.copy()
    
    for n in range(nt):
        solutions[n+1] = lax_wendroff(solutions[n], dt, dx, A)
    
    return solutions


implicit_sol = implicit_scheme(u0, A, dt, dx, nt)
explicit_sol = solve_explicit(u0, A, dt, dx, nt)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


for ax in (ax1, ax2):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 5)
    ax.grid(True)

ax1.set_title('Неявная схема (1-й порядок)')
ax2.set_title('Явная схема Лакса-Вендроффа (2-й порядок)')
ax1.set_ylabel('u')
ax2.set_ylabel('u')
ax1.set_xlabel('x')
ax2.set_xlabel('x')


line_imp1, = ax1.plot(x, implicit_sol[0,:,0], 'b-', label='u1')
line_imp2, = ax1.plot(x, implicit_sol[0,:,1], 'r-', label='u2')
line_exp1, = ax2.plot(x, explicit_sol[0,:,0], 'b-', label='u1')
line_exp2, = ax2.plot(x, explicit_sol[0,:,1], 'r-', label='u2')

ax1.legend()
ax2.legend()

def init():
    line_imp1.set_data(x, implicit_sol[0,:,0])
    line_imp2.set_data(x, implicit_sol[0,:,1])
    line_exp1.set_data(x, explicit_sol[0,:,0])
    line_exp2.set_data(x, explicit_sol[0,:,1])
    return line_imp1, line_imp2, line_exp1, line_exp2

def update(frame):
    t = frame * dt
    fig.suptitle(f'Сравнение схем, t = {t:.2f}')
    
    line_imp1.set_data(x, implicit_sol[frame,:,0])
    line_imp2.set_data(x, implicit_sol[frame,:,1])
    line_exp1.set_data(x, explicit_sol[frame,:,0])
    line_exp2.set_data(x, explicit_sol[frame,:,1])
    
    return line_imp1, line_imp2, line_exp1, line_exp2

ani = FuncAnimation(
    fig, update,
    frames=range(0, nt, 2),
    init_func=init,
    blit=True,
    interval=50,
    repeat_delay=1000
)

plt.tight_layout()
plt.show()