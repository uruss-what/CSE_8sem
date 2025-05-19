import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initial_condition(x):
    return np.exp(-3.0 * (x - 4.0)**2)

def exact_solution(x, t):
    return np.exp(-3.0 * ((x - t) - 4.0)**2)

def explicit_scheme(u0, nx, nt, a, b, c):
    solutions = np.zeros((nt + 1, nx))
    solutions[0] = u0.copy()
    u = u0.copy()
    for n in range(nt):
        u_old = u.copy()
        for i in range(1, nx - 1):
            u[i] = a * u_old[i + 1] + b * u_old[i] + c * u_old[i - 1]
        u[0] = u[1]
        u[-1] = u[-2]
        solutions[n + 1] = u.copy()
    return solutions

def solve_tridiagonal(a, b, c, d):
    n = len(d)
    x = np.zeros(n)
    c_star = np.zeros(n - 1)
    d_star = np.zeros(n)

    beta = b[0]
    c_star[0] = c[0] / beta
    d_star[0] = d[0] / beta

    for i in range(1, n):
        beta = b[i] - a[i] * c_star[i - 1]
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / beta
        if i < n - 1:
            c_star[i] = c[i] / beta

    x[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
    return x

def implicit_scheme(u0, nx, nt, a, b, c):
    solutions = np.zeros((nt + 1, nx))
    solutions[0] = u0.copy()
    N_int = nx - 2
    A = np.full(N_int, a)
    B = np.full(N_int, b)
    C = np.full(N_int, c)
    for n in range(nt):
        RHS = solutions[n, 1:-1].copy()
        interior = solve_tridiagonal(A, B, C, RHS)
        solutions[n + 1, 1:-1] = interior
        solutions[n + 1, 0] = solutions[n + 1, 1]
        solutions[n + 1, -1] = solutions[n + 1, -2]
    return solutions

def get_scheme_params(scheme, approx, sigma):
    if scheme == "явная":
        if approx == "вперед":
            return -sigma, 1.0 + sigma, 0.0
        elif approx == "назад":
            return 0.0, 1.0 - sigma, sigma
        elif approx == "центр":
            return -0.5*sigma, 1.0, 0.5*sigma
    elif scheme == "неявная":
        if approx == "вперед":
            return -sigma, 1.0 + sigma, 0.0
        elif approx == "назад":
            return -sigma, 1.0+sigma, 0
        elif approx == "центр":
            return -0.5*sigma, 1.0, 0.5*sigma
    raise ValueError("Некорректная комбинация схемы и аппроксимации")

def main():
    x_start, x_end = 0.0, 25.0
    nx = 501
    x = np.linspace(x_start, x_end, nx)
    dx = x[1] - x[0]

    T = 10.0
    sigma = 0.5
    dt = sigma * dx
    nt = int(T / dt)


    u0 = initial_condition(x)


    combinations = [
        ("явная", "вперед", "blue"),
        ("явная", "назад", "green"),
        ("явная", "центр", "cyan"),
        ("неявная", "вперед", "magenta"),
        ("неявная", "назад", "orange"),
        ("неявная", "центр", "purple"),
    ]

    all_solutions = []
    for scheme, approx, _ in combinations:
        a, b, c = get_scheme_params(scheme, approx, sigma)
        if scheme == "явная":
            sol = explicit_scheme(u0, nx, nt, a, b, c)
        else:
            sol = implicit_scheme(u0, nx, nt, a, b, c)
        all_solutions.append(sol)


    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Анимация всех схем: u(x, t)", fontsize=16)

    lines_num = []
    lines_exact = []

    for i, ((scheme, approx, color), sol) in enumerate(zip(combinations, all_solutions)):
        ax = axs[i // 3, i % 3]
        line_num, = ax.plot(x, sol[0], color=color, lw=2, label=f"{scheme}, {approx}")
        line_exact, = ax.plot(x, exact_solution(x, 0), 'r--', lw=1, label="Точное")
        lines_num.append(line_num)
        lines_exact.append(line_exact)

        ax.set_xlim(x_start, x_end)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"{scheme.capitalize()}, {approx}")
        ax.grid(True)
        ax.legend(fontsize=8)

    def update(frame):
        t = frame * dt
        for i in range(6):
            lines_num[i].set_data(x, all_solutions[i][frame])
            lines_exact[i].set_data(x, exact_solution(x, t))
            axs[i // 3][i % 3].set_title(f"{combinations[i][0].capitalize()}, {combinations[i][1]}\n")
        return lines_num + lines_exact

    ani = FuncAnimation(
        fig, update,
        frames=range(0, nt + 1, 5),
        blit=True,
        interval=50
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
