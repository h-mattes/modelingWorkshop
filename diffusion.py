import numpy as np
import matplotlib.pyplot as plt

def calculate_time_step(grid_spacing, diffusivity):
    return 0.5 * grid_spacing**2 / diffusivity


def set_initial_profile(grid_size=100, boundary_left=500, boundary_right=0):
    profile = np.empty(grid_size)
    profile[: grid_size//2] = boundary_left
    profile[grid_size//2 :] = boundary_right
    return profile


def make_grid(origin, domain_size, grid_spacing):
    grid = np.arange(start=0, stop=origin+domain_size, step=grid_spacing)
    return (grid, len(grid))


def plot_profile(grid, concentration, color="red", title="Concentration profile"):
    plt.figure() 
    plt.plot(grid, concentration, color)
    plt.xlabel("x")
    plt.ylabel("C")
    plt.title(title)


def solve_1d_diffusion(concentration, grid_spacing= 1.0, time_step=1.0, diffusivity=1.0):
    centered_difference = np.roll(concentration, -1) - 2*concentration + np.roll(concentration, 1)
    concentration[1:-1] += diffusivity * time_step / grid_spacing**2 * centered_difference[1:-1]


def diffusion_model():
    D = 100  # diffusivity
    Lx = 300  # domain size
    dx = 0.5
    C_left = 500
    C_right = 0
    nt = 5000

    x, nx = make_grid(0, Lx, dx) 
    dt = calculate_time_step(dx, D)
    C = set_initial_profile(nx, boundary_left = C_left, boundary_right = C_right)

    plot_profile(x, C, title="Initial concentration profile")

    for t in range(0, nt):
        solve_1d_diffusion(C, dx, dt, D)

    plot_profile(x, C, color="blue", title="Final concentration profile")