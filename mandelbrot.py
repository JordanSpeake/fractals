from matplotlib import pyplot as plt
import numpy as np
import argparse
from numba import jit

SAMPLES = 2500
STEPS = 50

@jit
def mandelbrot(step, z, c):
    z_next = np.power(z, 2) + c
    if step < STEPS and np.abs(z_next) < 2:
        return mandelbrot(step+1, z_next, c)
    return step, z_next, c

@jit
def burning_ship(step, z, c):
    z_next = np.power(np.abs(np.real(z)) + 1j * (np.abs(np.imag(z))), 2) + c
    if step < STEPS and np.abs(z_next) < 4:
        return burning_ship(step+1, z_next, c)
    return step, z_next, c

def render(fractal_function, region, cmap):
    real_range = np.linspace(region[0][0], region[0][1], SAMPLES, dtype=complex)
    imag_range = np.linspace(region[1][0], region[1][1], SAMPLES, dtype=complex)
    image_range = imag_range * 1j
    real, imag = np.meshgrid(real_range, imag_range)
    c_grid = real + (imag * 1j)
    z_grid = np.zeros_like(c_grid)
    step_grid = np.zeros_like(c_grid, dtype=int)
    step_grid, z_grid, c_grid = fractal_function(step_grid, z_grid, c_grid)
    plt.contourf(real, imag, step_grid, cmap=cmap)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fractal",
        type=str,
        nargs=1,
        required=True,
        help="Usage: --fractal [mandelbrot, burning_ship]",
    )
    arguments = parser.parse_args()
    fractal_type = arguments.fractal[0]
    if fractal_type == "mandelbrot":
        return mandelbrot, [[-2.5, 1.5],[-1.5, 1.5]], "viridis_r"
    if fractal_type == "burning_ship":
        return burning_ship, [[-2.5, 1.5],[-2, 1]], "inferno_r"
    raise Exception

def main():
    recursion_function, region, cmap = parse_arguments()
    recursion_function_vec = np.vectorize(recursion_function)
    render(recursion_function_vec, region, cmap)

main()
