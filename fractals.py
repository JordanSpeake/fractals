from matplotlib import pyplot as plt
import numpy as np
import argparse
from numba import jit

SAMPLES = 1000
STEPS = 25

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

def render(fractal_function):
    real_range = np.linspace(-2.5, 1.5, SAMPLES, dtype=complex)
    imag_range = np.linspace(-1.5, 1.5, SAMPLES, dtype=complex)
    image_range = imag_range * 1j
    real, imag = np.meshgrid(real_range, imag_range)
    c_grid = real + (imag * 1j)
    z_grid = np.zeros_like(c_grid)
    step_grid = np.zeros_like(c_grid, dtype=int)
    step_grid, z_grid, c_grid = fractal_function(step_grid, z_grid, c_grid)
    plt.contourf(real, imag, step_grid, cmap="inferno_r")
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fractaltype",
        type=str,
        nargs=1,
        required=True,
        help="--fractaltype [mandelbrot, burning_ship]",
    )
    arguments = parser.parse_args()
    fractal_type = arguments.fractaltype[0]
    if fractal_type == "mandelbrot":
        return mandelbrot
    if fractal_type == "burning_ship":
        print(burning_ship)
        return burning_ship
    else:
        raise Exception

def main():
    recursion_function = parse_arguments()
    recursion_function_vec = np.vectorize(recursion_function)
    render(recursion_function_vec)

main()
