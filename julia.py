from matplotlib import pyplot as plt
import numpy as np
import argparse
from numba import jit

SAMPLES = 5000
STEPS = 500

@jit
def julia(step, z, c):
    z_next = np.power(z, 2) + c
    if step < STEPS and np.abs(z_next) <= 4:
        return julia(step+1, z_next, c)
    return step, z_next, c

def render(origin):
    real_range = np.linspace(-1, 1, SAMPLES, dtype=complex)
    imag_range = np.linspace(-1, 1, SAMPLES, dtype=complex)
    image_range = imag_range * 1j
    real, imag = np.meshgrid(real_range, imag_range)
    z_grid = real + (imag * 1j)
    c_grid = np.ones_like(z_grid) * origin
    step_grid = np.zeros_like(z_grid, dtype=int)
    julia_vec = np.vectorize(julia)
    step_grid, _, _ = julia_vec(step_grid, z_grid, c_grid)
    plt.contourf(real, imag, step_grid, cmap="twilight_shifted")
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real",
        type=complex,
        nargs=1,
        required=True,
        help="Usage: --real 0.5",
    )
    parser.add_argument(
        "--imag",
        type=complex,
        nargs=1,
        required=True,
        help="Usage: --imag 0.5",
    )
    arguments = parser.parse_args()
    real_val = arguments.real[0]
    imag_val = arguments.imag[0]
    origin = real_val + 1j * imag_val
    return origin

def main():
    origin = parse_arguments()
    render(origin)

main()
