from matplotlib import pyplot as plt
import numpy as np

SAMPLES = 500
STEPS = 15

def recursion_function(step, z, c):
    z_next = np.power(z, 2) + c
    if step < STEPS and np.abs(z_next) < 2:
        return recursion_function(step+1, z_next, c)
    return step, z_next, c

def main():
    real_range = np.linspace(-2.5, 1.5, SAMPLES, dtype=complex)
    imag_range = np.linspace(-1.5, 1.5, SAMPLES, dtype=complex)
    image_range = imag_range * 1j
    real, imag = np.meshgrid(real_range, imag_range)
    c_grid = real + (imag * 1j)
    z_grid = np.zeros_like(c_grid)
    step_grid = np.zeros_like(c_grid, dtype=int)
    recursion_function_vec = np.vectorize(recursion_function)
    step_grid, z_grid, c_grid = recursion_function_vec(step_grid, z_grid, c_grid)
    plt.contourf(real, imag, step_grid)
    plt.show()

main()
