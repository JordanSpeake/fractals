import numpy as np
from matplotlib import pyplot as plt
from numba import jit

STEPS = 7

def display(c_set):
    plt.imshow(np.atleast_2d(c_set), cmap="binary")
    plt.show()

def serpinski_iter(in_set):
    out_set = np.repeat(np.repeat(np.atleast_2d(in_set), 3, axis=1), 3, axis=0)
    shape = out_set.shape
    for i in range(int(shape[0]/3)):
        for j in range(int(shape[1]/3)):
            out_set[3*i+1][3*j+1] = 0
    return out_set

def cantor_set_iter(c_set): # Unused
    c_set = np.repeat(c_set, 3)
    length = c_set.shape[0]
    for i in range(1, length):
        if np.all(c_set[i-1:i+1]):
            c_set[i] = 0
    return c_set

def main():
    serpinski_set = np.asarray([1])
    colour_set = np.asarray([1])
    for _ in range(STEPS):
        serpinski_set = serpinski_iter(serpinski_set)
    display(serpinski_set)

main()
