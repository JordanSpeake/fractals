import numpy as np
from matplotlib import pyplot as plt

STEPS = 3

def display(c_set):
    plt.imshow(np.atleast_2d(c_set), cmap="binary_r")
    plt.show()

def remove_middle_third(c_set):
    length = c_set.shape[0]
    for i in range(1, length):
        if np.all(c_set[i-1:i+1]):
            c_set[i] = 0
    return c_set

def main():
    cantor_set = np.asarray([1])
    for _ in range(STEPS):
        cantor_set = np.repeat(cantor_set, 3)
        cantor_set = remove_middle_third(cantor_set)
    display(cantor_set)

main()
