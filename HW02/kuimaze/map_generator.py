'''
Simple maze generator
'''

import numpy
from numpy.random import random_integers as rand


def maze(width=10, height=10, complexity=.75, density=.75):
    # Only odd shapes
    width = width + 1
    height = height + 1
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = numpy.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    # convert to maze.py format
    ret = [[0 for x in range(len(Z[0]) - 2)] for y in range(len(Z) - 2)]
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            if i > 0 and j > 0 and i < len(Z) - 1 and j < len(Z[0]) - 1:
                if Z[i][j] == 1:
                    ret[i - 1][j - 1] = [0, 0, 0]
                else:
                    ret[i - 1][j - 1] = [255, 255, 255]
                    if i == 1 and j == 1:
                        ret[i - 1][j - 1] = [0, 0, 255]
                    if i == len(Z) - 2 and j == len(Z[0]) - 2:
                        ret[i - 1][j - 1] = [255, 0, 0]
    return ret
