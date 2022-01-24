import numpy as np
from typing import List, Tuple


def initialize_weights(layers_size: List[int], array_generator_func=np.random.randn):
    weights = []

    weights_dimensions = list(zip(layers_size[1:], layers_size))
    for dimension in weights_dimensions:
        if array_generator_func is np.zeros:
            w = array_generator_func(dimension)
        else:
            w = array_generator_func(*dimension)
        weights.append(w)

    return weights


def initialize_biases(layers_size: List[int], array_generator_func=np.zeros):
    biases = []

    for dimension in layers_size[1:]:
        b = array_generator_func((dimension, 1))
        biases.append(b)
    return biases


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_input_vector(obstacles, player_x):
    min_x = 0
    max_x = 387  # max possible value for x (experimental)

    min_y = -100
    max_y = 656

    input_vector = []
    for obstacle in obstacles[:3]:
        x, y = obstacle['x'], obstacle['y']

        x = normalize(v=x, min_v=min_x, max_v=max_x)
        y = normalize(v=y, min_v=min_y, max_v=max_y)

        input_vector.extend([x, y])

    while len(input_vector) < 6:
        input_vector.extend([1, 1])

    player_x = normalize(v=player_x, min_v=min_x, max_v=max_x)

    input_vector.extend([player_x])
    num_features = len(input_vector)
    input_vector = np.array(input_vector).reshape(num_features, 1)
    return np.array(input_vector)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def normalize(v, min_v, max_v):
    return (v - min_v) / (max_v - min_v)
