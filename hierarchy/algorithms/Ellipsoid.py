# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:00:00 2022

@author: dv516
"""


import numpy as np
import math
import time
from typing import List, Tuple, Callable

# OPTIMIZER

# Greedy sampler that samples over the surface of a ellipsoid and
# reduces its mayor axis iteratively, moving the center if promising
# directions are found on the surface.
# A heuristic is proposed to estimate the least number of samples
# required to cover all directions as a function of the dimension.

def bounds_scaler(x: float, x_low: float, x_high: float):
    "Rescale a number to fall within the given bounds."
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


def enforce_bounds(points: np.ndarray, bounds: List[Tuple[float]]):
    "Make sure points stay within feasible region."
    projected_points = []
    for point in points:
        projected_point = []
        for coord, bound in zip(point, bounds):
            x_low, x_high = bound
            proj = bounds_scaler(coord, x_low, x_high)
            projected_point.append(proj)
        projected_points.append(np.array(projected_point))
    return projected_points


def unitball_surface_sampler(npoints: int, dimension: int):
    "Sample n points in the surface of the surface of the D dimensional unit ball."
    unitball_samples = []
    for _ in range(npoints):
        coord_samples = np.random.standard_normal(dimension)
        # r = np.sqrt(samples**2)
        r = np.linalg.norm(coord_samples)
        point = coord_samples / r
        unitball_samples.append(point)
    return unitball_samples

def direction_scaling(
    unitball_samples: List[np.ndarray],
    bounds: List[Tuple[float]],
    radius: float,
    mayor_axis: float,
):
    "Scale unit ball points to a ellipsoid acording to a characteristic length of each dimension"
    assert radius <= mayor_axis
    scaled_samples = []
    for sample in unitball_samples:
        scaled = []
        for coord, bound in zip(sample, bounds):
            assert bound[1] > bound[0]
            half_space = (bound[1] - bound[0]) / 2
            factor = radius / mayor_axis
            scaled.append(coord * half_space * factor)
        scaled_samples.append(np.array(scaled))
    return scaled_samples


def dimensionality_factor(dimension: int):
    """
    Given a state dimension estimate the minimum number of points that cover the main directions.
    Each dimension has a canonical vector, so an heuristic to account for the possible directions
    between canonical vectors is to number the pairing combinations between these directions
    plus the unmodified canonical direction, with 2 directions per canonical vector.
    """
    combinations_without_repetition = math.comb(dimension, 2)
    dim_factor = 2 * dimension * (combinations_without_repetition + 1)
    return  2 * dim_factor  # extra factor to account for poor discrepancy of sampler

def optimizer(
    f: Callable, N: int, bounds: List[Tuple[float]], 
) -> dict:
    """
    Optimizer aims to optimize a black-box function 'f' using the dimensionality
    'N_x', and box-'bounds' on the decision vector
    Input:
        f: function: taking as input a list of size N_x and outputing a float
        N: int: Evaluation budget
        bounds: List of size N where each element i is a tuple conisting of 2 floats
                (lower, upper) serving as box-bounds on the ith element of x
    Return:
        tuple:  1st element: lowest value found for f
                2nd element: list/array of size N_x giving the support point
    """

    ### Your code here ###

    t0 = time.time()

    dimension = len(bounds)
    N_x = len(bounds)

    center = [np.mean(bounds[i]) for i in range(N_x)]
    bounds_halfsize = [(bound[1] - bound[0]) / 2 for bound in bounds]

    # dimension heuristics
    dim_factor = dimensionality_factor(dimension)
    radius_reduction = 0.9

    # initialization
    max_radius = np.amax(bounds_halfsize)
    radius = max_radius
    evals = 0

    x = center
    fx = f(x)

    counter = 0
    safeguard = 1000000
    while evals < N and counter < safeguard:
        counter += 1

        unitball_samples = unitball_surface_sampler(dim_factor, dimension)
        ellipsoid_samples = direction_scaling(
            unitball_samples, bounds, radius, max_radius
        )

        points = []
        funs = []
        for sample in ellipsoid_samples:

            point = sample + center
            fun = f(point)
            evals += 1

            if fun < fx:
                x = point
                fx = fun

            points.append(point)
            funs.append(fun)

        lucky_index = np.argmin(funs)
        lucky_fun = f(points[lucky_index])
        if lucky_fun < f(center):
            center = points[lucky_index]
        radius *= radius_reduction

    res = {
        'x_best': list(x),
        'f_best': fx,
        'runtime': time.time() - t0,
    }

    return res
