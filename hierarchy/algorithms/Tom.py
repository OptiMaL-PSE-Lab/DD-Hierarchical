# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:00:00 2022

@author: dv516
"""

import numpy as np
import time
from typing import List, Tuple

def optimizer_dummy(f, N: int, bounds: List[Tuple[float]]) -> dict:
# '''
# Optimizer aims to optimize a black-box function 'f' using the dimensionality
# 'N_x', and box-'bounds' on the decision vector
# Input:
# f: function: taking as input a list of size N_x and outputing a float
# N: int: Evaluation budget
# bounds: List of size N where each element i is a tuple conisting of 2 floats
# (lower, upper) serving as box-bounds on the ith element of x
# Return:
# tuple: 1st element: lowest value found for f, f_min
# 2nd element: list/array of size N_x giving the decision variables
# associated with f_min
# '''

    t0 = time.time()

    bounds = np.array(bounds)
    ### Your code here
    iterations = 5
    surrogate_size = int(N/iterations)
    x_current = np.mean(bounds,axis=1)
    x_ranges = bounds[:,1]-bounds[:,0]

    def LHS(bounds,p):
        d = len(bounds)
        sample = np.zeros((p,len(bounds)))
        for i in range(0,d):
            sample[:,i] = np.linspace(bounds[i,0],bounds[i,1],p)
            np.random.shuffle(sample[:,i])
        return sample

    for iter in range(iterations):
        bounds_sample = np.array([x_current-0.5*x_ranges,x_current+0.5*x_ranges])


        samples = LHS(bounds_sample.T,surrogate_size)

        f_sampled = []
        for sample in samples:
            f_sampled.append(f(sample))

        f_sampled = (np.array(f_sampled)-min(f_sampled))/(max(f_sampled)-min(f_sampled))
        x_current = samples[np.argsort(f_sampled)[0]]

        x_ranges *= 0.5

    dummy = {
        'f_best': f(x_current),
        'x_best': list(x_current),
        'runtime': time.time() - t0,
    }
###

    return dummy
