import json
import time
import numpy as np

from sample_real___ import wrapper_ as wrapper_real
from hierarchy.DFO_ext_all_integr import wrapper, wrapper_bi, wrapper_tri
from data.planning.planning_sch_bilevel_lowdim import data
from hierarchy.algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper

if __name__=='__main__':
    
    Nt = 5
    data_copy = data.copy()
    data_copy[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})

    b = []
    b += [(0, 20000)]*24 + [(0, 500000)]*24 + [(0, 20000)]*12 + [(0, 30000)]*12 + [(0, 10000)]*24 + [(0, 4000000)]*24
    b += [(0, 15000)]*24 + [(0, 10000)]*12 + [(0, 5000)]*12 + [(0, 1000)]*5 + [(0, 2000)]*5

    b = np.array([list(b_) for b_ in b], dtype=float)

    with open('./results/Optima/hierarch_RegNN_16_1.json') as f:
        tri_opt = json.load(f)
    
    def f_Py(x):
        return wrapper_real(x, data_copy, 1000), [0]
    
    # s = 'TR'
    # t0 = time.time()
    # TR = optimizer(dfo_tri_f, x_test, b, 80,  0.01)
    # t1 = time.time()
    # print(f"{s} done: Best tri eval after {(t1-t0)/60} min: {TR[0]}")
    
    s = 'Py-BOBYQA'
    t0 = time.perf_counter()
    pybobyqa = PyBobyqaWrapper().solve(
            f_Py,
            tri_opt['x'],
            bounds=np.array(b).T,
            maxfun=500,
            constraints=1,
            seek_global_minimum=True,
            objfun_has_noise=False,
            scaling_within_bounds=True,
    )  
    t1 = time.perf_counter()
    print(f"{s} done: Best hierarchical eval after {(t1-t0)/60} min: {pybobyqa['f_best_so_far'][-1]}")
    print(f"from initial eval {pybobyqa['f_best_so_far'][0]:.3f}")
    
    real = {
            'x': list(pybobyqa['x_best_so_far'][-1]),
            'time': (t1-t0)/60,
            'upper': {'time': None, 'obj': None},
            'bi': {'time': None, 'obj': None},
            'tri': {'time': None, 'obj': None},
            'real': {'time': None, 'obj': None},
        }
    
    dfo_f = lambda x: wrapper(x, data_copy, 1000)
    dfo_bi_f = lambda x: wrapper_bi(x, data_copy, 1000)
    dfo_tri_f = lambda x: wrapper_tri(x, data_copy, 1000)


    t0 = time.time()
    upper = dfo_f(pybobyqa['x_best_so_far'][-1])
    t1 = time.time()
    real['upper'] = {'time': t1 - t0, 'obj': upper}
    t0 = time.time()
    bi = dfo_bi_f(pybobyqa['x_best_so_far'][-1])
    t1 = time.time()
    real['bi'] = {'time': t1 - t0, 'obj': bi}
    t0 = time.time()
    tri = dfo_tri_f(pybobyqa['x_best_so_far'][-1])
    t1 = time.time()
    real['tri'] = {'time': t1 - t0, 'obj': tri}
    t0 = time.time()
    real_obj = wrapper_real(pybobyqa['x_best_so_far'][-1], data_copy, 1000)
    t1 = time.time()
    real['real'] = {'time': t1 - t0, 'obj': real_obj}
    
    with open('./results/Optima/hierarchical_surr3.json', 'w') as f:
        json.dump(real, f)
    
    
