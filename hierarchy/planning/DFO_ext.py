import argparse
import pickle
from re import L
from cvxpy import PSD_ATOMS
from functools import partial
import numpy as np
from scipy.optimize import minimize, differential_evolution, show_options
from skquant.opt import minimize as skopt
import time

import json

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

# from data.planning_extended import data
from data.planning.planning_sch_bilevel import scheduling_data, data
from hierarchy.planning.Planning_Scheduling import f_Europe, f_Asia, f_America, centralised
from hierarchy.planning.Planning_Extended_dyn import simulate, state_to_control_t
from hierarchy.planning.Planning_Extended_dyn import centralised as centralised_planning
from hierarchy.planning.Planning_Scheduling_bi import scheduling_Asia # , centralised

from hierarchy.algorithms.PyBobyqa_wrapped.Wrapper_for_pybobyqa import PyBobyqaWrapper
from hierarchy.algorithms.DIRECT_wrapped.Wrapper_for_Direct import DIRECTWrapper

def get_Forecast(data):

    T_set = data[None]["T"][None]
    N_t = len(T_set)
    F_PA = np.zeros(N_t)
    F_PB = np.zeros(N_t)
    F_PC = np.zeros(N_t)
    F_PD = np.zeros(N_t)
    F_TEE = np.zeros(N_t)
    F_TGE = np.zeros(N_t)
    for t in T_set:
        F_PA[t-1] = data[None]['F']['PA',t]
        F_PB[t-1] = data[None]['F']['PB',t]
        F_PC[t-1] = data[None]['F']['PC',t]
        F_PD[t-1] = data[None]['F']['PD',t]
        F_TEE[t-1] = data[None]['F']['TEE',t]
        F_TGE[t-1] = data[None]['F']['TGE',t]

    Forecast = {}
    Forecast['PA'] = F_PA
    Forecast['PC'] = F_PC
    Forecast['TEE'] = F_TEE
    Forecast['PB'] = F_PB
    Forecast['PD'] = F_PD
    Forecast['TGE'] = F_TGE

    return Forecast


def get_cost(input_data, Production, TP, Sales, penalization=1):
    
    T_set = input_data[None]["T"][None]
    N_t = len(T_set)
    N_Tc = input_data[None]["N_t"][None]

    to_Tc = lambda t: state_to_control_t(t, N_Tc, T_set)

    Forecast = get_Forecast(input_data)

    Storage, Demand, Dummy_Sales = simulate(Production, TP, Forecast, Sales, input_data, random=False)

    P_set = input_data[None]['P'][None]
    R_set = input_data[None]['R'][None]
    T_set = input_data[None]['T'][None]
    L_set = input_data[None]['L'][None]
    
    Price = input_data[None]['Price']
    CP = input_data[None]['CP']
    CS = input_data[None]['CS']
    CS_I = input_data[None]['CS_I'][None]
    CS_SAIS = input_data[None]['CS_SAIS']
    CS_AIP = input_data[None]['CS_AIP'][None]
    CT = input_data[None]['CT']
    CP_I = input_data[None]['CP_I'][None]
    CP_AI = input_data[None]['CP_AI'][None]

    U = input_data[None]['U']
    X = input_data[None]['X']
    A = input_data[None]['A']
    Q = input_data[None]['Q']
    IIstar0 = input_data[None]['IIstar0'][None]
    Istar0 = input_data[None]['Istar0']
    IAISstar0 = input_data[None]['IAISstar0']
    IAIPstar0 = input_data[None]['IAIPstar0'][None]

    prod_cost = sum(
        CP[p]*Production[p][t-1] for p in P_set for t in T_set
    ) + sum(
        CP_I*Production['I'][t-1] for t in T_set
    ) + sum(
        CP_AI*Production['AI'][t-1] for t in T_set
    )
    transp_cost = sum(CT[l]*TP[l][to_Tc(t)-1] for l in L_set for t in T_set)
    store_cost = np.sum(
        [CS[p] * Storage[p][t-1] for p in P_set for t in T_set]
    ) + np.sum(
        [CS_SAIS['Asia'] * Storage['AIS_As'][t-1] for t in T_set]
    ) + np.sum(
        [CS_SAIS['America'] * Storage['AIS_Am'][t-1] for t in T_set]
    ) + np.sum(
        [CS_AIP * Storage['AIP'][t-1] for t in T_set]
    ) + np.sum(
        [CS_I * Storage['I'][t-1] for t in T_set]
    )

    for p in P_set:
        for t in T_set:
            assert Storage[p][t-1] >= -1e-5

    sales = np.sum([Price[p] * Dummy_Sales[p][t-1] for p in P_set for t in T_set])
    
    res_viol = sum(
        max(0, U[l, r]* sum(
                Production[p][t-1]*X[l, p]*Q[p] for p in P_set
            )/A[l, r] - 1
        )**2 for l in L_set for r in R_set for t in T_set
    )

    prod_UL = sum(
        max(
            0, Production[p][t-1]/500e3-1
        )**2 for p in P_set for t in T_set
    ) + sum(
        max(
            0, Production['AI'][t-1]/500e3-1
        )**2 for t in T_set
    ) + sum(
        max(
            0, Production['I'][t-1]/500e3-1
        )**2 for t in T_set
    )

    istar_constr = 0
    for t in T_set:
        if t > 4:
            istar_constr += sum(max(- Storage[p][t-1]/Istar0[p]+1, 0)**2 for p in P_set)
            istar_constr += max(- Storage['I'][t-1]/IIstar0+1, 0)**2
            istar_constr += max(- Storage['AIS_Am'][t-1]/IAISstar0["America"]+1, 0)**2
            istar_constr += max(- Storage['AIS_As'][t-1]/IAISstar0["Asia"]+1, 0)**2 
            istar_constr += max(- Storage['AIP'][t-1]/IAIPstar0+1, 0)**2 
        else:
            istar_constr += sum(max(- Storage[p][t-1]/Istar0[p]+1/4, 0)**2 for p in P_set)
            istar_constr += max(- Storage['I'][t-1]/IIstar0+1/4, 0)**2
            istar_constr += max(- Storage['AIS_Am'][t-1]/IAISstar0["America"]+1/4, 0)**2
            istar_constr += max(- Storage['AIS_As'][t-1]/IAISstar0["Asia"]+1/4, 0)**2 
            istar_constr += max(- Storage['AIP'][t-1]/IAIPstar0+1/4, 0)**2 

    ### Add safeS, safeSI, safeSAIS, safeSAIP

    penalty = istar_constr + prod_UL + res_viol

    return (prod_cost + transp_cost + store_cost - sales)/1e6 + penalty*penalization

def wrapper(x, input_data, penalty=1):
    T_set = input_data[None]["T"][None]
    N_t = len(T_set)
    N_Tc = input_data[None]["N_t"][None]

    to_Tc = lambda t: state_to_control_t(t, N_Tc, T_set)

    P_PA = x[:N_t]
    P_PB = x[N_t:2*N_t]
    P_TEE = x[2*N_t:3*N_t]
    P_TGE = x[3*N_t:4*N_t]
    P_PC = x[4*N_t:5*N_t]
    P_PD = x[5*N_t:6*N_t]

    SA_PA =  x[6*N_t:7*N_t]
    SA_PB =  x[7*N_t:8*N_t]
    SA_TEE = x[8*N_t:9*N_t]
    SA_TGE = x[9*N_t:10*N_t]
    SA_PC =  x[10*N_t:11*N_t]
    SA_PD =  x[11*N_t:12*N_t]

    PI = x[12*N_t:13*N_t]
    PAI = x[13*N_t:14*N_t]

    TP_As = x[14*N_t:14*N_t+1*N_Tc]
    TP_Am = x[14*N_t+1*N_Tc:14*N_t+2*N_Tc]

    Production = {} ; TP = {} ; Sales = {}
    Production['PA'] = P_PA
    Production['PC'] = P_PC
    Production['TEE'] = P_TEE
    Production['PB'] = P_PB
    Production['PD'] = P_PD
    Production['TGE'] = P_TGE
    Production['AI'] = PAI
    Production['I'] = PI
    Sales['PA'] =  SA_PA
    Sales['PC'] =  SA_PC
    Sales['TEE'] = SA_TEE
    Sales['PB'] =  SA_PB
    Sales['PD'] =  SA_PD
    Sales['TGE'] = SA_TGE
    TP['Asia'] = TP_As
    TP['America'] = TP_Am

    return get_cost(input_data, Production, TP, Sales, penalization=penalty)

def get_cost_bi(input_data, Production, TP, Sales, penalization=1):
    
    T_set = input_data[None]["T"][None]
    N_t = len(T_set)
    N_Tc = input_data[None]["N_t"][None]

    to_Tc = lambda t: state_to_control_t(t, N_Tc, T_set)

    Forecast = get_Forecast(input_data)

    Storage, Demand, Dummy_Sales = simulate(Production, TP, Forecast, Sales, input_data, random=False)

    P_set = input_data[None]['P'][None]
    R_set = input_data[None]['R'][None]
    T_set = input_data[None]['T'][None]
    L_set = input_data[None]['L'][None]
    
    Price = input_data[None]['Price']
    CP = input_data[None]['CP']
    CS = input_data[None]['CS']
    CS_I = input_data[None]['CS_I'][None]
    CS_SAIS = input_data[None]['CS_SAIS']
    CS_AIP = input_data[None]['CS_AIP'][None]
    CT = input_data[None]['CT']
    CP_I = input_data[None]['CP_I'][None]
    CP_AI = input_data[None]['CP_AI'][None]

    U = input_data[None]['U']
    X = input_data[None]['X']
    A = input_data[None]['A']
    Q = input_data[None]['Q']
    IIstar0 = input_data[None]['IIstar0'][None]
    Istar0 = input_data[None]['Istar0']
    IAISstar0 = input_data[None]['IAISstar0']
    IAIPstar0 = input_data[None]['IAIPstar0'][None]

    P_As = []
    for p in P_set:
        if X['Asia', p]:
            P_As.append(p)

    for p in P_As:
        if p in scheduling_data[None]['states'][None]:
            scheduling_data[None]['Prod'][p] = np.average([Production[p][t-1] for t in T_set])
    ### 
    # print('Average input to scheduling problem: ', scheduling_data[None]['Prod'])
    try:
        res_Sch = scheduling_Asia(scheduling_data)
        changeover = pyo.value(res_Sch.st_cost)*24
    except:
        changeover = 100.
    # storage = pyo.value(res_Sch.CCH_cost)*24

    prod_cost = sum(
        CP[p]*Production[p][t-1] for p in P_set for t in T_set
    ) + sum(
        CP_I*Production['I'][t-1] for t in T_set
    ) + sum(
        CP_AI*Production['AI'][t-1] for t in T_set
    )
    transp_cost = sum(CT[l]*TP[l][to_Tc(t)-1] for l in L_set for t in T_set)
    store_cost = np.sum(
        [CS[p] * Storage[p][t-1] for p in P_set for t in T_set]
    ) + np.sum(
        [CS_SAIS['Asia'] * Storage['AIS_As'][t-1] for t in T_set]
    ) + np.sum(
        [CS_SAIS['America'] * Storage['AIS_Am'][t-1] for t in T_set]
    ) + np.sum(
        [CS_AIP * Storage['AIP'][t-1] for t in T_set]
    ) + np.sum(
        [CS_I * Storage['I'][t-1] for t in T_set]
    )

    for p in P_set:
        for t in T_set:
            assert Storage[p][t-1] >= -1e-5

    sales = np.sum([Price[p] * Dummy_Sales[p][t-1] for p in P_set for t in T_set])
    
    res_viol = sum(
        max(0, U[l, r]* sum(
                Production[p][t-1]*X[l, p]*Q[p] for p in P_set
            )/A[l, r] - 1
        )**2 for l in L_set for r in R_set for t in T_set
    )

    prod_UL = sum(
        max(
            0, Production[p][t-1]/500e3-1
        )**2 for p in P_set for t in T_set
    ) + sum(
        max(
            0, Production['AI'][t-1]/500e3-1
        )**2 for t in T_set
    ) + sum(
        max(
            0, Production['I'][t-1]/500e3-1
        )**2 for t in T_set
    )

    istar_constr = 0
    for t in T_set:
        if t > 4:
            istar_constr += sum(max(- Storage[p][t-1]/Istar0[p]+1, 0)**2 for p in P_set)
            istar_constr += max(- Storage['I'][t-1]/IIstar0+1, 0)**2
            istar_constr += max(- Storage['AIS_Am'][t-1]/IAISstar0["America"]+1, 0)**2
            istar_constr += max(- Storage['AIS_As'][t-1]/IAISstar0["Asia"]+1, 0)**2 
            istar_constr += max(- Storage['AIP'][t-1]/IAIPstar0+1, 0)**2 
        else:
            istar_constr += sum(max(- Storage[p][t-1]/Istar0[p]+1/4, 0)**2 for p in P_set)
            istar_constr += max(- Storage['I'][t-1]/IIstar0+1/4, 0)**2
            istar_constr += max(- Storage['AIS_Am'][t-1]/IAISstar0["America"]+1/4, 0)**2
            istar_constr += max(- Storage['AIS_As'][t-1]/IAISstar0["Asia"]+1/4, 0)**2 
            istar_constr += max(- Storage['AIP'][t-1]/IAIPstar0+1/4, 0)**2 

    ### Add safeS, safeSI, safeSAIS, safeSAIP

    penalty = istar_constr + prod_UL + res_viol

    return (prod_cost + transp_cost + store_cost + changeover - sales)/1e6 + penalty*penalization

def wrapper_bi(x, input_data, penalty=1):
    T_set = input_data[None]["T"][None]
    N_t = len(T_set)
    N_Tc = input_data[None]["N_t"][None]

    to_Tc = lambda t: state_to_control_t(t, N_Tc, T_set)

    P_PA = x[:N_t]
    P_PB = x[N_t:2*N_t]
    P_TEE = x[2*N_t:3*N_t]
    P_TGE = x[3*N_t:4*N_t]
    P_PC = x[4*N_t:5*N_t]
    P_PD = x[5*N_t:6*N_t]

    SA_PA =  x[6*N_t:7*N_t]
    SA_PB =  x[7*N_t:8*N_t]
    SA_TEE = x[8*N_t:9*N_t]
    SA_TGE = x[9*N_t:10*N_t]
    SA_PC =  x[10*N_t:11*N_t]
    SA_PD =  x[11*N_t:12*N_t]

    PI = x[12*N_t:13*N_t]
    PAI = x[13*N_t:14*N_t]

    TP_As = x[14*N_t:14*N_t+1*N_Tc]
    TP_Am = x[14*N_t+1*N_Tc:14*N_t+2*N_Tc]

    Production = {} ; TP = {} ; Sales = {}
    Production['PA'] = P_PA
    Production['PC'] = P_PC
    Production['TEE'] = P_TEE
    Production['PB'] = P_PB
    Production['PD'] = P_PD
    Production['TGE'] = P_TGE
    Production['AI'] = PAI
    Production['I'] = PI
    Sales['PA'] =  SA_PA
    Sales['PC'] =  SA_PC
    Sales['TEE'] = SA_TEE
    Sales['PB'] =  SA_PB
    Sales['PD'] =  SA_PD
    Sales['TGE'] = SA_TGE
    TP['Asia'] = TP_As
    TP['America'] = TP_Am

    return get_cost_bi(input_data, Production, TP, Sales, penalization=penalty)

Nt = 5

data_copy = data.copy()
data_copy[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
res = centralised(data_copy)
f_actual = pyo.value(res.obj)
print(f"Centralised bilevel objective: {f_actual}")

data_copy = data.copy()
data_copy[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
res = centralised_planning(data_copy)
f_actual = pyo.value(res.obj)
print(f"Centralised planning objective: {f_actual}")

x_test = []
x_test += [pyo.value(res.Prod['PA', 'Asia',t]) for t in res.T]
x_test += [pyo.value(res.Prod['PB', 'Asia',t]) for t in res.T]
x_test += [pyo.value(res.Prod['TEE', 'Asia',t]) for t in res.T]
x_test += [pyo.value(res.Prod['TGE', 'Asia',t]) for t in res.T]
x_test += [pyo.value(res.Prod['PC','America',t]) for t in res.T]
x_test += [pyo.value(res.Prod['PD','America',t]) for t in res.T]

x_test += [pyo.value(res.SA['PA',  t]) for t in res.T]
x_test += [pyo.value(res.SA['PB',  t]) for t in res.T]
x_test += [pyo.value(res.SA['TEE', t]) for t in res.T]
x_test += [pyo.value(res.SA['TGE', t]) for t in res.T]
x_test += [pyo.value(res.SA['PC',  t]) for t in res.T]
x_test += [pyo.value(res.SA['PD',  t]) for t in res.T]

x_test += [pyo.value(res.PI[t]) for t in res.T]
x_test += [pyo.value(res.PAI[t]) for t in res.T]
x_test += [pyo.value(res.TP['Asia',tc]) for tc in res.Tc]
x_test += [pyo.value(res.TP['America',tc]) for tc in res.Tc]

with open('./results/optima/centralized_high_dim.json', 'w') as f:
    json.dump(x_test, f)


bounds1 = np.array(x_test)[:336].reshape((14, 24))
bounds2 = np.array(x_test)[336:].reshape((2, 5))
bounds = np.concatenate((np.max(bounds1, axis=1), np.max(bounds2, axis=1)))
bounds = [(0, b) for b in bounds]

b = []
b += [(0, 20000)]*48 + [(0, 500000)]*48 + [(0, 20000)]*24 + [(0, 30000)]*24 + [(0, 10000)]*48 + [(0, 4000000)]*48
b += [(0, 15000)]*48 + [(0, 10000)]*24 + [(0, 5000)]*24 + [(0, 1000)]*5 + [(0, 2000)]*5

DFO_obj = wrapper(x_test, data_copy, 100)

t0 = time.time()
DFO_bi_obj = wrapper_bi(x_test, data_copy, 100)
t1 = time.time()



dfo_bi_f = lambda x: wrapper_bi(x, data_copy, 100)

x0 = np.array([max(min(x_test[i]*(1+0.005*np.random.normal()),b[i][1]),0) for i in range(len(x_test))])


init_guess = dfo_f(x0)
init_bi_guess = dfo_bi_f(x_test)

print('Centralised optimal DFO objective: ', DFO_obj)
print('Centralised bi-level upper objective: ', DFO_bi_obj, ' in ', t1-t0, ' seconds')



# t0 = time.time()
# test = minimize(dfo_bi_f, x_test, bounds = b, args=(), method='Nelder-Mead', constraints=(), tol=None, callback=None, options={'maxfev': 3})
# # test = minimize(dfo_f, x0, args=(), method='Nelder-Mead', constraints=(), tol=None, callback=None, options={'maxfev': 10000})
# t1 = time.time()
# print('Test DFO optimum: ', test['fun'], ' in ', t1-t0, ' seconds starting from initial value of: ', init_guess)
# print()

# t0 = time.time()
# test = minimize(dfo_f, x0, bounds = b, args=(), method='Nelder-Mead', constraints=(), tol=None, callback=None, options={'maxfev': 70000})
# # test = minimize(dfo_f, x0, args=(), method='Nelder-Mead', constraints=(), tol=None, callback=None, options={'maxfev': 10000})
# t1 = time.time()
# diff = test.x - x_test
# dfo_f(test.x)
# print('Test DFO optimum: ', test['fun'], ' in ', t1-t0, ' seconds starting from initial value of: ', init_guess)
# print()

# show_options(solver='minimize')



b = np.array([list(b_) for b_ in b], dtype=float)

s = 'Nomad'
t0 = time.time()
result, history = \
    skopt(dfo_f, x0, b, 10000, method='nomad', SEED=0)
t1 = time.time()
print(f"{s} done: Best eval after {(t1-t0)/60} min for {10000} evals: {result.optval} with initial guess {init_guess}")

# s = 'Diff. evol.'
# t0 = time.time()
# result = differential_evolution(dfo_f, b, popsize=20, maxiter=15)
# t1 = time.time()
# print(s, result.fun, (t1-t0), ' seconds from init guess: ', init_guess)
# print(s, 'Done')


# s = 'Py-BOBYQA'
# def f_Py(x):
#     return dfo_f(x), [0]
# t0 = time.time()
# pybobyqa = PyBobyqaWrapper().solve(
#         f_Py,
#         x0,
#         bounds=np.array(b).T,
#         maxfun=1000,
#         constraints=1,
#         seek_global_minimum=True,
#         objfun_has_noise=False,
#         scaling_within_bounds=True,
# )  
# t1 = time.time()
# print(f"{s} done: Best eval after {(t1-t0)/60} min: {pybobyqa['f_best_so_far'][-1]}")

# t0 = time.time()
# s = 'DIRECT-L'
# def f_DIR(x, grad):
#     return dfo_f(x), [0]
# DIRECT = DIRECTWrapper().solve(f_DIR, x0, b, maxfun=5000, constraints=1)
# t1 = time.time()
# # DIRECT['f_best_so_far'] = preprocess_BO(DIRECT['f_best_so_far'], y0[0])
# print('DIRECT-L', DIRECT['f_best_so_far'][-1], (t1-t0), ' seconds from init guess: ', init_guess)
# print(s, 'Done')

# t0 = time.time()
# s = 'DIRECT-L'
# def f_DIR(x, grad):
#     return dfo_bi_f(x), [0]
# DIRECT = DIRECTWrapper().solve(f_DIR, x_test, b, maxfun=1000, constraints=1)
# t1 = time.time()
# # DIRECT['f_best_so_far'] = preprocess_BO(DIRECT['f_best_so_far'], y0[0])
# print('DIRECT-L', DIRECT['f_best_so_far'][-1], (t1-t0), ' seconds from init guess: ', init_bi_guess)
# print(s, 'Done')



# print('Bi-level DFO optimum on original DFO function: ', wrapper(DIRECT['x_best_so_far'][-1], data_copy, 100))
# print('Bi-level DFO optimum on bi-level DFO function: ', wrapper_bi(DIRECT['x_best_so_far'][-1], data_copy, 100))

