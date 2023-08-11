# import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.util.model_size import build_model_size_report
# from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from pyomo.opt import TerminationCondition

from data.planning.planning_sch_bilevel_lowdim import data, scheduling_data
from hierarchy.planning.Planning_Scheduling_bi import scheduling_Asia # , centralised
from sampling import batch_reactor

np.random.seed(0)


prod_list = [p for p in data[None]['P'][None] if p in scheduling_data[None]['states'][None]]

print(prod_list)

N_samples = 1000

upper_prod = 5e5 # implement better bounds for the upper limit on production
df = {p: [] for p in prod_list}
df['cost'] = []
df['feas'] = []

safe_storage = {
            "PA": data[None]['Istar0']['PA'],
            "PB": data[None]['Istar0']['PB'],
            "TEE": data[None]['Istar0']['TEE'],
            "TGE": data[None]['Istar0']['TGE'],
        }

dummy = np.array([1e5, 1e5, 1e7, 1e7])*10

upper_lim = {p: min([
        safe_storage[p]*10, 
        dummy[prod_list.index(p)], 
        5e5,
    ]) for p in prod_list}

for i in range(N_samples):
    input = scheduling_data.copy()
    for p in prod_list:
        prod = np.exp(np.random.random_sample()*np.log(upper_lim[p]))
        input[None]['Prod'][p] = prod
        df[p] += [prod]
    try:
        res_Sch, res = scheduling_Asia(input, res_out=True)
        changeover = pyo.value(res_Sch.CCH_cost)
        # print(res.solver.termination_condition, res.solver.status)
        # print('Before: ', res_Sch.MS.value)
        storage = pyo.value(res_Sch.st_cost)/20
        energy = 0
        feas = 1
        # print('Scheduling works at t=', t)
        for m in res_Sch.I:
            for n in res_Sch.N:
                if res_Sch.Ws[m,n].value==1:
                    # print(m,n,t)
                    # try:
                    res, _, data = batch_reactor(res_Sch.Bs[m, n].value, m[:-1])
                    # print(' works')
                    if (res.solver.termination_condition != TerminationCondition.infeasible) and (res.solver.termination_condition != TerminationCondition.infeasibleOrUnbounded):
                        energy += data['utility']/12000
                    else:
                        print('Control problem at:', m, n)
                        energy += 10/12*1e6

    except Exception as e:
        print('Scheduling problem due to: ')
        print(e)
        changeover = 100./12*1e6
        storage = 10/12*1e6
        energy = 10/12*1e6
        feas = 0

    # try:
    #     res_Sch = scheduling_Asia_bi_complete(scheduling_data)
    #     changeover = pyo.value(res_Sch.CCH_cost)
    #     storage = pyo.value(res_Sch.st_cost)/20
    #     energy = pyo.value(sum(
    #             res_Sch.energy_PA1[n] + res_Sch.energy_PA2[n] + \
    #             res_Sch.energy_PB1[n] + res_Sch.energy_PB2[n] + \
    #             res_Sch.energy_TEE1[n] + res_Sch.energy_TEE2[n] + \
    #             res_Sch.energy_TGE1[n] + res_Sch.energy_TGE2[n] + \
    #             res_Sch.energy_TI1[n] + res_Sch.energy_TI2[n] for n in res_Sch.N
    #         ))/12000
    #     feas = 1
    # except:
    #     changeover = 100/12*1e6
    #     storage = 10/12*1e6
    #     energy = 10/12*1e6
    #     feas = 0
    df['cost'] += [changeover + storage + energy]
    df['feas'] += [feas]


dataframe = pd.DataFrame.from_dict(df)
try:
    dir = './data/scheduling/scheduling_hierarch'
    dataframe.to_csv(dir) 
except:
    dir = '../data/scheduling/scheduling_hierarch'
    dataframe.to_csv(dir) 




    
