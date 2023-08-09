import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.util.model_size import build_model_size_report
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from data.planning.planning_sch_bilevel_lowdim import data, scheduling_data
from hierarchy.planning.Planning_Scheduling_bi import scheduling_Asia

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
    # res_Sch = scheduling_Asia(scheduling_data)
    try:
        res_Sch = scheduling_Asia(scheduling_data)
        changeover = pyo.value(res_Sch.CCH_cost)
        storage = pyo.value(res_Sch.st_cost)/20
        feas = 1
    except:
        changeover = 100/12*1e6
        storage = 10/12*1e6
        feas = 0
    df['cost'] += [changeover + storage]
    df['feas'] += [feas]

dataframe = pd.DataFrame.from_dict(df)
try:
    dir = './data/scheduling/scheduling_test'
    dataframe.to_csv(dir) 
except:
    dir = '../data/scheduling/scheduling_test'
    dataframe.to_csv(dir) 

