import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.util.model_size import build_model_size_report
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from data.planning.planning_sch_bilevel_lowdim import scheduling_data

np.random.seed(0)
materials = ['PA', 'PB', 'TEE', 'TGE', 'TI']
conc_end = {'PA': 3., 'PB': 3., 'TEE': 200., 'TGE': 200., 'TI': 10e3}
max_time = 0.53

# k1 = {m: 1/max_time*scheduling_data[None]['beta'][m+'1'] for m in materials}
# k2 = {m: 0.065/max_time*scheduling_data[None]['beta'][m+'1']*(1+np.random.normal(0,0.002)) for m in materials}
k1 = {m: 1 for m in materials}
k2 = {m: 0.065 for m in materials}
max_duration = {m: scheduling_data[None]['Bmax'][m+'1']*scheduling_data[None]['beta'][m+'1'] for m in materials}


def batch_reactor(conc, mat, log=False):
    m = pyo.ConcreteModel()
    m.t  = dae.ContinuousSet(bounds = (0,1))
    
    #Parameters
    m.vol = pyo.Param(initialize = 2.6260)
    
    #Variables
    m.CA = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100))
    m.CB = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100), initialize = 1e-8)
    m.CC = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100)) 
    m.energy = pyo.Var(m.t, within=pyo.NonNegativeReals)
    # m.tf = pyo.Param(initialize=5.)   
    m.tf = pyo.Var(within = pyo.Reals, bounds = (0,500), initialize =500)
    
    #Derivatives
    m.CA_dot = dae.DerivativeVar(m.CA, wrt = m.t)
    m.CB_dot = dae.DerivativeVar(m.CB, wrt = m.t)
    m.CC_dot = dae.DerivativeVar(m.CC, wrt = m.t)
    m.energy_dot = dae.DerivativeVar(m.energy, wrt=m.t)
    
    #Control variable
    m.u = pyo.Var(m.t, within = pyo.Reals, bounds = (1,9) , initialize = 5)
    
    # #Energy Cost
    # def _energy(m,t):
    #     return (m.vol**2*m.u[t]*m.tf)    
    # m.energy = dae.Integral(m.t, wrt = m.t, rule = _energy)

    def _dEnergydt(m, t):
        return (m.energy_dot[t] == m.vol**2*m.u[t]*m.tf)
    m.dEnergydt = pyo.Constraint(m.t, rule = _dEnergydt)
    
    #Mass Balance constraints
    def _dCAdt(m,t):
        return (m.CA_dot[t] == m.tf*(1.2836/max_duration[mat])*(-k1[mat]*m.u[t]*m.CA[t]))
    m.dCAdt =  pyo.Constraint(m.t, rule = _dCAdt)
     
    def _dCBdt(m,t):
        return (m.CB_dot[t] == m.tf*(1.2836/max_duration[mat])*(k1[mat]*m.u[t]*m.CA[t] -k2[mat]*m.CB[t]*m.u[t]**0.8))
    m.dCBdt =  pyo.Constraint(m.t, rule = _dCBdt)
    
    def _dCCdt(m,t):
        return (m.CC_dot[t] == m.tf*(1.2836/max_duration[mat])*(k2[mat]*m.CB[t]*m.u[t]**0.8))
    m.dCCdt =  pyo.Constraint(m.t, rule = _dCCdt)
    
    def _init_final(m):
        yield m.CA[0] == 17
        yield m.CB[0] == 0
        yield m.CC[0] == 0
        yield m.CB[1] >= 13/conc_end[mat]*conc
        yield m.CC[1] >= 2.5/conc_end[mat]*conc
    m.init_final = pyo.ConstraintList(rule =_init_final)
    
    m.objective = pyo.Objective(expr = 2*m.tf + 0.5*m.energy[1], sense = pyo.minimize)
    
    # Discretize and Solve model
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=20, ncp=5)
    #Piecewise Control ",
    discretizer.reduce_collocation_points(m, var = m.u , ncp =1, contset =m.t)
    
    if log:
        report = build_model_size_report(m)    
        print('Num constraints:' , report.activated.constraints)    
        print('Num variables: ', report.activated.variables)
        
    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = 6000
    results =  solver.solve(m,tee=False)
    # results =  solver.solve(m,tee=False).write()
    
    #Saving Data
    data = {v.getname(): dict() for v in m.component_objects(pyo.Var)}   
    data['time'] = [pyo.value(m.tf)*t for t in m.t]
    data['tproc'] = pyo.value(m.tf) 
    data['utility'] = pyo.value(m.energy[1])
    for var in m.component_objects(pyo.Var):
        if var.getname() != 'time':
            if var.getname() != 'tf' :
                data[var.getname()] = [pyo.value(var[t]) for t in m.t]
            else:
                data[var.getname()] = [pyo.value(var)]
    
    return (results,m,data)



def sampling(N_grid, material):
    volume =  np.linspace(0, conc_end[material], N_grid)
    columns = ['conc_end','tf','utility']
    df = {c: list() for c in columns}
    full_data = {}
    for i in range(len(volume)):
        results, m, data = batch_reactor(volume[i], material)
        full_data[str(volume[i])] = data
        df['conc_end'].append(volume[i])
        df['tf'].append(data['tproc'])
        df['utility'].append(data['utility'])
    return df, full_data

# results, m, data = batch_reactor(3)   
#  

if __name__=='__main__':
    N_samples = 10 # 120
    # for m in materials:
    for m in materials:
        dataframe, full_data = sampling(N_samples, m)
        #Saving Dataframe
        df = pd.DataFrame.from_dict(dataframe)
        try:
            dir = './data/CS2_sampling/Batch_Reactor_NN_'+m
            df.to_csv(dir) 
        except:
            dir = '../data/CS2_sampling/Batch_Reactor_NN_'+m
            df.to_csv(dir) 
