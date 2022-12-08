import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.util.model_size import build_model_size_report
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

def batch_reactor(volume):
    m = pyo.ConcreteModel()
    m.t  = dae.ContinuousSet(bounds = (0,1))
    
    #Parameters
    m.vol = pyo.Param(initialize = volume)
    
    #Variables
    m.CA = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100))
    m.CB = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100), initialize = 1e-8)
    m.CC = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100)) 
    m.energy = pyo.Var(m.t, within=pyo.NonNegativeReals)
    # m.tf = pyo.Param(initialize=5.)   
    m.tf = pyo.Var(within = pyo.Reals, bounds = (0,5), initialize =5)
    
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
        return (m.CA_dot[t] == m.tf*(-m.u[t]*m.CA[t]))
    m.dCAdt =  pyo.Constraint(m.t, rule = _dCAdt)
     
    def _dCBdt(m,t):
        return (m.CB_dot[t] == m.tf*(m.u[t]*m.CA[t] - 0.065*m.CB[t]*m.u[t]**0.8))
    m.dCBdt =  pyo.Constraint(m.t, rule = _dCBdt)
    
    def _dCCdt(m,t):
        return (m.CC_dot[t] == m.tf*(0.065*m.CB[t]*m.u[t]**0.8))
    m.dCCdt =  pyo.Constraint(m.t, rule = _dCCdt)
    
    def _init_final(m):
        yield m.CA[0] == 17
        yield m.CB[0] == 0
        yield m.CC[0] == 0
        yield m.CB[1] >= 13
        yield m.CC[1] >= 2.5
    m.init_final = pyo.ConstraintList(rule =_init_final)
    
    m.objective = pyo.Objective(expr = 2*m.tf + 0.5*m.energy[1], sense = pyo.minimize)
    
    # Discretize and Solve model
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=20, ncp=5)
    #Piecewise Control ",
    discretizer.reduce_collocation_points(m, var = m.u , ncp =1, contset =m.t)
    
    report = build_model_size_report(m)    
    print('Num constraints:' , report.activated.constraints)    
    print('Num variables: ', report.activated.variables)
    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = 6000
    results =  solver.solve(m,tee=False).write()
    
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



def sampling(N_grid):
    volume =  np.linspace(0, 7.5, N_grid)
    columns = ['volume','tf','utility']
    df = {c: list() for c in columns}
    full_data = {}
    for i in range(len(volume)):
        results, m, data = batch_reactor(volume[i])
        full_data[str(volume[i])] = data
        df['volume'].append(volume[i])
        df['tf'].append(data['tproc'])
        df['utility'].append(data['utility'])
    return df, full_data
    
# results, m, data = batch_reactor(3)   
#  
N_samples = 120 # 120
dataframe, full_data = sampling(N_samples)
#Saving Dataframe
df = pd.DataFrame.from_dict(dataframe)
try:
    dir = './data/CS1_sampling/Batch_Reactor_NN'
    df.to_csv(dir) 
except:
    dir = '../data/CS1_sampling/Batch_Reactor_NN'
    df.to_csv(dir) 
