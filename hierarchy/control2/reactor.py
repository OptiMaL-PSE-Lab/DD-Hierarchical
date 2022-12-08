import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.util.model_size import build_model_size_report
from matplotlib.ticker import FormatStrFormatter

m = pyo.ConcreteModel()
m.t  = dae.ContinuousSet(bounds = (0,1))

#Parameters
m.vol = pyo.Param(initialize = 2.6260) # fixed
m.Q =pyo.Param(initialize = 27.4967) # output
#Variables
m.CA = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100)) # input
m.CB = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100), initialize = 1e-8)
m.CC = pyo.Var(m.t, within = pyo.Reals, bounds = (0,100))
m.time = pyo.Var(m.t)
m.time[0].fix(0)    
m.tf = pyo.Var(within = pyo.Reals, bounds = (0,5), initialize =5)

#Derivatives
m.CA_dot = dae.DerivativeVar(m.CA, wrt = m.t)
m.CB_dot = dae.DerivativeVar(m.CB, wrt = m.t)
m.CC_dot = dae.DerivativeVar(m.CC, wrt = m.t)
m.time_dot = dae.DerivativeVar(m.time, wrt = m.t)

#Control variable
m.u = pyo.Var(m.t, within = pyo.Reals, bounds = (1,9) , initialize = 5)

#Energy Cost
def _energy(m,t):
    return (m.vol**2*m.tf*m.u[t])    
m.energy = dae.Integral(m.t, wrt = m.t, rule = _energy)

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

def _difftime(m,t):
    return(m.time_dot[t] == m.tf)
m.difftime = pyo.Constraint(m.t, rule = _difftime)

def _init_final(m):
    yield m.CA[0] == 17
    yield m.CB[0] == 0
    yield m.CC[0] == 0
    yield m.CB[1] >= 13
    yield m.CC[1] >= 2.5 
    yield m.energy == m.Q
m.init_final = pyo.ConstraintList(rule =_init_final)

m.objective = pyo.Objective(expr = 2*m.tf + 0.5*m.energy, sense = pyo.minimize)

# Discretize and Solve model
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=10, ncp=4)
#Piecewise Control ",
discretizer.reduce_collocation_points(m, var = m.u , ncp =1, contset =m.t)

report = build_model_size_report(m)    
print('Num constraints:' , report.activated.constraints)    
print('Num variables: ', report.activated.variables)
solver = pyo.SolverFactory('ipopt')
solver.options['max_iter'] = 6000
results =  solver.solve(m).write()

#Saving Data
data = {v.getname(): dict() for v in m.component_objects(pyo.Var)}   
data['time'] = [pyo.value(m.tf)*t for t in m.t]
data['tproc'] = pyo.value(m.tf) 
for var in m.component_objects(pyo.Var):
    if var.getname() != 'tf' :
            data[var.getname()] = [pyo.value(var[t]) for t in m.t]
    else:
            data[var.getname()] = [pyo.value(var)]
            
fig1, ax1 = plt.subplots(2, figsize= (6.5,6.5))
concentrations = ['CA','CB','CC']
ax1[0].plot(data['time'], data['CA'], label = 'A', color = 'r')
ax1[0].plot(data['time'], data['CB'], label = 'B', color = 'b')
ax1[0].plot(data['time'], data['CC'], label = 'C', color = 'g')
ax1[0].set_xlim(0,data['tproc'])
ax1[0].set_ylim(0, max(data['CA']))
ax1[0].set_ylabel('Concetration $\\rm[mol/m^3]$')
# ax1[0].set_xlabel('Time $\\rm[h]$')
ax1[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax1[0].grid(alpha =0.2 ,ls ='--')
ax1[0].legend()


ax1[1].plot(data['time'], data['u'], label = 'u', color = 'black')
ax1[1].set_xlim(0,data['tproc'])
ax1[1].set_ylim(min(data['u']), 1.1*max(data['u']))
ax1[1].grid(alpha =0.2 ,ls ='--')
ax1[1].set_ylabel('Control profile $\\rm[uU/\ h.m^3]$')
ax1[1].set_xlabel('Time $\\rm[h]$')
ax1[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax1[1].legend()
plt.show()
# fig1.savefig('Random_kinetics.png', dpi = 600,format = 'png',bbox_inches  = 'tight')  
print(m.tf())