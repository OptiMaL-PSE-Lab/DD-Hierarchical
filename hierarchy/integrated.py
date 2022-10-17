import pyomo.environ as pyo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyomo.dae as dae

data3 = {
        'TIME': 12,
        'N_EVENTS':  7,
        'STATES'  :{
            'S1'   :   {'initial':100,  's_price':   0, 'b_price':  60, 'storage': 50},
            'S2'   :   {'initial':  0,  's_price':   0, 'b_price':   0, 'storage': 0 },
            'S3'   :   {'initial':  0,  's_price':   0, 'b_price':   0, 'storage': 0 },
            'S4'   :   {'initial':  0,  's_price':   0, 'b_price':   0, 'storage': 0 },
            'S5'   :   {'initial':  0,  's_price': 120, 'b_price':   0, 'storage': 50 }
            },
        'ST_ARCS':{
            ('S1', 'Mixing')      : {'rho': 1.0},
            ('S2', 'Reaction')    : {'rho': 1.0},
            ('S3', 'Separation')  : {'rho': 1.0},
            },
        'TS_ARCS': {
            ('Mixing',    'S2')   : {'dur': 1.8,'rho': 1.0},
            ('Reaction',  'S3')   : {'dur': 2.4,'rho': 1.0},
            ('Separation','S4')   : {'dur': 1.8,'rho': 0.1},
            ('Separation','S5')   : {'dur': 1.8,'rho': 0.9}
            },
        'UNIT_TASKS':{
            ('Mixer',   'Mixing')       : {'Bmin': 2, 'Bmax': 5},
            ('Reactor',   'Reaction')   : {'Bmin': 2, 'Bmax': 5},
            ('Separator', 'Separation') : {'Bmin': 2, 'Bmax': 5},
            },
        'SEQ': [('Mixer','Reactor'),('Reactor','Separator')]        
        }

STN = data3

STATES = STN['STATES']
ST_ARCS = STN['ST_ARCS']
TS_ARCS = STN['TS_ARCS']
UNIT_TASKS = STN['UNIT_TASKS']
EVENTS = STN['N_EVENTS']
SEQ = STN['SEQ']
H = STN['TIME']

# list of tasks
TASKS = list([i for (j, i) in UNIT_TASKS])
#list of units
UNITS = list([j for (j, i) in UNIT_TASKS])

# K[i] set of units capable of task i indexed by unit
K = {j: set() for j in UNITS}
for (j, i) in UNIT_TASKS:
    K[j].add(i)

# K[i] set of units capable of task i indexed by task
L = {i: set() for i in TASKS}
for (j, i) in UNIT_TASKS:
    L[i].add(j)

# Bmax[(i,j)] maximum capacity of unit j for task i
Bmax = {(i, j): UNIT_TASKS[(j, i)]['Bmax'] for (j, i) in UNIT_TASKS}

# Bmin[(i,j)] minimum capacity of unit j for task i
Bmin = {(i, j): UNIT_TASKS[(j, i)]['Bmin'] for (j, i) in UNIT_TASKS}

# Duration of task for producing state s
dur = {(i): TS_ARCS[(i, s)]['dur'] for (i, s) in TS_ARCS}


# rho[(i,s)] input fraction of task i from state s
rho = {(i, s): ST_ARCS[(s, i)]['rho'] for (s, i) in ST_ARCS}

# rho_[(i,s)] output fraction of task i to state s
rho_ = {(i, s): TS_ARCS[(i, s)]['rho'] for (i, s) in TS_ARCS}


# T[s] set of tasks receiving material from state s
T = {s: set() for s in STATES}
for (s, i) in ST_ARCS:
    T[s].add(i)

# set of tasks producing material for state s
T_ = {s: set() for s in STATES}
for (i, s) in TS_ARCS:
    T_[s].add(i)

#Storage max
storage = {s: set() for s in STATES}
for s in STATES:
    storage[s] = STATES[s]['storage']

#Initial amount
initial = {s: set() for s in STATES}
for s in STATES:
    initial[s] = STATES[s]['initial']

#prices
sell_p = {s: set() for s in STATES}
buy_p = {s: set() for s in STATES}
for s in STATES:
    sell_p[s] = STATES[s]['s_price']
    buy_p[s] = STATES[s]['b_price']
    
#Integrated Scheduling and Dynamic Optimization
m = pyo.ConcreteModel()

#Indexes
m.TASKS = pyo.Set(initialize  = TASKS) 
m.UNITS = pyo.Set(initialize  = UNITS) 
m.STATES = pyo.Set(initialize = STATES.keys()) 
m.EVENTS = pyo.RangeSet(0,      EVENTS)

#Variables
m.wv    = pyo.Var(m.TASKS,  m.EVENTS, within = pyo.Boolean)
m.yv    = pyo.Var(m.UNITS,  m.EVENTS, within = pyo.Boolean)
m.vol   = pyo.Var(m.TASKS,  m.UNITS,  m.EVENTS, domain = pyo.NonNegativeReals)
m.ST    = pyo.Var(m.STATES, m.EVENTS, domain = pyo.NonNegativeReals)
m.Ts    = pyo.Var(m.TASKS,  m.UNITS,  m.EVENTS, domain = pyo.NonNegativeReals)
m.Tf    = pyo.Var(m.TASKS,  m.UNITS,  m.EVENTS, domain = pyo.NonNegativeReals)
m.STin = pyo.Var(m.STATES, domain = pyo.NonNegativeReals)

#Dynamic model for Reactor
D_UNITS = ['Reactor']
D_TASKS = ['Reaction']
m.tau  = dae.ContinuousSet(bounds = (0,1))
m.time = pyo.Var(D_UNITS, m.EVENTS,m.tau)
m.tf = pyo.Var(D_UNITS, m.EVENTS, within = pyo.Reals, bounds = (0,10), initialize =5)
for j in D_UNITS:
    for n in m.EVENTS:
        m.time[j,n,0].fix(0)
#Variables
m.CA = pyo.Var(D_UNITS, m.EVENTS, m.tau, within = pyo.Reals, bounds = (0,100))
m.CB = pyo.Var(D_UNITS, m.EVENTS, m.tau, within = pyo.Reals, bounds = (0,100), initialize = 1e-8)
m.CC = pyo.Var(D_UNITS, m.EVENTS, m.tau, within = pyo.Reals, bounds = (0,100))    
#Derivatives
m.CA_dot = dae.DerivativeVar(m.CA, wrt = m.tau)
m.CB_dot = dae.DerivativeVar(m.CB, wrt = m.tau)
m.CC_dot = dae.DerivativeVar(m.CC, wrt = m.tau)
m.dtime = dae.DerivativeVar(m.time, wrt = m.tau)

#Control variable
m.u = pyo.Var(D_UNITS, m.EVENTS, m.tau, within = pyo.Reals, bounds = (1,9) , initialize = 5)
#Energy Cost, 1 per event
def _energy_0(m,t):
    return (m.vol['Reaction','Reactor',0]**2*m.tf['Reactor',0]*m.u['Reactor',0,t])    
m.energy_0 = dae.Integral(m.tau, wrt = m.tau, rule = _energy_0)
def _energy_1(m,t):
    return (m.vol['Reaction','Reactor',1]**2*m.tf['Reactor',1]*m.u['Reactor',1,t])    
m.energy_1 = dae.Integral(m.tau, wrt = m.tau, rule = _energy_1)
def _energy_2(m,t):
    return (m.vol['Reaction','Reactor',2]**2*m.tf['Reactor',2]*m.u['Reactor',2,t])    
m.energy_2 = dae.Integral(m.tau, wrt = m.tau, rule = _energy_2)
def _energy_3(m,t):
    return (m.vol['Reaction','Reactor',3]**2*m.tf['Reactor',3]*m.u['Reactor',3,t])    
m.energy_3 = dae.Integral(m.tau, wrt = m.tau, rule = _energy_3)
def _energy_4(m,t):
    return (m.vol['Reaction','Reactor',4]**2*m.tf['Reactor',4]*m.u['Reactor',4,t])    
m.energy_4 = dae.Integral(m.tau, wrt = m.tau, rule = _energy_4)
def _energy_5(m,t):
    return (m.vol['Reaction','Reactor',5]**2*m.tf['Reactor',5]*m.u['Reactor',5,t])    
m.energy_5 = dae.Integral(m.tau, wrt = m.tau, rule = _energy_5)
def _energy_6(m,t):
    return (m.vol['Reaction','Reactor',6]**2*m.tf['Reactor',6]*m.u['Reactor',6,t])    
m.energy_6 = dae.Integral(m.tau, wrt = m.tau, rule = _energy_5)

#Mass Balance constraints
def _dCAdt(m,j,n,t):
    return (m.CA_dot[j,n,t] == m.tf[j,n]*(-m.u[j,n,t]*m.CA[j,n,t]))
m.dCAdt =  pyo.Constraint(D_UNITS,m.EVENTS, m.tau, rule = _dCAdt)

def _dCBdt(m,j,n,t):
    return (m.CB_dot[j,n,t] == m.tf[j,n]*(m.u[j,n,t]*m.CA[j,n,t] - 0.065*m.CB[j,n,t]*m.u[j,n,t]**0.8))
m.dCBdt =  pyo.Constraint(D_UNITS, m.EVENTS, m.tau, rule = _dCBdt)

def _dCCdt(m,j,n,t):
    return (m.CC_dot[j,n,t] == m.tf[j,n]*(0.065*m.CB[j,n,t]*m.u[j,n,t]**0.8))
m.dCCdt =  pyo.Constraint(D_UNITS, m.EVENTS, m.tau, rule = _dCCdt)

def _difftime(m,j,n,t):
    return(m.dtime[j,n,t] == m.tf[j,n])
m.difftime = pyo.Constraint(D_UNITS, m.EVENTS, m.tau, rule = _difftime)

m.ini_final_con = pyo.ConstraintList()
for j in D_UNITS:
    for n in m.EVENTS:
        m.ini_final_con.add(m.CA[j,n,0] == 17)
        m.ini_final_con.add(m.CB[j,n,0] == 0)
        m.ini_final_con.add(m.CC[j,n,0] == 0)
        m.ini_final_con.add(m.CB[j,n,1] >= 13)
        m.ini_final_con.add(m.CC[j,n,1] >= 2.5)

#Connecting Variables (tf) to the upper level and making it 0 if the unit is not being used
m.tf_schedule = pyo.Var(m.UNITS, m.EVENTS, domain = pyo.NonNegativeReals, bounds=(0,10))
m.linking = pyo.ConstraintList()
for j in D_UNITS:
    for n in m.EVENTS:
        m.linking.add(m.tf_schedule[j,n] <= 10*m.yv[j,n])
        m.linking.add(m.tf_schedule[j,n] <= m.tf[j,n])
        m.linking.add(m.tf_schedule[j,n] >= m.tf[j,n] - 10*(1-m.yv[j,n]))
#Determinin Processing times from the volume
for n in m.EVENTS:
    m.linking.add(m.tf_schedule['Mixer',n] == 1*m.vol['Mixing','Mixer',n])
    m.linking.add(m.tf_schedule['Separator',n] == 1.2*m.vol['Separation','Separator',n])

#Initial amount
m.initial_amount = pyo.ConstraintList()
for s in m.STATES:
    m.initial_amount.add(m.STin[s] <= initial[s])
#Storage_constraint
m.storage_constraint = pyo.ConstraintList()
for n in m.EVENTS:
    m.storage_constraint.add(m.ST[s,n] <= storage[s])


#Demand Variable
m.d  = pyo.Var(m.STATES, m.EVENTS, domain = pyo.NonNegativeReals)
for s in m.STATES:
    for n in m.EVENTS:
        if (s!= 'S5' or n != EVENTS):
            m.d[s,n].fix(0)

#Allocation Constriant
m.allocation_constraint = pyo.ConstraintList()
for j in m.UNITS:
    for n in m.EVENTS:
        m.allocation_constraint.add(sum (m.wv[i,n] for i in K[j]) == m.yv[j,n])

#Capacity Constraints
m.capacity_constraint = pyo.ConstraintList()
for i,j in zip(m.TASKS, m.UNITS):
    for n in m.EVENTS:
        m.capacity_constraint.add(m.vol[i,j,n] <= Bmax[i,j]*m.wv[i,n])
        m.capacity_constraint.add(m.vol[i,j,n] >= Bmin[i,j]*m.wv[i,n])

    
#Material Balances
m.mat_balance= pyo.ConstraintList()

for s in m.STATES:
    for n in m.EVENTS:
        if n == 0:
            rhs = m.STin[s] - m.d[s,n] 
            for i in T[s]:
                    rhs -= rho[(i,s)]*(sum(m.vol[i,j,n] for j in L[i]))
            m.mat_balance.add(m.ST[s,n] == rhs)
        else:
            rhs = m.ST[s,n-1] - m.d[s,n]
            for i in T_[s]:
                    rhs += rho_[(i,s)]*(sum(m.vol[i,j,n-1] for j in L[i])) 
            for i in T[s]:
                    rhs -= rho[(i,s)]*(sum(m.vol[i,j,n] for j in L[i]))
            m.mat_balance.add(m.ST[s,n] == rhs)
            
#Duration Constraints       
m.duration_constraint = pyo.ConstraintList()
for i,j in zip(m.TASKS, m.UNITS):
    for n in m.EVENTS:
        m.duration_constraint.add(m.Tf[i,j,n] == m.Ts[i,j,n] + m.tf_schedule[j,n])

#Same Task in same unit
m.same_task_unit_constraint = pyo.ConstraintList()

for i,j in zip(m.TASKS, m.UNITS):
    for n in m.EVENTS:
        if n == 0:
            m.same_task_unit_constraint.Skip()
        else:
            m.same_task_unit_constraint.add(m.Ts[i,j,n] >= m.Tf[i,j,n-1] - H*(2 - m.wv[i,n-1] - m.yv[j,n-1]))
            m.same_task_unit_constraint.add(m.Ts[i,j,n] >= m.Ts[i,j,n-1])
            m.same_task_unit_constraint.add(m.Tf[i,j,n] >= m.Tf[i,j,n-1]) 
            
#Different Task in different Unit
m.different_unit_constraint = pyo.ConstraintList()

for (a,b) in SEQ:
    for n in m.EVENTS:
        if n == 0:
            m.different_unit_constraint.Skip()
        else:
            for i in K[b]:
                lhs = m.Ts[i,b,n]
            for i in K[a]:
                rhs = m.Tf[i,a,n-1]- H*(2 - m.wv[i,n-1] -m.yv[a,n-1])
            m.different_unit_constraint.add(lhs >= rhs)

#Complete all previous task in the unit before starting a new task 
m.completition_constraint = pyo.ConstraintList()
for i,j in zip(m.TASKS, m.UNITS):
    for n in m.EVENTS:
        if n == EVENTS:
            m.completition_constraint.Skip()
        else:    
            m.completition_constraint.add(m.Ts[i,j,n+1] >= (sum((m.Tf[i,j,E] - m.Ts[i,j,E]) for E  in m.EVENTS if E<=n)))

#Time horizon constraint
m.TH_constraint = pyo.ConstraintList()
for i,j in zip(m.TASKS, m.UNITS):
    for n in m.EVENTS:
        m.TH_constraint.add(m.Tf[i,j,n] <= H)
        m.TH_constraint.add(m.Ts[i,j,n] <= H)
        
#Objective
rhs = 0
for s in m.STATES:
    for n in m.EVENTS:
        rhs += sell_p[s]*m.d[s,n] 
    rhs -= buy_p[s]*m.STin[s] 
# rhs -= (m.Tf['Separation','Separator',EVENTS])
rhs -= (m.energy_0 + m.energy_1 + m.energy_2 
              + m.energy_3 + m.energy_4 + m.energy_5 + m.energy_6
            )     
m.objective = pyo.Objective( expr = rhs, sense = pyo.maximize) 

# Discretize and Solve model
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=10, ncp=4)
#Piecewise Control "
discretizer.reduce_collocation_points(m, var = m.u , ncp =1, contset =m.tau)

solver = pyo.SolverFactory('Bonmin')
solver.solve(m, tee = True).write()

#Results
def results(m):
    Schedule ={}
    for i in m.TASKS:
        for j in m.UNITS:
            for n in m.EVENTS:
                if m.vol[i,j,n]() != None:
                    Schedule[j,n] = dict(Event = n,Ts = m.Ts[i,j,n](),
                                        Tf = m.Tf[i,j,n](),vol =m.vol[i,j,n]())
    Schedule_df = pd.DataFrame(Schedule)
    
    UnitAssignment = pd.DataFrame({n:[None for i in m.TASKS] for n in m.EVENTS}, index=m.TASKS)
    for i in m.TASKS:
        for n in m.EVENTS:
            UnitAssignment.loc[i,n] = m.wv[i,n]()
    return (Schedule,Schedule_df, UnitAssignment)

print(results(m)[1]['Mixer'])
print(results(m)[1]['Reactor'])
print(results(m)[1]['Separator'])
print(results(m)[2])     
Schedule = results(m)[0]

#Results
Profit =round(m.objective(),1) 
Q_total = round(m.energy_0() + m.energy_1() + m.energy_2() + m.energy_3() + m.energy_4() #+ m.energy_5()
                ,1)
Produced = round(m.d['S5',EVENTS](),1)    
Raw_material = round(m.STin['S1'](),1)

summary = [Profit,Q_total,Produced, Raw_material]

def gantt_chart(Shedule,H,UNITS, summary):
    #Gantt Chart
    UNITS.reverse()
    fig, gnt = plt.subplots(figsize = (14,7.5))
    N_units = len(UNITS)
    y_limit = 40*N_units
    gnt.set_ylim(0, y_limit)
    gnt.set_xlim(0, H*1.05)
    
    #Chart layout
    gnt.set_axisbelow(True)
    gnt.grid(alpha = 0.9)
    label_size = 20
    # Setting labels for x-axis and y-axis
    gnt.set_xlabel('TIME [h]', fontsize = label_size+2)
         
    # Setting ticks on y-axis
    ticks_position = np.linspace(0,y_limit,N_units+2)[1:-1]
    gnt.set_yticks(ticks_position)
    gnt.set_yticklabels(UNITS, fontsize = label_size)
    gnt.tick_params(axis = 'x', labelsize = label_size)
    gnt.set_facecolor("whitesmoke")
    gnt.set_ylabel('UNIT', fontsize = label_size+2)
    
    #Summary
    gnt.text(9,94,'Profit:               '+str(summary[0]) + ' [$]'+
                    '\nUtility:              '+str(summary[1]) + ' [uU]'+
                    '\nProduction:      '+str(summary[2]) +' $\\rm[m^3]$' +
                    '\nConsumption:  '+str(summary[3]) +' $\\rm[m^3]$'
                     ,size =17.5,
             bbox=dict(boxstyle="square",
                   ec=(0, 0, 0),
                   fc=(1, 1, 1),
                   ),
             )
    bar_pos = {j: tuple() for j in UNITS}
    for (j,i) in zip(UNITS,ticks_position):
        bar_pos[j] = (i - 10,20)
        
    colors = ['lightcoral', 'cornflowerblue','lightgreen']
    c = {j: str() for j in UNITS}
    for (j,i) in zip(UNITS,colors):
        c[j] = i 
    
    for (j,n) in Schedule:
        if round(Schedule[j,n]['vol'],1) != 0.0:
            dura = round(Schedule[j,n]['Tf'] - Schedule[j,n]['Ts'],1)
            gnt.broken_barh([(Schedule[j,n]['Ts'], 
                              (Schedule[j,n]['Tf'] - Schedule[j,n]['Ts']))], 
                              (bar_pos[j]), facecolors= c[j], edgecolor = 'black',
                               linewidth = 1)
    # fig.savefig('CS1_DD.png.svg', dpi = 600,format = 'svg',bbox_inches  = 'tight')
    # # fig.savefig('Feasible_region.pdf', dpi = 600,format = 'pdf',bbox_inches  = 'tight')
    # fig.savefig('CS1_DD.png', dpi = 600,format = 'png',bbox_inches  = 'tight') 
        
    return (plt.show())   

gantt_chart(Schedule,H, UNITS,summary)
print(m.objective(),m.d['S5',EVENTS](),m.STin['S1'](),m.ST['S5',EVENTS](),m.nconstraints(),m.nvariables())
