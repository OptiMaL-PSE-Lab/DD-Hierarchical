
import json
import time
import numpy as np
import multiprocessing as mp
import pyomo.environ as pyo

from pyomo.opt import TerminationCondition

from functools import partial

from sampling import batch_reactor
from data.planning.planning_sch_bilevel_lowdim import scheduling_data, data
from hierarchy.planning.Planning_Extended_dyn import simulate, state_to_control_t
from hierarchy.planning.Planning_Scheduling_bi import scheduling_Asia # , centralised

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



def parallel_(P_set, X, data_, Production, t):

    P_As = []
    for p in P_set:
        if X['Asia', p]:
            P_As.append(p) 

    for p in P_As:
        if p in data_[None]['states'][None]:
            data_[None]['Prod'][p] = Production[p][t-1]
    ### 
    # print('Average input to scheduling problem: ', scheduling_data[None]['Prod'])
    
    try:
        res_Sch, res = scheduling_Asia(data_, res_out=True)
        changeover = pyo.value(res_Sch.CCH_cost)
        print(res.solver.termination_condition, res.solver.status)
        # print('Before: ', res_Sch.MS.value)
        storage = pyo.value(res_Sch.st_cost)/20
        energy = 0
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
                        print('Control problem at:', m, n, t)
                        energy += 10/12*1e6
                    # except Exception as e:
                    #     print('Control problem at:', m, n, t, e)
                    #     energy += 10/12*1e6

        
#         if res_Sch.Ws[m,n].value==1:
        #             res_Sch.beta[m].fix(data['tproc'])
        #         else:
        #             res_Sch.beta[m].fix(0)

        # res_Sch.T_CCH.fix()
        # res_Sch.Ws.fix()
        # res_Sch.Wf.fix()
        # res_Sch.Bs.fix()
        # res_Sch.Bp.fix()
        # res_Sch.Bf.fix()
        # res_Sch.B_cons.fix()
        # res_Sch.B_prod.fix()
        # res_Sch.S.fix()
        # res_Sch.Y.fix()


        # solver = pyo.SolverFactory("gurobi_direct")
        # res_Sch = solver.solve(res_Sch)

        # print('After: ', res_Sch.MS.value)

        ### Next: solve scheduling again
        

    except Exception as e:
        print('Scheduling problem at: ', t, ' due to: ')
        print(e)
        changeover = 100./12*1e6
        storage = 10/12*1e6
        energy = 10/12*1e6
    # storage = pyo.value(res_Sch.CCH_cost)*24

    return changeover + storage + energy


def get_cost_(input_data, Production, TP, Sales, penalization=1):
    
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



    wrap_tri_parallel = partial(parallel_, P_set, X, scheduling_data.copy(), Production)

    pool = mp.Pool(len(T_set))
    # res = pool.map(wrap_tri_parallel, [float(t) for t in T_set])
    res = pool.map(wrap_tri_parallel, T_set)
    change = np.sum(res)

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

    return (prod_cost + transp_cost + store_cost + change - sales)/1e6 + penalty*penalization

def wrapper_(x, input_data, penalty=1):
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

    return get_cost_(input_data, Production, TP, Sales, penalization=penalty)

if __name__=='__main__':

    Nt = 5
    data_copy = data.copy()
    data_copy[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})


    methods = [
        # 'surr_distr/hierarch_RegNN_11_1',
        # 'surr_distr/hierarch_RegNN_11_2',
        # 'surr_distr/hierarch_RegNN_11_3',
        # 'surr_distr/hierarch_RegNN_12_2',
        # 'surr_distr/hierarch_RegNN_12_3',
        # 'surr_distr/hierarch_RegNN_16_1',
        # 'surr_distr/hierarch_RegNN_22_1',
        # 'surr_distr/hierarch_RegNN_24_1',
        # 'surr_distr/hierarch_RegNN_26_1',
        # 'surr_distr/hierarch_RegNN_42_20',
        # 'surr_distr/hierarch_RegNN_63_3',
        # 'surr_distr/hierarch_RegNN_79_40',
        # 'surr_distr/integrated_RegNN_5_1',
        # 'surr_distr/integrated_RegNN_5_2',
        # 'surr_distr/integrated_RegNN_5_23',
        #'surr_distr/integrated_RegNN_5_30', # didn't work
        # 'surr_distr/integrated_RegNN_5_31',
        # 'surr_distr/integrated_RegNN_6_16',
        # 'surr_distr/integrated_RegNN_11_1',
        # 'surr_distr/integrated_RegNN_41_12',
        # 'surr_distr/integrated_RegNN_41_31',
        # 'surr_distr/integrated_RegNN_45_2',
        # 'surr_distr/integrated_RegNN_51_8',
        # 'surr_distr/integrated_RegNN_60_20',
        # 'surr_distr/scheduling_RegNN_9_1',
        # 'surr_distr/scheduling_RegNN_10_1',
        # 'surr_distr/scheduling_RegNN_11_1',
        # 'surr_distr/scheduling_RegNN_17_17',
        # 'surr_distr/scheduling_RegNN_21_19',
        # 'surr_distr/scheduling_RegNN_31_27',
        # 'surr_distr/scheduling_RegNN_41_20',
        # 'surr_distr/scheduling_RegNN_42_20',
        # 'surr_distr/scheduling_RegNN_50_20',
        # 'surr_distr/scheduling_RegNN_56_19',
        # 'surr_distr/scheduling_RegNN_78_40',
        # 'surr_distr/scheduling_RegNN_79_40',
        'surr_distr/integrated_RegNN_21_1',
	'surr_distr/scheduling_RegNN_5_20',
	'surr_distr/integrated_RegNN_42_1',
	'surr_distr/integrated_RegNN_79_40',
        'surr_distr/integrated_RegNN_37_17',
        # 'centralized',
	    # 'integrated_RegNN_43_20',
        # 'integrated_RegNN_45_2',
        # 'integrated_RegNN_5_31',
        # 'integrated_RegNN_5_2',
        # 'scheduling_RegNN_78_40',
        # 'scheduling_RegNN_42_20',
        # 'scheduling_RegNN_10_1',
	# 'bi_Py-BOBYQA',
	# 'bi_Py-BOBYQA',
	# 'bi_Py-BOBYQA_low',
	# 'bi_Py-BOBYQA_distr',
	# 'tri_Py-BOBYQA',
	# 'tri_Py-BOBYQA_low',
	# 'tri_Py-BOBYQA_distr',
    ]
   
    for m in methods:
        try:
            dir = f"./results/Optima/{m}.json"
        
            with open(dir, 'r+') as file:
                opt = json.load(file)
                x = opt['x']
                t0 = time.perf_counter()
                real = wrapper_(x, data_copy, 1000)
                t = time.perf_counter()
                print(f"Real hierarchical objective: {real:.3f} in {t-t0:.3f} seconds")
                opt['real'] = {'obj': real, 'time': t - t0}
        
            with open(dir, 'w') as f:
                json.dump(opt, f)
        except:
            print('Error with ', m)






