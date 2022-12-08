import json
import numpy as np
import matplotlib.pyplot as plt

from hierarchy.planning.Planning_Extended_dyn import simulate, state_to_control_t
from data.planning.planning_sch_bilevel_lowdim import scheduling_data, data

def average_from_list(array_total):
    f_median = np.median(array_total, axis = 0)
    f_min = np.min(array_total, axis = 0)
    f_max = np.max(array_total, axis = 0)
    return f_median, f_min, f_max

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

def plots(input_data, Production, TP, Sales, planning=True, scheduling=False, control=False):
    
    T_set = input_data[None]["T"][None]
    N_t = len(T_set)
    N_Tc = input_data[None]["N_t"][None]

    to_Tc = lambda t: state_to_control_t(t, N_Tc, T_set)

    Forecast = get_Forecast(input_data)

    N_st = 10

    Storage_big = {} ; Demand_big = {} ; Sales_big = {}
    Storage, Demand, dummy_Sales = simulate(Production, TP, Forecast, Sales, input_data, seed=0)
    for element in Storage:
        Storage_big[element] = np.zeros((N_st, N_t))
    for element in Demand:
        Demand_big[element] = np.zeros((N_st, N_t))
    for element in dummy_Sales:
        Sales_big[element] = np.zeros((N_st, N_t))
        
    for stoch in range(N_st):
        Storage, Demand, dummy_Sales = simulate(Production, TP, Forecast, Sales, input_data, seed=stoch)

        for element in Storage:
            Storage_big[element][stoch] = Storage[element]
        for element in Demand:
            Demand_big[element][stoch] = Demand[element]
        for element in dummy_Sales:
            Sales_big[element][stoch] = dummy_Sales[element]
    
    Storage_median = {} ; Demand_median = {} ; Sales_median = {}
    Storage_min = {} ; Demand_min = {} ; Sales_min = {}
    Storage_max = {} ; Demand_max = {} ; Sales_max = {}
    for element in Storage:
        median, min, max = average_from_list(Storage_big[element])
        Storage_median[element] = median
        Storage_min[element] = min
        Storage_max[element] = max
    for element in Demand:
        median, min, max = average_from_list(Demand_big[element])
        Demand_median[element] = median
        Demand_min[element] = min
        Demand_max[element] = max
    for element in Demand:
        median, min, max = average_from_list(Sales_big[element])
        Sales_median[element] = median
        Sales_min[element] = min
        Sales_max[element] = max

    P_PA     = Production['PA'] 
    P_PC    = Production['PC'] 
    P_TEE   = Production['TEE']
    P_PB    = Production['PB'] 
    P_PD    = Production['PD'] 
    P_TGE   = Production['TGE']
    PAI     = Production['AI'] 
    PI  = Production['I']    
    TP_As   = TP['Asia']       
    TP_Am   = TP['America']    

    fig, axs = plt.subplots(4, 2, figsize=(60, 30))

    axs[0,0].step(np.arange(N_t)+1, PI, label='P_I', where = 'post')
    axs[0,0].step(np.arange(N_t)+1, PAI, label='P_AI', where = 'post')
    axs[0,0].step(np.arange(N_Tc)+1, TP_Am, label='F_Am', where = 'post')
    axs[0,0].step(np.arange(N_Tc)+1, TP_As, label='F_As', where = 'post')
    # axs[0,0].legend()
    axs[0,0].set_ylabel('AI production in Europe')
    axs[0,0].set_xlabel('Time in weeks')

    axs[1,0].step(np.arange(N_t)+1, P_PB, c = 'blue', label='P_PB', where = 'post')
    axs[1,0].step(np.arange(N_t)+1, P_PD, c='orange', label='P_PD', where = 'post')
    axs[1,0].step(
        np.arange(N_t)+1, Sales_median['PB'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Sales_min['PB'], Sales_max['PB'], 
        label='S_PB', alpha = .5, step = 'post', color='cyan',
        )
    axs[1,0].step(
        np.arange(N_t)+1, Sales_median['PD'], 
        where = 'post', c='green', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Sales_min['PD'], Sales_max['PD'], 
        label='S_PD', alpha = .5, step = 'post', color='green',
        )
    axs[1,0].step(
        np.arange(N_t)+1, Demand_median['PB'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Demand_min['PB'], Demand_max['PB'], 
        label='D_PB', alpha = .5, step = 'post', color='darkblue'
        )
    axs[1,0].step(
        np.arange(N_t)+1, Demand_median['PD'], 
        where = 'post', c='darkorange', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Demand_min['PD'], Demand_max['PD'], 
        label='D_PD', alpha = .5, step = 'post', color='darkorange'
        )
    # axs[0,1].legend()
    axs[1,0].set_ylabel('PB and PD production')
    axs[1,0].set_xlabel('Time in weeks')

    # IIstar = input_data[None]['IIstar0'][None]
    # IAIPstar = input_data[None]['IAIPstar0'][None]
    # axs[1,0].step(
    #     np.arange(N_t)+1, Storage_median['I'], 
    #     where = 'post', c='darkblue', linestyle='--'
    #     ) 
    # axs[1,0].fill_between(
    #     np.arange(N_t)+1, Storage_min['I'], Storage_max['I'], 
    #     label='I storage', alpha = .5, step = 'post'
    #     )
    # axs[1,0].plot([1, N_t], [IIstar, IIstar], c = 'darkblue', linestyle='--', label = 'Safe S_I')
    # axs[1,0].step(
    #     np.arange(N_t)+1, Storage_median['AIP'], 
    #     where = 'post', c='darkorange', linestyle='--'
    #     ) 
    # axs[1,0].fill_between(
    #     np.arange(N_t)+1, Storage_min['AIP'], Storage_max['AIP'], 
    #     label='AI storage', alpha = .5, step = 'post'
    #     )
    # axs[1,0].plot([1, N_t], [IAIPstar, IAIPstar], c = 'darkorange', linestyle='--', label = 'Safe S_AI_Eu')
    # # axs[1,0].legend()
    # axs[1,0].set_ylabel('Storage in Europe')
    # axs[1,0].set_xlabel('Time in weeks')

    # IAISstar_As = input_data[None]['IAISstar0']['Asia']
    # IAISstar_Am = input_data[None]['IAISstar0']['America']
    # axs[2,0].step(
    #     np.arange(N_t)+1, Storage_median['AIS_As'], 
    #     where = 'post', c='darkblue', linestyle='--'
    #     ) 
    # axs[2,0].fill_between(
    #     np.arange(N_t)+1, Storage_min['AIS_As'], Storage_max['AIS_As'], 
    #     label='S_AI_As', alpha = .5, step = 'post'
    #     )
    # axs[2,0].plot([1, N_t], [IAISstar_As, IAISstar_As], c='darkblue', linestyle='--', label = 'Safe S_AI_As')
    # axs[2,0].step(
    #     np.arange(N_t)+1, Storage_median['AIS_Am'], 
    #     where = 'post', c='darkorange', linestyle='--'
    #     ) 
    # axs[2,0].fill_between(
    #     np.arange(N_t)+1, Storage_min['AIS_Am'], Storage_max['AIS_Am'], 
    #     label='S_AI_Am', alpha = .5, step = 'post'
    #     )
    # axs[2,0].plot([1, N_t], [IAISstar_Am, IAISstar_Am], c='darkorange', linestyle='--', label = 'Safe S_AI_Am')
    # # axs[2,0].legend()
    # axs[2,0].set_ylabel('AI storage')
    # axs[2,0].set_xlabel('Time in weeks')

    axs[2,0].step(
        np.arange(N_t)+1, Demand_median['TGE'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[2,0].fill_between(
        np.arange(N_t)+1, Demand_min['TGE'], Demand_max['TGE'], 
        label='D_TEE', alpha = .5, step = 'post', color = 'blue',
        )
    axs[2,0].step(np.arange(N_t)+1, P_TGE, label='P_TGE', where = 'post')
    axs[2,0].step(
        np.arange(N_t)+1, Sales_median['TGE'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[2,0].fill_between(
        np.arange(N_t)+1, Sales_min['TGE'], Sales_max['TGE'], 
        label='SA_TGE', alpha = .5, step = 'post', color='cyan',
        )
    axs[2,0].set_ylabel('TGE production')
    axs[2,0].set_xlabel('Time in weeks')

    axs[0,1].step(np.arange(N_t)+1, P_PA, c = 'blue', label='P_PA', where = 'post')
    axs[0,1].step(np.arange(N_t)+1, P_PC, c='orange', label='P_PC', where = 'post')
    axs[0,1].step(
        np.arange(N_t)+1, Sales_median['PA'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Sales_min['PA'], Sales_max['PA'], 
        label='S_PA', alpha = .5, step = 'post', color='cyan',
        )
    axs[0,1].step(
        np.arange(N_t)+1, Sales_median['PC'], 
        where = 'post', c='green', linestyle='--'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Sales_min['PC'], Sales_max['PC'], 
        label='S_PC', alpha = .5, step = 'post', color='green',
        )
    axs[0,1].step(
        np.arange(N_t)+1, Demand_median['PA'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Demand_min['PA'], Demand_max['PA'], 
        label='D_PA', alpha = .5, step = 'post', color='darkblue'
        )
    axs[0,1].step(
        np.arange(N_t)+1, Demand_median['PC'], 
        where = 'post', c='darkorange', linestyle='--'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Demand_min['PC'], Demand_max['PC'], 
        label='D_PC', alpha = .5, step = 'post', color='darkorange'
        )
    # axs[0,1].legend()
    axs[0,1].set_ylabel('PA and PC production')
    axs[0,1].set_xlabel('Time in weeks')

    axs[3,0].step(
        np.arange(N_t)+1, Demand_median['TEE'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[3,0].fill_between(
        np.arange(N_t)+1, Demand_min['TEE'], Demand_max['TEE'], 
        label='D_TEE', alpha = .5, step = 'post', color = 'blue',
        )
    axs[3,0].step(np.arange(N_t)+1, P_TEE, label='P_TEE', where = 'post')
    axs[3,0].step(
        np.arange(N_t)+1, Sales_median['TEE'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[3,0].fill_between(
        np.arange(N_t)+1, Sales_min['TEE'], Sales_max['TEE'], 
        label='SA_TEE', alpha = .5, step = 'post', color='cyan',
        )
    axs[3,0].set_ylabel('TEE production')
    axs[3,0].set_xlabel('Time in weeks')
    # axs[3,0].legend()

    # IPAstar = input_data[None]['Istar0']['PA']
    # ITEEstar = input_data[None]['Istar0']['TEE']
    # axs[1,1].step(
    #     np.arange(N_t)+1, Storage_median['TEE'], 
    #     where = 'post', c='darkblue', linestyle='--'
    #     ) 
    # axs[1,1].fill_between(
    #     np.arange(N_t)+1, Storage_min['TEE'], Storage_max['TEE'], 
    #     label='S_TEE', alpha = .5, step = 'post'
    #     )
    # axs[1,1].plot([1, N_t], [ITEEstar, ITEEstar], label = 'Safe S_TEE')
    # # axs[1,1].legend()
    # axs[1,1].set_ylabel('Storage of TEE')
    # axs[1,1].set_xlabel('Time in weeks')

    # IPCstar = input_data[None]['Istar0']['PC']
    # IPAstar = input_data[None]['Istar0']['PA']
    # axs[2,1].step(
    #     np.arange(N_t)+1, Storage_median['PC'], 
    #     where = 'post', c='darkblue', linestyle='--'
    #     ) 
    # axs[2,1].fill_between(
    #     np.arange(N_t)+1, Storage_min['PC'], Storage_max['PC'], 
    #     label='S_PC', alpha = .5, step = 'post'
    #     )
    # axs[2,1].plot([1, N_t], [IPCstar, IPCstar], c='darkblue', label = 'Safe S_PC')
    # axs[2,1].step(
    #     np.arange(N_t)+1, Storage_median['PA'], 
    #     where = 'post', c='darkorange', linestyle='--'
    #     ) 
    # axs[2,1].fill_between(
    #     np.arange(N_t)+1, Storage_min['PA'], Storage_max['PA'], 
    #     label='S_PA', alpha = .5, step = 'post'
    #     )
    # axs[2,1].plot([1, N_t], [IPAstar, IPAstar], c='darkorange', label = 'Safe S_PA')
    # # axs[2,1].legend()
    # axs[2,1].set_ylabel('PA and PC storage')
    # axs[2,1].set_xlabel('Time in weeks')

    plt.show()

    return fig, axs

def get_plots(x, input_data, planning=True, scheduling=False, control=False):
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

    return plots(input_data, Production, TP, Sales, planning=planning, scheduling=scheduling, control=control)

Nt = 5

data_copy = data.copy()
data_copy[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})

with open('./results/optima/tri_Py-BOBYQA.json') as f:
    bi_opt = json.load(f)

# fig, axs = get_plots(bi_opt['x'], data_copy)
# fig.show()

with open('./results/optima/centralized.json') as f:
    opt = json.load(f)

# fig, axs = get_plots(opt['x'], data_copy)
# fig.show()

idx = np.where(np.abs(np.array(bi_opt['x']) - np.array(opt['x'])) > 1)
print(idx)
