import json
import numpy as np
import matplotlib.pyplot as plt

from hierarchy.planning.Planning_Extended_dyn import simulate, state_to_control_t
from data.planning.planning_sch_bilevel_lowdim import scheduling_data, data

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

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

    P_PA    = Production['PA'] 
    P_PC    = Production['PC'] 
    P_TEE   = Production['TEE']
    P_PB    = Production['PB'] 
    P_PD    = Production['PD'] 
    P_TGE   = Production['TGE']
    PAI     = Production['AI'] 
    PI      = Production['I']    
    TP_As   = TP['Asia']       
    TP_Am   = TP['America']    

    fig, axs = plt.subplots(5, 2, figsize=(20, 12))

    axs[0,0].step(np.arange(N_t)+1, PI, label='I Production', where = 'post', c='darkblue')
    axs[0,0].step(np.arange(N_t)+1, PAI, label='AI Production', where = 'post', c='darkorange')
    axs[0,0].step(np.arange(N_Tc)+1, TP_Am, label='America transport', where = 'post', c='green')
    axs[0,0].step(np.arange(N_Tc)+1, TP_As, label='Asia transport', where = 'post', c='brown')
    axs[0,0].set_ylabel('AI production in Europe')
    axs[0,0].set_xlabel('Time in weeks')
    axs[0,0].legend(ncol=2)

    axs[2,0].step(np.arange(N_t)+1, P_PC, c = 'blue', label='PC production', where = 'post')
    axs[2,0].step(np.arange(N_t)+1, P_PD, c='orange', label='PD production', where = 'post')
    axs[2,0].step(
        np.arange(N_t)+1, Sales_median['PC'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[2,0].fill_between(
        np.arange(N_t)+1, Sales_min['PC'], Sales_max['PC'], 
        label='PC sales', alpha = .5, step = 'post', color='cyan',
        )
    axs[2,0].step(
        np.arange(N_t)+1, Sales_median['PD'], 
        where = 'post', c='orangered', linestyle='--'
        ) 
    axs[2,0].fill_between(
        np.arange(N_t)+1, Sales_min['PD'], Sales_max['PD'], 
        label='PD sales', alpha = .5, step = 'post', color='orangered',
        )
    axs[2,0].step(
        np.arange(N_t)+1, Demand_median['PC'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[2,0].fill_between(
        np.arange(N_t)+1, Demand_min['PC'], Demand_max['PC'], 
        label='PC demand', alpha = .5, step = 'post', color='darkblue'
        )
    axs[2,0].step(
        np.arange(N_t)+1, Demand_median['PD'], 
        where = 'post', c='darkorange', linestyle='--'
        ) 
    axs[2,0].fill_between(
        np.arange(N_t)+1, Demand_min['PD'], Demand_max['PD'], 
        label='PD demand', alpha = .5, step = 'post', color='darkorange'
        )
    axs[2,0].set_ylabel('PC and PD production')
    axs[2,0].set_xlabel('Time in weeks')
    axs[2,0].legend(ncol=3)

    IIstar = input_data[None]['IIstar0'][None]
    IAIPstar = input_data[None]['IAIPstar0'][None]
    axs[0,1].step(
        np.arange(N_t)+1, Storage_median['I'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Storage_min['I'], Storage_max['I'], 
        label='I storage', alpha = .5, step = 'post', color='blue',
        )
    axs[0,1].plot([1, N_t], [IIstar, IIstar], c = 'darkblue', linestyle='--', label = 'Safe I storage')
    axs[0,1].step(
        np.arange(N_t)+1, Storage_median['AIP'], 
        where = 'post', c='darkorange', linestyle='--'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Storage_min['AIP'], Storage_max['AIP'], 
        label='AI storage', alpha = .5, step = 'post', color='orange',
        )
    axs[0,1].plot([1, N_t], [IAIPstar, IAIPstar], c = 'darkorange', linestyle='--', label = 'Safe AI storage')

    IAISstar_As = input_data[None]['IAISstar0']['Asia']
    IAISstar_Am = input_data[None]['IAISstar0']['America']
    axs[0,1].step(
        np.arange(N_t)+1, Storage_median['AIS_As'], 
        where = 'post', c='darkgreen', linestyle='-'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Storage_min['AIS_As'], Storage_max['AIS_As'], 
        label='Asia storage', alpha = .5, step = 'post', color='green',
        )
    axs[0,1].plot([1, N_t], [IAISstar_As, IAISstar_As], c='darkgreen', linestyle='--', label = 'Asia safe storage')
    axs[0,1].step(
        np.arange(N_t)+1, Storage_median['AIS_Am'], 
        where = 'post', c='brown', linestyle='-'
        ) 
    axs[0,1].fill_between(
        np.arange(N_t)+1, Storage_min['AIS_Am'], Storage_max['AIS_Am'], 
        label='America storage', alpha = .5, step = 'post',color='brown'
        )
    axs[0,1].plot([1, N_t], [IAISstar_Am, IAISstar_Am], c='brown', linestyle='--', label = 'America safe storage')
    # axs[2,0].legend()
    axs[0,1].set_ylabel('Secondary AI storage')
    axs[0,1].set_xlabel('Time in weeks')
    axs[0,1].legend(ncol=4)

    axs[3,0].step(
        np.arange(N_t)+1, Demand_median['TGE'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[3,0].fill_between(
        np.arange(N_t)+1, Demand_min['TGE'], Demand_max['TGE'], 
        label='Demand', alpha = .5, step = 'post', color = 'blue',
        )
    axs[3,0].step(np.arange(N_t)+1, P_TGE, label='Production', where = 'post', color='black')
    axs[3,0].step(
        np.arange(N_t)+1, Sales_median['TGE'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[3,0].fill_between(
        np.arange(N_t)+1, Sales_min['TGE'], Sales_max['TGE'], 
        label='Sales', alpha = .5, step = 'post', color='cyan',
        )
    axs[3,0].set_ylabel('TGE production')
    axs[3,0].set_xlabel('Time in weeks')
    axs[3,0].legend(loc='lower center', ncol=3)

    axs[1,0].step(np.arange(N_t)+1, P_PA, c = 'blue', label='PA production', where = 'post')
    axs[1,0].step(np.arange(N_t)+1, P_PB, c='orange', label='PB production', where = 'post')
    axs[1,0].step(
        np.arange(N_t)+1, Sales_median['PA'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Sales_min['PA'], Sales_max['PA'], 
        label='PA sales', alpha = .5, step = 'post', color='cyan',
        )
    axs[1,0].step(
        np.arange(N_t)+1, Sales_median['PB'], 
        where = 'post', c='orangered', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Sales_min['PB'], Sales_max['PB'], 
        label='PB sales', alpha = .5, step = 'post', color='orangered',
        )
    axs[1,0].step(
        np.arange(N_t)+1, Demand_median['PA'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Demand_min['PA'], Demand_max['PA'], 
        label='PA demand', alpha = .5, step = 'post', color='darkblue'
        )
    axs[1,0].step(
        np.arange(N_t)+1, Demand_median['PB'], 
        where = 'post', c='darkorange', linestyle='--'
        ) 
    axs[1,0].fill_between(
        np.arange(N_t)+1, Demand_min['PB'], Demand_max['PB'], 
        label='PB demand', alpha = .5, step = 'post', color='darkorange'
        )
    axs[1,0].set_ylabel('PA and PB production')
    axs[1,0].set_xlabel('Time in weeks')
    axs[1,0].legend(ncol=3)

    axs[4,0].step(
        np.arange(N_t)+1, Demand_median['TEE'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[4,0].fill_between(
        np.arange(N_t)+1, Demand_min['TEE'], Demand_max['TEE'], 
        label='Demand', alpha = .5, step = 'post', color = 'blue',
        )
    axs[4,0].step(np.arange(N_t)+1, P_TEE, label='Production', where = 'post', color='black')
    axs[4,0].step(
        np.arange(N_t)+1, Sales_median['TEE'], 
        where = 'post', c='cyan', linestyle='--'
        ) 
    axs[4,0].fill_between(
        np.arange(N_t)+1, Sales_min['TEE'], Sales_max['TEE'], 
        label='Sales', alpha = .5, step = 'post', color='cyan',
        )
    axs[4,0].set_ylabel('TEE production')
    axs[4,0].set_xlabel('Time in weeks')
    axs[4,0].legend(loc='lower center', ncol=3)

    IPAstar = input_data[None]['Istar0']['PA']
    ITEEstar = input_data[None]['Istar0']['TEE']
    axs[4,1].step(
        np.arange(N_t)+1, Storage_median['TEE'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[4,1].fill_between(
        np.arange(N_t)+1, Storage_min['TEE'], Storage_max['TEE'], 
        label='Storage', alpha = .5, step = 'post'
        )
    axs[4,1].plot([1, N_t], [ITEEstar, ITEEstar], label = 'Safe Storage')
    axs[4,1].legend()
    axs[4,1].set_ylabel('Storage of TEE')
    axs[4,1].set_xlabel('Time in weeks')

    ITGEstar = input_data[None]['Istar0']['TGE']
    axs[3,1].step(
        np.arange(N_t)+1, Storage_median['TGE'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[3,1].fill_between(
        np.arange(N_t)+1, Storage_min['TGE'], Storage_max['TGE'], 
        label='Storage', alpha = .5, step = 'post'
        )
    axs[3,1].plot([1, N_t], [ITGEstar, ITGEstar], label = 'Safe Storage')
    axs[3,1].legend()
    axs[3,1].set_ylabel('Storage of TGE')
    axs[3,1].set_xlabel('Time in weeks')


    IPBstar = input_data[None]['Istar0']['PB']
    IPAstar = input_data[None]['Istar0']['PA']
    axs[1,1].step(
        np.arange(N_t)+1, Storage_median['PA'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[1,1].fill_between(
        np.arange(N_t)+1, Storage_min['PA'], Storage_max['PA'], 
        label='PA storage', alpha = .5, step = 'post'
        )
    axs[1,1].plot([1, N_t], [IPAstar, IPAstar], c='darkblue', label = 'PA safe storage')
    axs[1,1].step(
        np.arange(N_t)+1, Storage_median['PB'], 
        where = 'post', c='darkorange', linestyle='--'
        ) 
    axs[1,1].fill_between(
        np.arange(N_t)+1, Storage_min['PB'], Storage_max['PB'], 
        label='PB storage', alpha = .5, step = 'post'
        )
    axs[1,1].plot([1, N_t], [IPAstar, IPAstar], c='darkorange', label = 'PB safe storage')
    # axs[2,1].legend()
    axs[1,1].set_ylabel('PA and PB storage')
    axs[1,1].set_xlabel('Time in weeks')
    axs[1,1].legend(ncol=4)

    IPCstar = input_data[None]['Istar0']['PC']
    IPDstar = input_data[None]['Istar0']['PD']
    axs[2,1].step(
        np.arange(N_t)+1, Storage_median['PC'], 
        where = 'post', c='darkblue', linestyle='--'
        ) 
    axs[2,1].fill_between(
        np.arange(N_t)+1, Storage_min['PC'], Storage_max['PC'], 
        label='PC storage', alpha = .5, step = 'post'
        )
    axs[2,1].plot([1, N_t], [IPCstar, IPCstar], c='darkblue', label = 'PC safe storage')
    axs[2,1].step(
        np.arange(N_t)+1, Storage_median['PD'], 
        where = 'post', c='darkorange', linestyle='--'
        ) 
    axs[2,1].fill_between(
        np.arange(N_t)+1, Storage_min['PD'], Storage_max['PD'], 
        label='PD storage', alpha = .5, step = 'post'
        )
    axs[2,1].plot([1, N_t], [IPDstar, IPDstar], c='darkorange', label = 'PD safe storage')
    axs[2,1].legend(ncol=4)
    axs[2,1].set_ylabel('PC and PD storage')
    axs[2,1].set_xlabel('Time in weeks')

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

def plot_traj(method, Nt, type='DFO',SAVE=False):

    data_copy = data.copy()
    data_copy[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})

    with open('./results/Optima/'+method+'.json') as f:
        opt = json.load(f)

    fig, axs = get_plots(opt['x'], data_copy)

    if 'real' in opt:
        real_obj = opt['real']['obj']
    else:
        real_obj = np.nan
    if type=='DFO':
        axs[0,0].set_title(f"Objectives: upper: {opt['upper']['obj']:.3f}, bi: {opt['bi']['obj']:.3f}, tri: {opt['tri']['obj']:.3f}, real: {real_obj:.3f}")
        axs[0,1].set_title(f"Derivative-free optimisation time: {opt['time']:.3f} min")
    elif type=='surrogate':
        if 'large' in opt:
            dummy = opt['large']
            time = opt['opt_time_l']
        elif 'medium' in opt:
            dummy = opt['medium']
            time = opt['opt_time_m']
        elif 'small' in opt:
            dummy = opt['small']
            time = opt['opt_time_s']
        else:
            dummy = opt['hierarchy']
            time = opt['opt_time_s']
        axs[0,1].set_title(f"Surrogate optimisation time: {time/60:.3f} min")
        axs[0,0].set_title(f"Objectives: upper: {dummy['upper']['obj']:.3f}, bi: {dummy['bi']['obj']:.3f}, tri: {dummy['tri']['obj']:.3f}, real: {real_obj:.3f}")
    else:
        raise ValueError("type should be either 'DFO' or 'surrogate'")

    fig.savefig('./results/Figures/'+method+'.svg')


