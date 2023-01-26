# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:51:19 2021

@author: dv516
"""

### TO DO: BO + split into problems and benchmarks ###

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import partial

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import onnx
from omlt import OmltBlock, OffsetScaling
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
from omlt.io import load_onnx_neural_network

from omlt.neuralnet import ReluBigMFormulation

from data.planning.planning_sch_bilevel import data as import_data
from data.planning.planning_sch_bilevel import scheduling_data

def state_to_control_t(t, N_t, T_set):
    dummy_array = np.arange(1, 1+N_t)
    N_total = len(T_set)
    T_min = T_set[0]
    idx = 1 + (t - T_min) // int(N_total/N_t)
    return min(idx, N_t)

try:
    dir = './data/CS2_Sampling/Batch_Reactor_NN_PA'
    df = pd.read_csv(dir) 
except:
    dir = '../data/CS2_Sampling/Batch_Reactor_NN_PA'
    df = pd.read_csv(dir) 
inputs = ['conc_end']
outputs = ['tf','utility']

dfin = df[inputs]
dfout = df[outputs]

#Scaling
x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()

dfin = (dfin - dfin.mean()).divide(dfin.std())
dfout = (dfout - dfout.mean()).divide(dfout.std())

#Save the scaling parameters of the inputs for OMLT
scaled_lb = dfin.min()[inputs].values
scaled_ub = dfin.max()[inputs].values
scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}
# scaling factors
scaler = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )

try:
    dir = './results/Models/CS2_ReLU_PA.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/CS2_ReLU_PA.onnx'
    onnx_model = onnx.load(dir)
net = load_onnx_neural_network(onnx_model, scaler, scaled_input_bounds)


try:
    dir = './data/CS2_Sampling/Batch_Reactor_NN_PB'
    df = pd.read_csv(dir) 
except:
    dir = '../data/CS2_Sampling/Batch_Reactor_NN_PB'
    df = pd.read_csv(dir)

dfin_PB = df[inputs]
dfout_PB = df[outputs]

#Scaling
x_offset, x_factor = dfin_PB.mean().to_dict(), dfin_PB.std().to_dict()
y_offset, y_factor = dfout_PB.mean().to_dict(), dfout_PB.std().to_dict()

dfin_PB = (dfin_PB - dfin_PB.mean()).divide(dfin_PB.std())
dfout_PB = (dfout_PB - dfout_PB.mean()).divide(dfout_PB.std())

#Save the scaling parameters of the inputs for OMLT
scaled_lb = dfin_PB.min()[inputs].values
scaled_ub = dfin_PB.max()[inputs].values
scaled_input_bounds_PB = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}
# scaling factors
scaler_PB = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )

try:
    dir = './results/Models/CS2_ReLU_PB.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/CS2_ReLU_PB.onnx'
    onnx_model = onnx.load(dir)
net_PB = load_onnx_neural_network(onnx_model, scaler_PB, scaled_input_bounds_PB)


try:
    dir = './data/CS2_Sampling/Batch_Reactor_NN_TEE'
    df = pd.read_csv(dir) 
except:
    dir = '../data/CS2_Sampling/Batch_Reactor_NN_TEE'
    df = pd.read_csv(dir)

dfin_TEE = df[inputs]
dfout_TEE = df[outputs]

#Scaling
x_offset, x_factor = dfin_TEE.mean().to_dict(), dfin_TEE.std().to_dict()
y_offset, y_factor = dfout_TEE.mean().to_dict(), dfout_TEE.std().to_dict()

dfin_TEE = (dfin_TEE - dfin_TEE.mean()).divide(dfin_TEE.std())
dfout_TEE = (dfout_TEE - dfout_TEE.mean()).divide(dfout_TEE.std())

#Save the scaling parameters of the inputs for OMLT
scaled_lb = dfin_TEE.min()[inputs].values
scaled_ub = dfin_TEE.max()[inputs].values
scaled_input_bounds_TEE = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}
# scaling factors
scaler_TEE = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )

try:
    dir = './results/Models/CS2_ReLU_TEE.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/CS2_ReLU_TEE.onnx'
    onnx_model = onnx.load(dir)
net_TEE = load_onnx_neural_network(onnx_model, scaler_TEE, scaled_input_bounds_TEE)


try:
    dir = './data/CS2_Sampling/Batch_Reactor_NN_TGE'
    df = pd.read_csv(dir) 
except:
    dir = '../data/CS2_Sampling/Batch_Reactor_NN_TGE'
    df = pd.read_csv(dir)

dfin_TGE = df[inputs]
dfout_TGE = df[outputs]

#Scaling
x_offset, x_factor = dfin_TGE.mean().to_dict(), dfin_TGE.std().to_dict()
y_offset, y_factor = dfout_TGE.mean().to_dict(), dfout_TGE.std().to_dict()

dfin_TGE = (dfin_TGE - dfin_TGE.mean()).divide(dfin_TGE.std())
dfout_TGE = (dfout_TGE - dfout_TGE.mean()).divide(dfout_TGE.std())

#Save the scaling parameters of the inputs for OMLT
scaled_lb = dfin_TGE.min()[inputs].values
scaled_ub = dfin_TGE.max()[inputs].values
scaled_input_bounds_TGE = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}
# scaling factors
scaler_TGE = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )

try:
    dir = './results/Models/CS2_ReLU_TGE.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/CS2_ReLU_TGE.onnx'
    onnx_model = onnx.load(dir)
net_TGE = load_onnx_neural_network(onnx_model, scaler_TGE, scaled_input_bounds_TGE)


try:
    dir = './data/CS2_Sampling/Batch_Reactor_NN_TI'
    df = pd.read_csv(dir) 
except:
    dir = '../data/CS2_Sampling/Batch_Reactor_NN_TI'
    df = pd.read_csv(dir)

dfin_TI = df[inputs]
dfout_TI = df[outputs]

#Scaling
x_offset, x_factor = dfin_TI.mean().to_dict(), dfin_TI.std().to_dict()
y_offset, y_factor = dfout_TI.mean().to_dict(), dfout_TI.std().to_dict()

dfin_TI = (dfin_TI - dfin_TI.mean()).divide(dfin_TI.std())
dfout_TI = (dfout_TI - dfout_TI.mean()).divide(dfout_TI.std())

#Save the scaling parameters of the inputs for OMLT
scaled_lb = dfin_TI.min()[inputs].values
scaled_ub = dfin_TI.max()[inputs].values
scaled_input_bounds_TI = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}
# scaling factors
scaler_TI = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )

try:
    dir = './results/Models/CS2_ReLU_TI.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/CS2_ReLU_TI.onnx'
    onnx_model = onnx.load(dir)
net_TI = load_onnx_neural_network(onnx_model, scaler_TI, scaled_input_bounds_TI)



def centralised(data, fix_TP=False):

    N_t = data[None]['N_t'][None]
    to_Tc = lambda t: state_to_control_t(t, N_t, data[None]['T'][None])

    model = pyo.AbstractModel()

    model.P = pyo.Set()
    model.L = pyo.Set()
    model.R = pyo.Set()
    model.T = pyo.Set()
    model.N_t = pyo.Param()
    model.Tc = pyo.Set()

    model.CP = pyo.Param(model.P)
    model.U = pyo.Param(model.L, model.R)
    model.CT = pyo.Param(model.L)
    model.CS = pyo.Param(model.P)
    model.Price = pyo.Param(model.P)
    model.LT = pyo.Param(model.L)
    model.Q = pyo.Param(model.P)
    model.A = pyo.Param(model.L, model.R)
    model.X = pyo.Param(model.L, model.P)

    model.Plan = pyo.Param()

    model.SAIP0 = pyo.Param()
    model.SI0 = pyo.Param()
    model.SAIS0 = pyo.Param(model.L)
    model.S0 = pyo.Param(model.P)
    model.IAIPstar0 = pyo.Param()
    model.IIstar0 = pyo.Param()
    model.IAISstar0 = pyo.Param(model.L)
    model.Istar0 = pyo.Param(model.P)
    model.CS_SAIS = pyo.Param(model.L)
    model.CS_AIP = pyo.Param()
    model.CS_I = pyo.Param()
    model.RM_Cost = pyo.Param()
    model.AI_Cost = pyo.Param()
    model.CP_I = pyo.Param()
    model.CP_AI = pyo.Param()
    model.SP_AI = pyo.Param()

    model.F = pyo.Param(model.P, model.T)

    model.Prod = pyo.Var(model.P, model.L, model.T, within=pyo.NonNegativeReals)
    model.PAI = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PI = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.S = pyo.Var(model.P, model.T, within=pyo.NonNegativeReals)
    model.SAIS = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)
    model.SAIP = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.SI = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.TP = pyo.Var(model.L, model.Tc, within=pyo.NonNegativeReals)
    model.SA = pyo.Var(model.P, model.T, within=pyo.NonNegativeReals)
    model.Cost_Eu = pyo.Var()
    model.Cost_Am = pyo.Var()
    model.Cost_As = pyo.Var()
    model.Cost_Central = pyo.Var()

    model.CCH = pyo.Param(model.P, model.P)
    model.Y = pyo.Var(model.P, model.L, model.T, within=pyo.Binary)
    model.B = pyo.Var(model.P, model.P, model.L, model.T, within=pyo.Binary)

    if fix_TP:
        model.TP_fixed = pyo.Param(model.L, model.Tc)

        def constrain_TP(m, l, tc):
            return m.TP[l, tc] == m.TP_fixed[l, tc]
        model.h_TP = pyo.Constraint(model.L, model.Tc, rule=constrain_TP)

    def o(m):
        return m.Cost_Central / 1e6
    model.obj = pyo.Objective(rule=o)

    def cost_total(m):
        prod_cost = (
            sum(m.CP[p] * m.Prod[p, l, t] for p in m.P for l in m.L for t in m.T)
            + sum(m.CP_I * m.PI[t] for t in m.T)
            + sum(m.CP_AI * m.PAI[t] for t in m.T)
        )
        transp_cost = sum(m.CT[l] * m.TP[l, to_Tc(t)] for l in m.L for t in m.T)
        store_cost = (
            sum(m.CS[p] * m.S[p, t] for p in m.P for t in m.T)
            + sum(m.CS_SAIS[l] * m.SAIS[l, t] for l in m.L for t in m.T)
            + sum(m.CS_AIP * m.SAIP[t] for t in m.T)
            + sum(m.CS_I * m.SI[t] for t in m.T)
        )
        # + sum(m.CS_Europe['RM']*m.SI[t] for t in m.T)
        sales = sum(m.Price[p] * m.SA[p, t] for p in m.P for t in m.T)
        changeover = sum(
            m.CCH[p1, p2]*m.B[p1, p2, l, t] for p2 in m.P for p1 in m.P for l in m.L for t in m.T if p1 != p2
            ) + sum(
                1000*m.Y[p, l, t] for p in m.P for l in m.L for t in m.T
            )
        return m.Cost_Central == prod_cost + transp_cost + store_cost + changeover - sales

    model.Central_Cost = pyo.Constraint(rule=cost_total)

    def CoEu(m):
        prod_cost = sum(m.CP_I * m.PI[t] for t in m.T) + sum(
            m.CP_AI * m.PAI[t] for t in m.T
        )
        # no transp_cost
        store_cost = sum(m.CS_AIP * m.SAIP[t] for t in m.T) + sum(
            m.CS_I * m.SI[t] for t in m.T
        )
        sales = sum(m.SP_AI * m.TP[l, to_Tc(t)] for l in m.L for t in m.T)
        return m.Cost_Eu == prod_cost + store_cost - sales

    model.EuCost = pyo.Constraint(rule=CoEu)

    def CoAs(m):
        prod_cost = sum(m.CP[p] * m.Prod[p, "Asia", t] for p in m.P for t in m.T)
        transp_cost = sum(m.CT["Asia"] * m.TP["Asia", to_Tc(t)] for t in m.T)
        store_cost = (
            sum(m.CS_SAIS["Asia"] * m.SAIS["Asia", t] for t in m.T)
            + sum(m.CS["PA"] * m.S["PA", t] for t in m.T)
            + sum(m.CS["PB"] * m.S["PB", t] for t in m.T)
            + sum(m.CS["TEE"] * m.S["TEE", t] for t in m.T)
            + sum(m.CS["TGE"] * m.S["TGE", t] for t in m.T)
        )
        sales = (
            sum(m.Price["PA"] * m.SA["PA", t] for t in m.T)
            + sum(m.Price["PB"] * m.SA["PB", t] for t in m.T)
            + sum(m.Price["TEE"] * m.SA["TEE", t] for t in m.T)
            + sum(m.Price["TGE"] * m.SA["TGE", t] for t in m.T)
            - sum(m.SP_AI * m.TP["Asia", to_Tc(t)] for t in m.T)
        )
        changeover = sum(
                m.CCH[p1, p2]*m.B[p1, p2, "Asia", t] for p1 in m.P for p2 in m.P for t in m.T if p1 != p2
            ) + sum(
                1000*m.Y[p, "Asia", t] for p in m.P for t in m.T
            )
        return m.Cost_As == prod_cost + store_cost + transp_cost + changeover - sales

    model.AsCost = pyo.Constraint(rule=CoAs)

    def CoAm(m):
        prod_cost = sum(m.CP[p] * m.Prod[p, "America", t] for p in m.P for t in m.T)
        transp_cost = sum(m.CT["America"] * m.TP["America", to_Tc(t)] for t in m.T)
        store_cost = sum(m.CS_SAIS["America"] * m.SAIS["America", t] for t in m.T) + \
            sum(m.CS["PC"] * m.S["PC", t] for t in m.T) + \
            sum(m.CS["PD"] * m.S["PD", t] for t in m.T)
        sales = sum(m.Price["PC"] * m.SA["PC", t] for t in m.T) + \
            sum(m.Price["PD"] * m.SA["PD", t] for t in m.T) - sum(
            m.SP_AI * m.TP["America", to_Tc(t)] for t in m.T
        )
        changeover = sum(
            m.CCH[p1, p2]*m.B[p1, p2, "America", t] for p1 in m.P for p2 in m.P for t in m.T if p1 != p2
            ) + sum(
                1000*m.Y[p, "America", t] for p in m.P for t in m.T
            )
        return m.Cost_Am == prod_cost + store_cost + transp_cost + changeover - sales

    model.AmCost = pyo.Constraint(rule=CoAm)

    def link_B(m, p, l, t):
        return sum(m.B[p, p2, l, t] for p2 in m.P) <= 1
    def link_B1(m, p1, p2, l, t):
        if p1 != p2:
            return m.B[p1, p2, l, t] + m.B[p2, p1, l, t] <= m.Y[p1, l, t]
        else:
            return m.B[p1, p2, l, t] + m.B[p2, p1, l, t] == 0
    def link_B2(m, p1, p2, l, t):
        if p1 != p2:
            return m.B[p1, p2, l, t] + m.B[p2, p1, l, t] <= m.Y[p2, l, t]
        else:
            return pyo.Constraint.Skip

    def link_Prod_Y(m, p, l, t):
        return m.Prod[p, l, t] <= m.Y[p, l, t]*1e7

    def link_B_Y2(m, l, t):
        lhs = 0.5*sum(m.B[p1, p2, l, t] + m.B[p2, p1, l, t] for p2 in m.P for p1 in m.P)
        rhs = sum(m.Y[p, l, t] for p in m.P) - 1
        return lhs >= rhs

    def link_B_Y1(m, p, l, t):
        lhs = sum(m.B[p, p2, l, t] + m.B[p2, p, l, t] for p2 in m.P)
        rhs = m.Y[p, l, t] 
        return lhs >= rhs

    model.Blink = pyo.Constraint(model.P, model.L, model.T, rule=link_B)
    model.B1link = pyo.Constraint(model.P, model.P, model.L, model.T, rule=link_B1)
    model.B2link = pyo.Constraint(model.P, model.P, model.L, model.T, rule=link_B2)
    model.PYlink = pyo.Constraint(model.P, model.L, model.T, rule=link_Prod_Y)
    model.BY2link = pyo.Constraint(model.L, model.T, rule=link_B_Y2)
    model.BY1link = pyo.Constraint(model.P, model.L, model.T, rule=link_B_Y1)

    def resource_constraint(m, l, r, t):
        return (
            m.U[l, r]
            * sum(
                m.Prod[p, l, t]*m.Q[p] for p in m.P
            )
            <= m.A[l, r]
        )

    model.res_constr = pyo.Constraint(
        model.L, model.R, model.T, rule=resource_constraint
    )

    def inv_t(m, p, t):
        if t > 1:
            return m.S[p, t] == m.S[p, t - 1] - m.SA[p, t] + sum(
                m.Prod[p, l, t] for l in m.L
            )
        else:
            return pyo.Constraint.Skip

    def inv_0(m, p):
        return m.S[p, 1] == m.S0[p] - m.SA[p, 1] + sum(m.Prod[p, l, 1] for l in m.L)

    model.S_t = pyo.Constraint(model.P, model.T, rule=inv_t)
    model.S_0 = pyo.Constraint(model.P, rule=inv_0)

    def invAIS_t(m, l, t):
        t_TP = t - m.LT[l]
        if t > 1:
            if t_TP > 0:
                return m.SAIS[l, t] == m.SAIS[l, t - 1] + m.TP[l, to_Tc(t_TP)] - 1.1 * sum(
                    m.Prod[p, l, t] * m.Q[p] for p in m.P
                )
            else:
                return m.SAIS[l, t] == m.SAIS[l, t - 1]  - 1.1 * sum(
                    m.Prod[p, l, t] * m.Q[p] for p in m.P
                )
        else:
            return pyo.Constraint.Skip

    def invAIS_0(m, l):
        return m.SAIS[l, 1] == m.SAIS0[l] - 1.1 * sum(
            m.Prod[p, l, 1] * m.Q[p] for p in m.P
        )

    model.SAIS_t = pyo.Constraint(model.L, model.T, rule=invAIS_t)
    model.SAIS_0 = pyo.Constraint(model.L, rule=invAIS_0)

    def invAIP_t(m, t):
        if t > 1:
            return m.SAIP[t] == m.SAIP[t - 1] + m.PAI[t] - sum(m.TP[l, to_Tc(t)] for l in m.L)
        else:
            return pyo.Constraint.Skip

    def invAIP_0(m):
        return m.SAIP[1] == m.SAIP0 + m.PAI[1] - sum(m.TP[l, 1] for l in m.L)

    model.SAIP_t = pyo.Constraint(model.T, rule=invAIP_t)
    model.SAIP_0 = pyo.Constraint(rule=invAIP_0)

    def invI_t(m, t):
        if t > 1:
            return m.SI[t] == m.SI[t - 1] + m.PI[t] - 1.1 * m.PAI[t]
        else:
            return pyo.Constraint.Skip

    def invI_0(m):
        return m.SI[1] == m.SI0 + m.PI[1] - 1.1 * m.PAI[1]

    model.SI_t = pyo.Constraint(model.T, rule=invI_t)
    model.SI_0 = pyo.Constraint(rule=invI_0)

    def Prod_UL(m, p, l, t):
        return m.Prod[p, l, t] <= 500e3 * m.X[l, p]

    model.P_UL = pyo.Constraint(model.P, model.L, model.T, rule=Prod_UL)

    # Sales equals forecast

    def sales_UL(m, p, t):
        return m.SA[p, t] <= m.F[p, t]

    # def sales_LB(m, p, t):
    #     return m.SA[p,t] >= 0.5*m.F[p,t]
    model.SA_UL = pyo.Constraint(model.P, model.T, rule=sales_UL)
    # model.SA_LB = pyo.Constraint(model.P, model.T, rule = sales_LB)

    def safeS(m, p, t):
        if t > 4:
            return m.S[p, t] >= m.Istar0[p]
        else:
            # return m.S[p, t] >= 0
            return m.S[p, t] >= m.Istar0[p] / 4

    def safeSI(m, t):
        if t > 4:
            return m.SI[t] >= m.IIstar0
        else:
            # return m.SI[t] >= 0
            return m.SI[t] >= m.IIstar0 / 4

    def safeSAIS(m, l, t):
        if t > 4:
            return m.SAIS[l, t] >= m.IAISstar0[l]
        else:
            # return m.SAIS[l, t] >= 0
            return m.SAIS[l, t] >= m.IAISstar0[l] / 4

    def safeSAIP(m, t):
        if t > 4:
            return m.SAIP[t] >= m.IAIPstar0
        else:
            # return m.SAIP[t] >= 0
            return m.SAIP[t] >= m.IAIPstar0 / 4

    model.safeI = pyo.Constraint(model.P, model.T, rule=safeS)
    model.safeII = pyo.Constraint(model.T, rule=safeSI)
    model.safeIAIS = pyo.Constraint(model.L, model.T, rule=safeSAIS)
    model.safeIAIP = pyo.Constraint(model.T, rule=safeSAIP)

    ins = model.create_instance(data)
    # set initial stuff

    # solver = pyo.SolverFactory("mosek")
    # solver = pyo.SolverFactory("bonmin")
    solver = pyo.SolverFactory("gurobi_direct")
    solver.options['TimeLimit'] = 300
    solver.solve(ins)

    return ins



def scheduling_Asia_bi_complete(data, tightened=True):

    model = pyo.ConcreteModel()
    model.N = pyo.Set(initialize=data[None]['N'])
    model.I = pyo.Set(initialize=data[None]['I'])
    model.R = pyo.Set(initialize=data[None]['R'])
    model.states = pyo.Set(initialize=data[None]['states'])

    model.tasks = pyo.Set(within = model.R*model.I, initialize=data[None]['tasks'])
    model.N_last = pyo.Param(initialize=data[None]['N_last'])
    model.alpha = pyo.Param(model.I, initialize=data[None]['alpha'])
    model.beta = pyo.Param(model.I, initialize=data[None]['beta'])
    # model.beta_var = pyo.Var(model.I, within=pyo.NonNegativeReals)
    model.H = pyo.Param(initialize=data[None]['H'])
    model.Bmin = pyo.Param(model.I, initialize=data[None]['Bmin'])
    model.Bmax = pyo.Param(model.I, initialize=data[None]['Bmax'])
    model.rho = pyo.Param(model.I, model.states, initialize=data[None]['rho'])
    model.S_in = pyo.Set(within = model.I*model.states, initialize=data[None]['S_in'])
    model.S_out = pyo.Set(within = model.I*model.states, initialize=data[None]['S_out'])
    model.in_s = pyo.Set(within = model.states*model.I, initialize=data[None]['in_s'])
    model.out_s = pyo.Set(within = model.states*model.I, initialize=data[None]['out_s'])
    model.Prod = pyo.Param(model.states, initialize=data[None]['Prod'])
    model.proc_time = pyo.Param(model.I, model.I, initialize=data[None]['proc_time'])
    model.kappa = pyo.Param(model.I, model.I, initialize=data[None]['kappa'])
    model.S0 = pyo.Param(model.states, initialize=data[None]['S0'])
    model.CS = pyo.Param(model.states, initialize=data[None]['CS'])

    model.T = pyo.Var(model.N, within=pyo.NonNegativeReals)
    model.Ts = pyo.Var(model.I, model.N, within=pyo.NonNegativeReals)
    model.Tf = pyo.Var(model.I, model.N, within=pyo.NonNegativeReals)
    model.D = pyo.Var(model.I, model.N, within=pyo.NonNegativeReals)
    model.T_CCH = pyo.Var(model.I, model.N, within=pyo.NonNegativeReals)
    model.Ws = pyo.Var(model.I, model.N, within=pyo.Binary)
    # model.Wp = pyo.Var(model.I, model.N, within=pyo.Binary)
    model.Wf = pyo.Var(model.I, model.N, within=pyo.Binary)
    # model.Zp = pyo.Var(model.R, model.N, within=pyo.Binary)
    # model.Zf = pyo.Var(model.R, model.N, within=pyo.Binary)
    # model.Zs = pyo.Var(model.R, model.N, within=pyo.Binary)
    model.Bs = pyo.Var(model.I, model.N, within=pyo.NonNegativeReals)
    model.Bp = pyo.Var(model.I, model.N, within=pyo.NonNegativeReals)
    model.Bf = pyo.Var(model.I, model.N, within=pyo.NonNegativeReals)
    model.B_cons = pyo.Var(model.I, model.states, model.N, within=pyo.NonNegativeReals)
    model.B_prod = pyo.Var(model.I, model.states, model.N, within=pyo.NonNegativeReals)
    model.S = pyo.Var(model.states, model.N, within=pyo.NonNegativeReals)
    model.MS = pyo.Var()
    model.Y = pyo.Var(model.I, model.I, model.N, within = pyo.Binary)
    # model.Y = pyo.Var(model.I, model.I, model.N, within = pyo.NonNegativeReals)

    model.CCH_cost = pyo.Var()
    model.st_cost = pyo.Var()

    model.linking = pyo.ConstraintList()

    model.tf_PA1 = pyo.Var(model.N)
    model.tf_PA2 = pyo.Var(model.N)
    model.energy_PA1 = pyo.Var(model.N)
    model.energy_PA2 = pyo.Var(model.N)

    model.NNPA1_0 = OmltBlock()
    model.NNPA1_1 = OmltBlock()
    model.NNPA1_2 = OmltBlock()
    model.NNPA1_3 = OmltBlock()
    model.NNPA1_4 = OmltBlock()
    model.NNPA1_5 = OmltBlock()
    model.NNPA1_6 = OmltBlock()
    # model.NN_7 = OmltBlock()

    model.NNPA1_0.build_formulation(ReluBigMFormulation(net))
    model.NNPA1_1.build_formulation(ReluBigMFormulation(net))
    model.NNPA1_2.build_formulation(ReluBigMFormulation(net))
    model.NNPA1_3.build_formulation(ReluBigMFormulation(net))
    model.NNPA1_4.build_formulation(ReluBigMFormulation(net))
    model.NNPA1_5.build_formulation(ReluBigMFormulation(net))
    model.NNPA1_6.build_formulation(ReluBigMFormulation(net))
    # model.NN_7.build_formulation(ReluBigMFormulation(net))

    model.linking.add(model.Bs['PA1',0] == model.NNPA1_0.inputs[0])
    model.linking.add(model.Bs['PA1',1] == model.NNPA1_1.inputs[0])
    model.linking.add(model.Bs['PA1',2] == model.NNPA1_2.inputs[0])
    model.linking.add(model.Bs['PA1',3] == model.NNPA1_3.inputs[0])
    model.linking.add(model.Bs['PA1',4] == model.NNPA1_4.inputs[0])
    model.linking.add(model.Bs['PA1',5] == model.NNPA1_5.inputs[0])
    model.linking.add(model.Bs['PA1',6] == model.NNPA1_6.inputs[0])
    # m.linking.add(m.vol['Reaction','Reactor',7] == m.NN_7.inputs[0])

    # # #Output tf
    model.linking.add(model.tf_PA1[0] == model.NNPA1_0.outputs[0])
    model.linking.add(model.tf_PA1[1] == model.NNPA1_1.outputs[0])
    model.linking.add(model.tf_PA1[2] == model.NNPA1_2.outputs[0])
    model.linking.add(model.tf_PA1[3] == model.NNPA1_3.outputs[0])
    model.linking.add(model.tf_PA1[4] == model.NNPA1_4.outputs[0])
    model.linking.add(model.tf_PA1[5] == model.NNPA1_5.outputs[0])
    model.linking.add(model.tf_PA1[6] == model.NNPA1_6.outputs[0])
    # m.linking.add(m.tf['Reactor',7] == m.NN_7.outputs[0])

    # # Output Q
    model.linking.add(model.energy_PA1[0] == model.NNPA1_0.outputs[1])
    model.linking.add(model.energy_PA1[1] == model.NNPA1_1.outputs[1])
    model.linking.add(model.energy_PA1[2] == model.NNPA1_2.outputs[1])
    model.linking.add(model.energy_PA1[3] == model.NNPA1_3.outputs[1])
    model.linking.add(model.energy_PA1[4] == model.NNPA1_4.outputs[1])
    model.linking.add(model.energy_PA1[5] == model.NNPA1_5.outputs[1])
    model.linking.add(model.energy_PA1[6] == model.NNPA1_6.outputs[1])
    # m.linking.add(m.Q['Reaction','Reactor',7] == m.NN_7.outputs[1])

    model.NNPA2_0 = OmltBlock()
    model.NNPA2_1 = OmltBlock()
    model.NNPA2_2 = OmltBlock()
    model.NNPA2_3 = OmltBlock()
    model.NNPA2_4 = OmltBlock()
    model.NNPA2_5 = OmltBlock()
    model.NNPA2_6 = OmltBlock()
    # model.NN_7 = OmltBlock()

    model.NNPA2_0.build_formulation(ReluBigMFormulation(net))
    model.NNPA2_1.build_formulation(ReluBigMFormulation(net))
    model.NNPA2_2.build_formulation(ReluBigMFormulation(net))
    model.NNPA2_3.build_formulation(ReluBigMFormulation(net))
    model.NNPA2_4.build_formulation(ReluBigMFormulation(net))
    model.NNPA2_5.build_formulation(ReluBigMFormulation(net))
    model.NNPA2_6.build_formulation(ReluBigMFormulation(net))
    # model.NN_7.build_formulation(ReluBigMFormulation(net))

    model.linking.add(model.Bs['PA2',0] == model.NNPA2_0.inputs[0])
    model.linking.add(model.Bs['PA2',1] == model.NNPA2_1.inputs[0])
    model.linking.add(model.Bs['PA2',2] == model.NNPA2_2.inputs[0])
    model.linking.add(model.Bs['PA2',3] == model.NNPA2_3.inputs[0])
    model.linking.add(model.Bs['PA2',4] == model.NNPA2_4.inputs[0])
    model.linking.add(model.Bs['PA2',5] == model.NNPA2_5.inputs[0])
    model.linking.add(model.Bs['PA2',6] == model.NNPA2_6.inputs[0])
    # m.linking.add(m.vol['Reaction','Reactor',7] == m.NN_7.inputs[0])

    # # #Output tf
    model.linking.add(model.tf_PA2[0] == model.NNPA2_0.outputs[0])
    model.linking.add(model.tf_PA2[1] == model.NNPA2_1.outputs[0])
    model.linking.add(model.tf_PA2[2] == model.NNPA2_2.outputs[0])
    model.linking.add(model.tf_PA2[3] == model.NNPA2_3.outputs[0])
    model.linking.add(model.tf_PA2[4] == model.NNPA2_4.outputs[0])
    model.linking.add(model.tf_PA2[5] == model.NNPA2_5.outputs[0])
    model.linking.add(model.tf_PA2[6] == model.NNPA2_6.outputs[0])
    # m.linking.add(m.tf['Reactor',7] == m.NN_7.outputs[0])

    # # Output Q
    model.linking.add(model.energy_PA2[0] == model.NNPA2_0.outputs[1])
    model.linking.add(model.energy_PA2[1] == model.NNPA2_1.outputs[1])
    model.linking.add(model.energy_PA2[2] == model.NNPA2_2.outputs[1])
    model.linking.add(model.energy_PA2[3] == model.NNPA2_3.outputs[1])
    model.linking.add(model.energy_PA2[4] == model.NNPA2_4.outputs[1])
    model.linking.add(model.energy_PA2[5] == model.NNPA2_5.outputs[1])
    model.linking.add(model.energy_PA2[6] == model.NNPA2_6.outputs[1])
    # m.linking.add(m.Q['Reaction','Reactor',7] == m.NN_7.outputs[1])

    
    model.tf_PB1 = pyo.Var(model.N)
    model.tf_PB2 = pyo.Var(model.N)
    model.energy_PB1 = pyo.Var(model.N)
    model.energy_PB2 = pyo.Var(model.N)

    model.NNPB1_0 = OmltBlock()
    model.NNPB1_1 = OmltBlock()
    model.NNPB1_2 = OmltBlock()
    model.NNPB1_3 = OmltBlock()
    model.NNPB1_4 = OmltBlock()
    model.NNPB1_5 = OmltBlock()
    model.NNPB1_6 = OmltBlock()
    # model.NN_7 = OmltBlock()

    model.NNPB1_0.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB1_1.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB1_2.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB1_3.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB1_4.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB1_5.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB1_6.build_formulation(ReluBigMFormulation(net_PB))
    # model.NN_7.build_formulation(ReluBigMFormulation(net))

    model.linking.add(model.Bs['PB1',0] == model.NNPB1_0.inputs[0])
    model.linking.add(model.Bs['PB1',1] == model.NNPB1_1.inputs[0])
    model.linking.add(model.Bs['PB1',2] == model.NNPB1_2.inputs[0])
    model.linking.add(model.Bs['PB1',3] == model.NNPB1_3.inputs[0])
    model.linking.add(model.Bs['PB1',4] == model.NNPB1_4.inputs[0])
    model.linking.add(model.Bs['PB1',5] == model.NNPB1_5.inputs[0])
    model.linking.add(model.Bs['PB1',6] == model.NNPB1_6.inputs[0])
    # m.linking.add(m.vol['Reaction','Reactor',7] == m.NN_7.inputs[0])

    # # #Output tf
    model.linking.add(model.tf_PB1[0] == model.NNPB1_0.outputs[0])
    model.linking.add(model.tf_PB1[1] == model.NNPB1_1.outputs[0])
    model.linking.add(model.tf_PB1[2] == model.NNPB1_2.outputs[0])
    model.linking.add(model.tf_PB1[3] == model.NNPB1_3.outputs[0])
    model.linking.add(model.tf_PB1[4] == model.NNPB1_4.outputs[0])
    model.linking.add(model.tf_PB1[5] == model.NNPB1_5.outputs[0])
    model.linking.add(model.tf_PB1[6] == model.NNPB1_6.outputs[0])
    # m.linking.add(m.tf['Reactor',7] == m.NN_7.outputs[0])

    # # Output Q
    model.linking.add(model.energy_PB1[0] == model.NNPB1_0.outputs[1])
    model.linking.add(model.energy_PB1[1] == model.NNPB1_1.outputs[1])
    model.linking.add(model.energy_PB1[2] == model.NNPB1_2.outputs[1])
    model.linking.add(model.energy_PB1[3] == model.NNPB1_3.outputs[1])
    model.linking.add(model.energy_PB1[4] == model.NNPB1_4.outputs[1])
    model.linking.add(model.energy_PB1[5] == model.NNPB1_5.outputs[1])
    model.linking.add(model.energy_PB1[6] == model.NNPB1_6.outputs[1])
    # m.linking.add(m.Q['Reaction','Reactor',7] == m.NN_7.outputs[1])

    model.NNPB2_0 = OmltBlock()
    model.NNPB2_1 = OmltBlock()
    model.NNPB2_2 = OmltBlock()
    model.NNPB2_3 = OmltBlock()
    model.NNPB2_4 = OmltBlock()
    model.NNPB2_5 = OmltBlock()
    model.NNPB2_6 = OmltBlock()
    # model.NN_7 = OmltBlock()

    model.NNPB2_0.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB2_1.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB2_2.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB2_3.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB2_4.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB2_5.build_formulation(ReluBigMFormulation(net_PB))
    model.NNPB2_6.build_formulation(ReluBigMFormulation(net_PB))
    # model.NN_7.build_formulation(ReluBigMFormulation(net))

    model.linking.add(model.Bs['PB2',0] == model.NNPB2_0.inputs[0])
    model.linking.add(model.Bs['PB2',1] == model.NNPB2_1.inputs[0])
    model.linking.add(model.Bs['PB2',2] == model.NNPB2_2.inputs[0])
    model.linking.add(model.Bs['PB2',3] == model.NNPB2_3.inputs[0])
    model.linking.add(model.Bs['PB2',4] == model.NNPB2_4.inputs[0])
    model.linking.add(model.Bs['PB2',5] == model.NNPB2_5.inputs[0])
    model.linking.add(model.Bs['PB2',6] == model.NNPB2_6.inputs[0])
    # m.linking.add(m.vol['Reaction','Reactor',7] == m.NN_7.inputs[0])

    # # #Output tf
    model.linking.add(model.tf_PB2[0] == model.NNPB2_0.outputs[0])
    model.linking.add(model.tf_PB2[1] == model.NNPB2_1.outputs[0])
    model.linking.add(model.tf_PB2[2] == model.NNPB2_2.outputs[0])
    model.linking.add(model.tf_PB2[3] == model.NNPB2_3.outputs[0])
    model.linking.add(model.tf_PB2[4] == model.NNPB2_4.outputs[0])
    model.linking.add(model.tf_PB2[5] == model.NNPB2_5.outputs[0])
    model.linking.add(model.tf_PB2[6] == model.NNPB2_6.outputs[0])
    # m.linking.add(m.tf['Reactor',7] == m.NN_7.outputs[0])

    # # Output Q
    model.linking.add(model.energy_PB2[0] == model.NNPB2_0.outputs[1])
    model.linking.add(model.energy_PB2[1] == model.NNPB2_1.outputs[1])
    model.linking.add(model.energy_PB2[2] == model.NNPB2_2.outputs[1])
    model.linking.add(model.energy_PB2[3] == model.NNPB2_3.outputs[1])
    model.linking.add(model.energy_PB2[4] == model.NNPB2_4.outputs[1])
    model.linking.add(model.energy_PB2[5] == model.NNPB2_5.outputs[1])
    model.linking.add(model.energy_PB2[6] == model.NNPB2_6.outputs[1])
    # m.linking.add(m.Q['Reaction','Reactor',7] == m.NN_7.outputs[1])


    ## Initial state
    def init(m, s):
        return m.S[s, 0] == m.S0[s] - sum(m.B_cons[i,s,0] for i in m.I if (s,i) in m.in_s)
    model.c_init = pyo.Constraint(model.states, rule=init)
    def initT(m):
        return m.T[0] == 0
    model.c_initT = pyo.Constraint(rule=initT)
    
    ## Assignment constraints
    def constr1(m, r, n):
        return sum(m.Ws[i,n] for i in m.I if (r,i) in m.tasks) <= 1
    model.c1 = pyo.Constraint(model.R, model.N, rule=constr1)
    def constr2(m, r, n):
        return sum(m.Wf[i,n] for i in m.I if (r,i) in m.tasks) <= 1
    model.c2 = pyo.Constraint(model.R, model.N, rule=constr2)
    def constr3(m, r, n):
        return sum(m.Ws[i,n_] - m.Wf[i, n_] for n_ in m.N for i in m.I if (r,i) in m.tasks and n_ <= n) <= 1
    model.c3 = pyo.Constraint(model.R, model.N, rule=constr3)
    def constr4(m, i):
        return sum(m.Ws[i,n] for n in m.N) == sum(m.Wf[i,n] for n in m.N)
    model.c4 = pyo.Constraint(model.I, rule=constr4)
    def constr5(m, i):
        return m.Wf[i,0] == 0.
    model.c5 = pyo.Constraint(model.I, rule=constr5)
    def constr6(m, i, n):
        if n == m.N_last:
            return m.Ws[i,n] == 0.
        else:
            return pyo.Constraint.Skip
    model.c6 = pyo.Constraint(model.I, model.N, rule=constr6)
    ### Test ###
    def test_constr(m, r, i, n):
        if n < m.N_last and (r,i) in m.tasks:
            return sum(m.Ws[i_,n] for i_ in m.I if (r,i_) in m.tasks) >= m.Wf[i,n]
        else:
            return pyo.Constraint.Skip
    model.c_test = pyo.Constraint(model.R, model.I, model.N, rule=test_constr)
    ###      ###

    ## Duration, finish time, and time-matching constraints

    def constr7(m, i, n):
        if i[:-1] != 'PA' and i[:-1] != 'PB':
            return m.D[i, n] == m.alpha[i]*m.Ws[i,n] + m.beta[i]*m.Bs[i,n]
        elif i=='PA1':
            return m.D[i, n] == (m.alpha[i]+ m.tf_PA1[n])*m.Ws[i,n]
        elif i=='PA2':
            return m.D[i, n] == (m.alpha[i]+ m.tf_PA2[n])*m.Ws[i,n]
        elif i=='PB1':
            return m.D[i, n] == (m.alpha[i]+ m.tf_PB1[n])*m.Ws[i,n]
        else:
            return m.D[i, n] == (m.alpha[i]+ m.tf_PB2[n])*m.Ws[i,n]
    model.c7 = pyo.Constraint(model.I, model.N, rule=constr7)

    # def constr7(m, i, n):
    #     return m.D[i, n] == m.alpha[i]*m.Ws[i,n] + m.beta[i]*m.Bs[i,n]
    # model.c7 = pyo.Constraint(model.I, model.N, rule=constr7)

    # def constr7(m, i, n):
    #     return m.D[i, n] == m.alpha[i]*m.Ws[i,n] + m.beta_var[i]*m.Bs[i,n]
    # model.c7 = pyo.Constraint(model.I, model.N, rule=constr7)
    # def constr7plus(m, i):
    #     return m.beta[i] == m.beta_var[i]
    # model.c7plus = pyo.Constraint(model.I, rule=constr7plus)
    def constr7bis(m, i, n):
        if n+1 <= m.N_last:
            return m.T_CCH[i,n] == sum(m.proc_time[i,i_]*m.Y[i,i_, n+1] for i_ in m.I)
        else:
            return m.T_CCH[i,n] == 0
    model.c7bis = pyo.Constraint(model.I, model.N, rule=constr7bis)
    # def constr8(m, i, n):
    #     return m.Tf[i, n] <= m.Ts[i, n] + m.D[i, n] + m.H*(1-m.Ws[i,n])
    # model.c8 = pyo.Constraint(model.I, model.N, rule=constr8)
    ## Commented out assuming changeovers and nothing is zero-wait 
    def constr9(m, i, n):
        return m.Tf[i, n] >= m.Ts[i, n] + m.D[i, n] + m.T_CCH[i, n] - m.H*(1-m.Ws[i,n])
    model.c9 = pyo.Constraint(model.I, model.N, rule=constr9)
    def constr10(m, i, n):
        if n-1 >= 0:
            return m.Tf[i, n] - m.Tf[i, n-1] <= m.H*m.Ws[i,n]
        else:
            return pyo.Constraint.Skip
    model.c10 = pyo.Constraint(model.I, model.N, rule=constr10)
    def constr11(m, i, n):
        if n-1 >= 0:
            return m.Tf[i, n] - m.Tf[i, n-1] >= m.D[i,n] + m.T_CCH[i, n]
        else:
            return pyo.Constraint.Skip
    model.c11 = pyo.Constraint(model.I, model.N, rule=constr11)
    def constr12(m, i, n):
        return m.Ts[i, n] == m.T[n]
    model.c12 = pyo.Constraint(model.I, model.N, rule=constr12)
    def constr13(m, i, n):
        if n-1 >= 0:
            return m.Tf[i, n-1] <= m.T[n] + m.H*(1-m.Wf[i,n])
        else:
            return pyo.Constraint.Skip
    model.c13 = pyo.Constraint(model.I, model.N, rule=constr13)
    ### Assuming changeovers
    def constr13bis(m, i, n):
        if n-1 >= 0:
            return m.Tf[i, n-1] >= m.T[n] - m.H*(1-m.Wf[i,n])
        else:
            return pyo.Constraint.Skip
    model.c13bis = pyo.Constraint(model.I, model.N, rule=constr13bis)
    def constr13_2(m):
        return m.T[m.N_last] == m.MS
    def constr13_3(m, n):
        if n+1 <= m.N_last:
            return m.T[n+1] >= m.T[n]
        else:
            return pyo.Constraint.Skip
    model.c13_2 = pyo.Constraint(rule=constr13_2)
    model.c13_3 = pyo.Constraint(model.N, rule=constr13_3)

    ## Batch size constraints
    def constr14low(m, i, n):
        return m.Bmin[i]*m.Ws[i,n] <= m.Bs[i,n]
    model.c14low = pyo.Constraint(model.I, model.N, rule=constr14low)
    def constr14high(m, i, n):
        return m.Bmax[i]*m.Ws[i,n] >= m.Bs[i,n]
    model.c14high = pyo.Constraint(model.I, model.N, rule=constr14high)
    def constr15low(m, i, n):
        return m.Bmin[i]*m.Wf[i,n] <= m.Bf[i,n]
    model.c15low = pyo.Constraint(model.I, model.N, rule=constr15low)
    def constr15high(m, i, n):
        return m.Bmax[i]*m.Wf[i,n] >= m.Bf[i,n]
    model.c15high = pyo.Constraint(model.I, model.N, rule=constr15high)
    def constr16high(m, i, n):
        return m.Bmax[i]*(sum(m.Ws[i,n_] for n_ in m.N if n_<n) - sum(m.Wf[i,n_] for n_ in m.N if n_<=n) ) >= m.Bp[i,n]
    model.c16high = pyo.Constraint(model.I, model.N, rule=constr16high)
    def constr16low(m, i, n):
        return m.Bmin[i]*(sum(m.Ws[i,n_] for n_ in m.N if n_<n) - sum(m.Wf[i,n_] for n_ in m.N if n_<=n) ) <= m.Bp[i,n]
    model.c16low = pyo.Constraint(model.I, model.N, rule=constr16low)
    def constr17(m, i, n):
        if n-1 >= 0:
            return m.Bs[i, n-1] + m.Bp[i,n-1] == m.Bp[i,n] + m.Bf[i,n]
        else:
            return pyo.Constraint.Skip
    model.c17 = pyo.Constraint(model.I, model.N, rule=constr17)
    def constr18(m, i, n, s):
        if (i,s) in m.S_in:
            return m.B_cons[i,s,n] == m.rho[i, s]*m.Bs[i,n]
        elif (i,s) in m.S_out:
            return m.B_prod[i,s,n] == m.rho[i, s]*m.Bf[i,n]
        else:
            return pyo.Constraint.Skip
    model.c18 = pyo.Constraint(model.I, model.N, model.states, rule=constr18)
    def constr19(m, i, n, s):
        if (i,s) in m.S_in:
            return m.B_cons[i,s,n] <= m.Bmax[i]*m.rho[i, s]*m.Ws[i,n]
        elif (i,s) in m.S_out:
            return m.B_prod[i,s,n] <= m.Bmax[i]*m.rho[i, s]*m.Wf[i,n]
        else:
            return pyo.Constraint.Skip
    model.c19 = pyo.Constraint(model.I, model.N, model.states, rule=constr19)

    ## Mass balancfe constraint
    def constr20(m, s, n):
        if n>=1:
            return m.S[s,n] == m.S[s,n-1] + sum(m.B_prod[i,s,n] for i in m.I if (s,i) in m.out_s) - sum(m.B_cons[i,s,n] for i in m.I if (s,i) in m.in_s)
        else:
            return pyo.Constraint.Skip
    model.c20 = pyo.Constraint(model.states, model.N, rule=constr20)

    def o(m):
        return m.MS + 1*sum(m.Y[i,i_,n]/1e10 for i in m.I for i_ in m.I for n in m.N) # + sum(m.energy_1[n] + m.energy_2[n] for n in m.N
    model.obj = pyo.Objective(rule=o)

    # Meet demand
    def meet_demand(m, s):
        return m.S[s, m.N_last] >= m.Prod[s]
    model.meet_Prod = pyo.Constraint(model.states, rule=meet_demand)

    ## Tightening constraints
    def constr21(m, r):
        return sum(m.D[i,n] + m.T_CCH[i, n] for i in m.I for n in m.N if (r,i) in m.tasks) <= m.MS
    model.c21 = pyo.Constraint(model.R, rule=constr21)
    def constr22(m, r, n):
        return sum(m.D[i,n_] + m.T_CCH[i, n] for i in m.I for n_ in m.N if (r,i) in m.tasks and n_ >= n) <= m.MS - m.T[n]
    model.c22 = pyo.Constraint(model.R, model.N, rule=constr22)
    if tightened:
        def constr23(m, r, n): # <- problem
            changeover = 0
            # changeover = sum(m.proc_time[i,i_]*m.Y[i,i_,n_] for n_ in m.N for i in m.I for i_ in m.I if (r,i) in m.tasks and (r,i_) in m.tasks and (n_<=n))
            return sum(m.alpha[i]*m.Wf[i, n_] + m.beta[i]*m.Bf[i, n_] + m.T_CCH[i,n_] for i in m.I for n_ in m.N if (r,i) in m.tasks and n_ < n) + changeover <= m.T[n]
        model.c23 = pyo.Constraint(model.R, model.N, rule=constr23)

    def constr25(m, r, i, i_, n):
        if (r,i) in m.tasks and (r,i_) in m.tasks and (n>=1) and (n<m.N_last) and m.kappa[i, i_]>0:
            return m.Y[i, i_, n] >= m.Wf[i, n] + m.Ws[i_, n] - 1
        else:
            return pyo.Constraint.Skip
    model.c25 = pyo.Constraint(model.R, model.I, model.I, model.N, rule=constr25)
    
    def constr_costs(m):
        return m.CCH_cost == sum(m.Y[i,i_,n]*m.kappa[i,i_] for i in m.I for i_ in m.I for n in m.N)
    model.c_CCH = pyo.Constraint(rule=constr_costs)

    def constr_storage(m):
        return m.st_cost == sum(
                m.CS[s]*m.S[s,n] for s in m.states for n in m.N
            )
    model.c_storage = pyo.Constraint(rule=constr_storage)

    solver = pyo.SolverFactory("gurobi_direct")
    # solver.options['TimeLimit'] = 60.
    res = solver.solve(model)

    return model

def simulate(Production, TP, Forecast, Sales, data, seed=0, random=True):
    
    N_tc = data[None]['N_t'][None]
    T_set = data[None]["T"][None]
    to_Tc = lambda t: state_to_control_t(t, N_tc, T_set)

    Storage = {}
    Demand = {}
    N_t = len(T_set)
    P_set = data[None]["P"][None]

    Demand["PA"] = np.zeros(N_t)
    Demand["PC"] = np.zeros(N_t)
    Demand["TEE"] = np.zeros(N_t)
    Demand["PB"] = np.zeros(N_t)
    Demand["PD"] = np.zeros(N_t)
    Demand["TGE"] = np.zeros(N_t)

    for t in T_set:
        for p in P_set:
            if random:
                Demand[p][t - 1] = np.random.uniform(
                    0.8 * Forecast[p][t - 1], 1.2 * Forecast[p][t - 1]
                )
            else:
                Demand[p][t - 1] = Forecast[p][t-1]

    Storage["PA"] = np.zeros(N_t)
    Storage["PC"] = np.zeros(N_t)
    Storage["TEE"] = np.zeros(N_t)
    Storage["PB"] = np.zeros(N_t)
    Storage["PD"] = np.zeros(N_t)
    Storage["TGE"] = np.zeros(N_t)
    Storage["AIP"] = np.zeros(N_t)
    Storage["AIS_Am"] = np.zeros(N_t)
    Storage["AIS_As"] = np.zeros(N_t)
    Storage["I"] = np.zeros(N_t)

    for p in P_set:
        Storage[p][0] = max(0,data[None]["S0"][p] - min(Demand[p][0], Sales[p][0]) + Production[p][0])
    Storage["AIS_Am"][0] = max(0,
        data[None]["SAIS0"]["America"]
        - 1.1 * (Production["PC"][0] * data[None]["Q"]["PC"] + Production["PD"][0] * data[None]["Q"]["PD"])
    )
    Storage["AIS_As"][0] = max(0,
        data[None]["SAIS0"]["Asia"]
        - 1.1
        * (
            Production["PA"][0] * data[None]["Q"]["PA"]
                    + Production["TEE"][0] * data[None]["Q"]["TEE"] + 
                    Production["PB"][0] * data[None]["Q"]["PB"]
                    + Production["TGE"][0] * data[None]["Q"]["TGE"]
        ) # + TP["Asia"][1]
    )
    Storage["AIP"][0] = max(0,
        data[None]["SAIP0"][None] + Production["AI"][0] - (TP["America"][0] + TP["Asia"][0])
    )
    Storage["I"][0] = max(0,
        data[None]["SI0"][None] + Production["I"][0] - 1.1 * Production["AI"][0]
    )

    for t in T_set:
        for p in P_set:
            if t - 1 > 0:
                Storage[p][t - 1] = max(0,
                    Storage[p][t - 2] - min(Demand[p][t - 1], Sales[p][t-1]) + Production[p][t - 1]
                )
        t_Am = data[None]["LT"]["America"]
        t_As = data[None]["LT"]["Asia"]
        if t - 1 - t_Am >= 0:
            Storage["AIS_Am"][t - 1] = max(0,
                Storage["AIS_Am"][t - 2]
                + TP["America"][to_Tc(t-t_Am)-1]
                - 1.1 * (Production["PC"][t - 1] * data[None]["Q"]["PC"] + Production["PD"][t - 1] * data[None]["Q"]["PD"])
            )
        elif t - 1 > 0:
            Storage["AIS_Am"][t - 1] = max(0,
                Storage["AIS_Am"][t - 2] # + TP["America"]
                - 1.1 * (Production["PC"][t - 1] * data[None]["Q"]["PC"] + Production["PD"][t - 1] * data[None]["Q"]["PD"])
            )
        if t - 1 - t_As >= 0:
            Storage["AIS_As"][t - 1] = max(0,
                Storage["AIS_As"][t - 2]
                + TP["Asia"][to_Tc(t-t_As)-1]
                - 1.1
                * (
                    Production["PA"][t - 1] * data[None]["Q"]["PA"]
                    + Production["TEE"][t - 1] * data[None]["Q"]["TEE"] + 
                    Production["PB"][t - 1] * data[None]["Q"]["PB"]
                    + Production["TGE"][t - 1] * data[None]["Q"]["TGE"]
                )
            )
        elif t - 1 > 0:
            Storage["AIS_As"][t - 1] = max(0,
                Storage["AIS_As"][t - 2] # + TP["Asia"]
                - 1.1
                * (
                    Production["PA"][t - 1] * data[None]["Q"]["PA"]
                    + Production["TEE"][t - 1] * data[None]["Q"]["TEE"] + 
                    Production["PB"][t - 1] * data[None]["Q"]["PB"]
                    + Production["TGE"][t - 1] * data[None]["Q"]["TGE"]
                )
            )
        if t - 1 > 0:
            Storage["AIP"][t - 1] = max(0,
                Storage["AIP"][t - 2]
                + Production["AI"][t - 1]
                - (TP["America"][to_Tc(t)-1] + TP["Asia"][to_Tc(t)-1])
            )
            Storage["I"][t - 1] = max(0,
                Storage["I"][t - 2]
                + Production["I"][t - 1]
                - 1.1 * Production["AI"][t - 1]
            )
    return Storage, Demand


# Nt=5
# import_data[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
# t0 = time.time()
# res = centralised(import_data)
# tf = time.time()
# print('Centralized time: ', tf - t0)
# print(pyo.value(res.obj))
# print([pyo.value(res.Cost_Eu), pyo.value(res.Cost_As), pyo.value(res.Cost_Am)])

# m = res
# store_cost = (
#             sum(m.CS_SAIS["Asia"] * m.SAIS["Asia", t] for t in m.T)
#             + sum(m.CS["PA"] * m.S["PA", t] for t in m.T)
#             + sum(m.CS["PB"] * m.S["PB", t] for t in m.T)
#             + sum(m.CS["TEE"] * m.S["TEE", t] for t in m.T)
#             + sum(m.CS["TGE"] * m.S["TGE", t] for t in m.T)
#         )
# changeover = sum(
#                 m.CCH[p1, p2]*m.B[p1, p2, "Asia", t] for p1 in m.P for p2 in m.P for t in m.T if p1 != p2
#             ) + sum(
#                 1000*m.Y[p, "Asia", t] for p in m.P for t in m.T
#             )
# print(f"Storage cost: {pyo.value(store_cost)}")
# print(f"Changeover cost: {pyo.value(changeover)}")

# for p in res.P:
#     if p in scheduling_data[None]['states'][None]:
#         scheduling_data[None]['Prod'][p] = pyo.value(sum(res.Prod[p, 'Asia', t] for t in res.T)/len(res.T))
# print(scheduling_data[None]['Prod'])

# t0 = time.time()

# res_Sch = scheduling_Asia_bi_complete(scheduling_data)
# print('Objective: ', pyo.value(res_Sch.obj))
# print('Makeover: ',  pyo.value(res_Sch.MS))
# print(pyo.value(res_Sch.S['AI',0]))
# print(pyo.value(res_Sch.S['AI',res_Sch.N_last]))
# print(pyo.value(res_Sch.S['TI',res_Sch.N_last]))
# print(pyo.value(res_Sch.S['PA',res_Sch.N_last]))

# energy_PA = sum(res_Sch.energy_PA1[n] + res_Sch.energy_PA2[n] for n in res_Sch.N)
# print(f"Energy cost PA: {pyo.value(energy_PA)}")
# energy_PB = sum(res_Sch.energy_PB1[n] + res_Sch.energy_PB2[n] for n in res_Sch.N)
# print(f"Energy cost PB: {pyo.value(energy_PB)}")


# t1 = time.time()

# print('Tightened: ', (t1-t0))

# # for n in res_Sch.N:
# #     for i in res_Sch.I:
# #         if pyo.value(res_Sch.D[i,n]) > 1e-12:
# #             print('D: ', f"{pyo.value(res_Sch.D[i,n]):.3f}", i, ' at ', n)
# #             print('Ts: ', f"{pyo.value(res_Sch.Ts[i,n]):.3f}", i, ' at ', n)
# #             print('Tf: ', f"{pyo.value(res_Sch.Tf[i,n]):.3f}", i, ' at ', n)
# #         if pyo.value(res_Sch.Bs[i,n]) > 1e-12:
# #             # print('Bs: ', f"{pyo.value(res_Sch.Bs[i,n]):.3f}", i, ' at ', n) 
# #             print('Ws: ', f"{pyo.value(res_Sch.Ws[i,n]):.3f}", i, ' at ', n) 
# #         # for s in res_Sch.states:
# #         #     if (i,s) in res_Sch.S_in:
# #         #         if pyo.value(res_Sch.B_cons[i, s, n]) > 1e-12:
# #         #             print(i,s)
# #         #             print('B_cons: ', pyo.value(res_Sch.B_cons[i, s, n]), i, ' at ', n)  
# #         if pyo.value(res_Sch.Bf[i,n]) > 1e-12:
# #             # print('Bf: ', f"{pyo.value(res_Sch.Bf[i,n]):.3f}", i, ' at ', n)  
# #             print('Wf ', f"{pyo.value(res_Sch.Wf[i,n]):.3f}", i, ' at ', n)  
# #         # for s in res_Sch.states:
# #         #     if (i,s) in res_Sch.S_out:
# #         #         if pyo.value(res_Sch.B_prod[i, s, n]) > 1e-12:
# #         #             print(i,s)
# #         #             print('B_prod: ', pyo.value(res_Sch.B_prod[i, s, n]), i, ' at ', n)  
# #         # if pyo.value(res_Sch.Bp[i,n]) > 1e-12:
# #         #     print('Bp: ', f"{pyo.value(res_Sch.Bp[i,n]):.3f}", i, ' at ', n) 


# # for n in res_Sch.N:
# #     print(f"Timepoint {n}: {pyo.value(res_Sch.T[n]):.3f}")
# # #     print('TEE', pyo.value(res_Sch.S['TEE', n]))
# # #     print('TGE', pyo.value(res_Sch.S['TGE', n]))
# # #     print('TI', pyo.value(res_Sch.S['TI', n]))
# # #     print('PA', pyo.value(res_Sch.S['PA', n]))
# # #     print('PB', pyo.value(res_Sch.S['PB', n]))


# bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
# text_style = {'color':'white', 'weight':'light', 'ha':'center', 'va':'center'}
# colors = mpl.cm.Dark2.colors

# U1_I_list = ['TI1', 'TEE1', 'TGE1', 'PA1', 'PB1']
# U2_I_list = ['TI2', 'TEE2', 'TGE2', 'PA2', 'PB2']

# for n in res_Sch.N:
#     t = pyo.value(res_Sch.T[n])
#     plt.plot([t, t], [-0.5, 1.5], '--k')
# # plt.plot([0, 0], [-0.5, 1.5], '--k')
# # plt.show()

# # for i in res_Sch.I:
# #     for i_ in res_Sch.I:
# #         for n in res_Sch.N:
# #             Y_ = pyo.value(res_Sch.Y[i,i_,n])
# #             if Y_ > 1e-3:
# #                 print('Y_ : ', Y_, i,i_,n)

# for i in res_Sch.I:
#     Bmin = pyo.value(res_Sch.Bmin[i])
#     for n in res_Sch.N:
#         B = pyo.value(res_Sch.Bs[i,n])
#         if B > Bmin:
#             if i in U1_I_list:
#                 j = 0
#             else:
#                 j = 1
#             ts = pyo.value(res_Sch.T[n])
#             tf = pyo.value(res_Sch.T[n]+res_Sch.D[i,n]) 
#             text = f"{i[:-1]}"
            
#             if j == 0:
#                 k = U1_I_list.index(i)
#             else:
#                 k = U2_I_list.index(i)
#             plt.plot([ts, tf], [j]*2, c=colors[k%5], **bar_style)
#             plt.text((ts + tf)/2, j, text, **text_style, fontsize=10)

# print(f"Storage cost: {pyo.value(res_Sch.st_cost)*24}")
# print(f"Changeover cost: {pyo.value(res_Sch.CCH_cost)*24}")

# labels = ['Machine 1', 'Machine 2']
# plt.yticks([0, 1], labels=labels)
# plt.show()
# plt.clf()
# # plt.savefig('test.png')

# ## 

