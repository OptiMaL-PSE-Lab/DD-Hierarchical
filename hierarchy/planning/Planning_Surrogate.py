# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:51:19 2021

@author: dv516
"""

### TO DO: BO + split into problems and benchmarks ###

import time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from functools import partial

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

import onnx
import torch
import onnxruntime as rt

from omlt import OmltBlock, OffsetScaling
from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
from omlt.io import load_onnx_neural_network

from omlt.neuralnet import ReluBigMFormulation
from omlt.gbt import GBTBigMFormulation, GradientBoostedTreeModel

# from data.planning.planning_extended import data as import_data
from data.planning.planning_sch_bilevel_lowdim import data, scheduling_data

prod_list = [p for p in data[None]['P'][None] if p in scheduling_data[None]['states'][None]]

try:
    dir = './data/scheduling/scheduling'
    df = pd.read_csv(dir) 
except:
    dir = '../data/scheduling/scheduling'
    df = pd.read_csv(dir) 

inputs = prod_list
outputs = ['cost']
out_cl = ['feas']

dfin = df[inputs]
dfout = df[outputs]
dfout_class = df[out_cl]

lb, ub = dfin.min()[inputs].values, dfin.max()[inputs].values
input_bounds = {i: (0, ub[i]) for i in range(len(inputs))}

#Scaling
x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()
y_offset_cl, y_factor_cl = dfout_class.mean().to_dict(), dfout_class.std().to_dict()
top = dfin.columns

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

scaler_cl = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: 0 for i in range(len(outputs))},
        factor_outputs={i: 1 for i in range(len(outputs))}
    )

# try:
#     dir = './results/Models/scheduling_RegNN.onnx'
#     onnx_model = onnx.load(dir)
# except:
#     dir = '../results/Models/scheduling_RegNN.onnx'
#     onnx_model = onnx.load(dir)
# net = load_onnx_neural_network(onnx_model, scaler, scaled_input_bounds)

try:
    dir = './results/Models/scheduling_RegTree.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/scheduling_RegTree.onnx'
    onnx_model = onnx.load(dir)
# tree = GradientBoostedTreeModel(onnx_model, scaler, scaled_input_bounds=scaled_input_bounds)
tree = GradientBoostedTreeModel(onnx_model, scaled_input_bounds=input_bounds)

try:
    dir = './results/Models/scheduling_ClassNN.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/scheduling_ClassNN.onnx'
    onnx_model = onnx.load(dir)
net_class = load_onnx_neural_network(onnx_model, scaler_cl, scaled_input_bounds)

try:
    dir = './results/Models/scheduling_ClassTree.onnx'
    onnx_model = onnx.load(dir)
except:
    dir = '../results/Models/scheduling_ClassTree.onnx'
    onnx_model = onnx.load(dir)
# tree_class = GradientBoostedTreeModel(onnx_model, scaler_cl, scaled_input_bounds=scaled_input_bounds)
tree_class = GradientBoostedTreeModel(onnx_model, scaled_input_bounds=input_bounds)


def state_to_control_t(t, N_t, T_set):
    dummy_array = np.arange(1, 1+N_t)
    N_total = len(T_set)
    T_min = T_set[0]
    idx = 1 + (t - T_min) // int(N_total/N_t)
    return min(idx, N_t)

def centralised_all(data, fix_TP=False, size='small', regression_type='NN', class_type=None, nodes=None, time_limit=None):

    N_t = data[None]['N_t'][None]
    to_Tc = lambda t: state_to_control_t(t, N_t, data[None]['T'][None])

    model = pyo.ConcreteModel()

    model.P = pyo.Set(initialize=data[None]['P'][None])
    model.L = pyo.Set(initialize=data[None]['L'][None])
    model.R = pyo.Set(initialize=data[None]['R'][None])
    model.T = pyo.Set(initialize=data[None]['T'][None])
    model.N_t = pyo.Param(initialize=data[None]['N_t'][None])
    model.Tc = pyo.Set(initialize=data[None]['Tc'][None])

    model.CP = pyo.Param(model.P, initialize=data[None]['CP'])
    model.U = pyo.Param(model.L, model.R, initialize=data[None]['U'])
    model.CT = pyo.Param(model.L, initialize=data[None]['CT'])
    model.CS = pyo.Param(model.P, initialize=data[None]['CS'])
    model.Price = pyo.Param(model.P, initialize=data[None]['Price'])
    model.LT = pyo.Param(model.L, initialize=data[None]['LT'])
    model.Q = pyo.Param(model.P, initialize=data[None]['Q'])
    model.A = pyo.Param(model.L, model.R, initialize=data[None]['A'])
    model.X = pyo.Param(model.L, model.P, initialize=data[None]['X'])

    model.SAIP0 = pyo.Param(initialize=data[None]['SAIP0'][None])
    model.SI0 = pyo.Param(initialize=data[None]['SI0'][None])
    model.SAIS0 = pyo.Param(model.L, initialize=data[None]['SAIS0'])
    model.S0 = pyo.Param(model.P, initialize=data[None]['S0'])
    model.IAIPstar0 = pyo.Param(initialize=data[None]['IAIPstar0'][None])
    model.IIstar0 = pyo.Param(initialize=data[None]['IIstar0'][None])
    model.IAISstar0 = pyo.Param(model.L, initialize=data[None]['IAISstar0'])
    model.Istar0 = pyo.Param(model.P, initialize=data[None]['Istar0'])
    model.CS_SAIS = pyo.Param(model.L, initialize=data[None]['CS_SAIS'])
    model.CS_AIP = pyo.Param(initialize=data[None]['CS_AIP'][None])
    model.CS_I = pyo.Param(initialize=data[None]['CS_I'][None])
    model.RM_Cost = pyo.Param(initialize=data[None]['RM_Cost'][None])
    model.AI_Cost = pyo.Param(initialize=data[None]['AI_Cost'][None])
    model.CP_I = pyo.Param(initialize=data[None]['CP_I'][None])
    model.CP_AI = pyo.Param(initialize=data[None]['CP_AI'][None])
    model.SP_AI = pyo.Param(initialize=data[None]['SP_AI'][None])

    model.F = pyo.Param(model.P, model.T, initialize=data[None]['F'])

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

    model.cost_adjust = pyo.Var(within = pyo.NonNegativeReals)

    model.outputs = pyo.Var(model.T, within = pyo.NonNegativeReals)

    if regression_type == 'NN':
        if nodes is not None:
            node1, node2 = nodes
            dir = f"./results/Models/scheduling_RegNN_{node1}_{node2}.onnx"
        else:
            dir = './results/Models/scheduling_RegNN.onnx'
        try:
            onnx_model = onnx.load(dir)
        except:
            onnx_model = onnx.load(dir)
        net = load_onnx_neural_network(onnx_model, scaler, scaled_input_bounds)
        formulation = ReluBigMFormulation(net)
    else:
        formulation = GBTBigMFormulation(tree)

    model.linking = pyo.ConstraintList()

    model.NN1 = OmltBlock()
    model.NN2 = OmltBlock()

    model.NN1.build_formulation(formulation) 
    model.NN2.build_formulation(formulation) 

    model.linking.add(model.Prod['PA','Asia',  1] == model.NN1.inputs[0])
    model.linking.add(model.Prod['PB','Asia',  1] == model.NN1.inputs[1])
    model.linking.add(model.Prod['TEE','Asia', 1] == model.NN1.inputs[2])
    model.linking.add(model.Prod['TGE','Asia', 1] == model.NN1.inputs[3])

    model.linking.add(model.Prod['PA','Asia',  2] == model.NN2.inputs[0])
    model.linking.add(model.Prod['PB','Asia',  2] == model.NN2.inputs[1])
    model.linking.add(model.Prod['TEE','Asia', 2] == model.NN2.inputs[2])
    model.linking.add(model.Prod['TGE','Asia', 2] == model.NN2.inputs[3])

    model.linking.add(model.outputs[1] == model.NN1.outputs[0])
    model.linking.add(model.outputs[2] == model.NN2.outputs[0])

    if size != 'small':

        model.NN3 = OmltBlock()
        model.NN4 = OmltBlock()
        model.NN5 = OmltBlock()
        model.NN6 = OmltBlock()

        model.NN3.build_formulation(formulation) 
        model.NN4.build_formulation(formulation) 
        model.NN5.build_formulation(formulation) 
        model.NN6.build_formulation(formulation) 

        model.linking.add(model.Prod['PA','Asia',  3] == model.NN3.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  3] == model.NN3.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 3] == model.NN3.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 3] == model.NN3.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  4] == model.NN4.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  4] == model.NN4.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 4] == model.NN4.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 4] == model.NN4.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  5] == model.NN5.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  5] == model.NN5.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 5] == model.NN5.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 5] == model.NN5.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  6] == model.NN6.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  6] == model.NN6.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 6] == model.NN6.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 6] == model.NN6.inputs[3])

        model.linking.add(model.outputs[3] == model.NN3.outputs[0])
        model.linking.add(model.outputs[4] == model.NN4.outputs[0])
        model.linking.add(model.outputs[5] == model.NN5.outputs[0])
        model.linking.add(model.outputs[6] == model.NN6.outputs[0])

    if size == 'large':

        model.NN7 = OmltBlock()
        model.NN8 = OmltBlock()
        model.NN9 = OmltBlock()
        model.NN10 = OmltBlock()
        model.NN11 = OmltBlock()
        model.NN12 = OmltBlock()

        model.NN7.build_formulation(formulation) 
        model.NN8.build_formulation(formulation) 
        model.NN9.build_formulation(formulation) 
        model.NN10.build_formulation(formulation)
        model.NN11.build_formulation(formulation)
        model.NN12.build_formulation(formulation)

        model.linking.add(model.Prod['PA','Asia',  7] == model.NN7.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  7] == model.NN7.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 7] == model.NN7.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 7] == model.NN7.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  8] == model.NN8.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  8] == model.NN8.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 8] == model.NN8.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 8] == model.NN8.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  9] == model.NN9.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  9] == model.NN9.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 9] == model.NN9.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 9] == model.NN9.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  10] == model.NN10.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  10] == model.NN10.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 10] == model.NN10.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 10] == model.NN10.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  11] == model.NN11.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  11] == model.NN11.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 11] == model.NN11.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 11] == model.NN11.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  12] == model.NN12.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  12] == model.NN12.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 12] == model.NN12.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 12] == model.NN12.inputs[3])

        model.linking.add(model.outputs[7] == model.NN7.outputs[0])
        model.linking.add(model.outputs[8] == model.NN8.outputs[0])
        model.linking.add(model.outputs[9] == model.NN9.outputs[0])
        model.linking.add(model.outputs[10] == model.NN10.outputs[0])
        model.linking.add(model.outputs[11] == model.NN11.outputs[0])
        model.linking.add(model.outputs[12] == model.NN12.outputs[0])

    model.linking.add(model.cost_adjust == sum(model.outputs[t] for t in model.T))


    if class_type == 'NN' or class_type == 'Tree':
        # model.feas1 = model.Var()

        model.class1 = OmltBlock()
        model.class2 = OmltBlock()

        if class_type == 'NN':
            formulation = ReluBigMFormulation(net_class)
        else:
            formulation = GBTBigMFormulation(tree_class)

        model.class1.build_formulation(formulation) 
        model.class2.build_formulation(formulation) 

        model.linking.add(model.Prod['PA','Asia',  1] == model.class1.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  1] == model.class1.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 1] == model.class1.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 1] == model.class1.inputs[3])

        model.linking.add(model.Prod['PA','Asia',  2] == model.class2.inputs[0])
        model.linking.add(model.Prod['PB','Asia',  2] == model.class2.inputs[1])
        model.linking.add(model.Prod['TEE','Asia', 2] == model.class2.inputs[2])
        model.linking.add(model.Prod['TGE','Asia', 2] == model.class2.inputs[3])

        if size != 'small':
        
            model.linking.add(model.Prod['PA','Asia',  3] == model.class3.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  3] == model.class3.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 3] == model.class3.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 3] == model.class3.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  4] == model.class4.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  4] == model.class4.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 4] == model.class4.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 4] == model.class4.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  5] == model.class5.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  5] == model.class5.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 5] == model.class5.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 5] == model.class5.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  6] == model.class6.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  6] == model.class6.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 6] == model.class6.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 6] == model.class6.inputs[3])

        if size == 'large':

            model.linking.add(model.Prod['PA','Asia',  7] == model.class7.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  7] == model.class7.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 7] == model.class7.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 7] == model.class7.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  8] == model.class8.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  8] == model.class8.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 8] == model.class8.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 8] == model.class8.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  9] == model.class9.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  9] == model.class9.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 9] == model.class9.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 9] == model.class9.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  10] == model.class10.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  10] == model.class10.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 10] == model.class10.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 10] == model.class10.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  11] == model.class11.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  11] == model.class11.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 11] == model.class11.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 11] == model.class11.inputs[3])

            model.linking.add(model.Prod['PA','Asia',  12] == model.class12.inputs[0])
            model.linking.add(model.Prod['PB','Asia',  12] == model.class12.inputs[1])
            model.linking.add(model.Prod['TEE','Asia', 12] == model.class12.inputs[2])
            model.linking.add(model.Prod['TGE','Asia', 12] == model.class12.inputs[3])

        # model.linking.add(model.feas == model.class1.outputs[0] + model.class2.outputs[0])



    if fix_TP:
        model.TP_fixed = pyo.Param(model.L, model.Tc)

        def constrain_TP(m, l, tc):
            return m.TP[l, tc] == m.TP_fixed[l, tc]
        model.h_TP = pyo.Constraint(model.L, model.Tc, rule=constrain_TP)

    def o(m):
        return (m.Cost_Central + m.cost_adjust) / 1e6
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
        return m.Cost_Central == prod_cost + transp_cost + store_cost - sales

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
        return m.Cost_As == prod_cost + store_cost + transp_cost - sales

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
        return m.Cost_Am == prod_cost + store_cost + transp_cost - sales

    model.AmCost = pyo.Constraint(rule=CoAm)

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

    ### 
    def sales_UL(m, p, t):
        return m.SA[p, t] <= m.F[p, t] ####
    ###

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

    # ins = model.create_instance(data)
    # set initial stuff

    # solver = pyo.SolverFactory("mosek")
    solver = pyo.SolverFactory("gurobi_direct")
    if time_limit is not None:
        solver.options['TimeLimit'] = time_limit
    solver.solve(model)

    return model

# t0 = time.perf_counter()
# Nt=5
# data[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
# res = centralised_all(data, regression_type = 'NN', class_type = 'NN')
# # res = centralised_all(data, size='medium', regression_type = 'NN', time_limit=60) # , nodes=(40,20)
# t = time.perf_counter()
# print(f"Objective: {pyo.value(res.obj)} in {(t - t0)/60} min")

# print([pyo.value(res.NN1.inputs[i]) for i in res.NN1.inputs], [pyo.value(res.Prod[p, 'Asia', 1]) for p in prod_list])
# print([pyo.value(res.NN2.inputs[i]) for i in res.NN2.inputs], [pyo.value(res.Prod[p, 'Asia', 2]) for p in prod_list])
# print(pyo.value(res.NN1.outputs[0]))
# print(pyo.value(res.NN2.outputs[0]))

# print(pyo.value(res.class1.outputs[0]))
# print(pyo.value(res.class2.outputs[0]))
# # print(pyo.value(res.class1.inputs[0]), pyo.value(res.class1.outputs[1]))
# # print(pyo.value(res.class2.inputs[0]), pyo.value(res.class2.outputs[1]))


# x_1 = np.array([[pyo.value(res.NN1.inputs[i]) for i in range(4)]]).astype(np.float32)
# x_2 = np.array([[pyo.value(res.NN2.inputs[i]) for i in range(4)]]).astype(np.float32)

# print(x_1, x_2)

# def scale(x, x_offset, x_factor, names):
#     out = x.copy()
#     for i, name in enumerate(names):
#         out[0][i] = (x[0][i] - x_offset[name])/x_factor[name]
#     return out

# def descale(y, y_offset, y_factor):
#     return y*y_factor + y_offset

# x1_scaled = scale(x_1, x_offset, x_factor, top)
# x2_scaled = scale(x_2, x_offset, x_factor, top)

# sess_tree = rt.InferenceSession("./results/Models/scheduling_RegTree.onnx")
# y_tree1 = sess_tree.run(None, {"float_input": x_1}) # x1_scaled
# y_tree2 = sess_tree.run(None, {"float_input": x_2})
# # y1_ = descale(y_tree1[0], y_offset['cost'], y_factor['cost'])
# # y2_ = descale(y_tree2[0], y_offset['cost'], y_factor['cost'])
# print(y_tree1) #, y1_)
# print(y_tree2) #, y2_)

# class_tree = rt.InferenceSession("./results/Models/scheduling_ClassTree.onnx")
# y_tree1_cl = class_tree.run(None, {"float_input": x_1})
# y_tree2_cl = class_tree.run(None, {"float_input": x_2})
# print(y_tree1_cl)
# print(y_tree2_cl)


# sess_NN = rt.InferenceSession("./results/Models/scheduling_RegNN.onnx")
# y_NN1 = sess_NN.run(None, {"input": x1_scaled})
# y_NN2 = sess_NN.run(None, {"input": x2_scaled})
# y1_ = descale(y_NN1[0], y_offset['cost'], y_factor['cost'])
# y2_ = descale(y_NN2[0], y_offset['cost'], y_factor['cost'])
# print(y_NN1, y1_)
# print(y_NN2, y2_)

# class_NN = rt.InferenceSession("./results/Models/scheduling_ClassNN.onnx")
# y_NN1_class = class_NN.run(None, {"input": x1_scaled})
# y_NN2_class = class_NN.run(None, {"input": x2_scaled})
# print(y_NN1_class)
# print(y_NN2_class)



