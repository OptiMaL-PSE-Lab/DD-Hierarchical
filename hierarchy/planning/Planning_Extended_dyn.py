# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 18:51:19 2021

@author: dv516
"""

### TO DO: BO + split into problems and benchmarks ###

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

# from data.planning.planning_extended import data as import_data
from data.planning.planning_sch_bilevel import data as import_data

def state_to_control_t(t, N_t, T_set):
    dummy_array = np.arange(1, 1+N_t)
    N_total = len(T_set)
    T_min = T_set[0]
    idx = 1 + (t - T_min) // int(N_total/N_t)
    return min(idx, N_t)


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

    ins = model.create_instance(data)
    # set initial stuff

    # solver = pyo.SolverFactory("mosek")
    solver = pyo.SolverFactory("gurobi_direct")
    solver.solve(ins)

    return ins

# Nt=6
# import_data[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
# res = centralised(import_data)
# print(pyo.value(res.obj))

def Europe(data, arg="Augmented"):

    N_t = data[None]['N_t'][None]
    to_Tc = lambda t: state_to_control_t(t, N_t, data[None]['T'][None])

    model = pyo.AbstractModel()
    model.Tc = pyo.Set()

    model.I = pyo.Set()
    model.z_I = pyo.Set()
    model.N_t = pyo.Param()
    model.x = pyo.Var(model.I)
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)

    model.P = pyo.Set()
    model.L = pyo.Set()
    model.R = pyo.Set()
    model.T = pyo.Set()

    model.CP = pyo.Param(model.P)
    model.U = pyo.Param(model.L, model.R)
    model.CT = pyo.Param(model.L)
    model.CS = pyo.Param(model.P)
    model.Price = pyo.Param(model.P)
    model.LT = pyo.Param(model.L)
    model.Q = pyo.Param(model.P)
    model.A = pyo.Param(model.L, model.R)
    model.X = pyo.Param(model.L, model.P)

    model.SAIP0 = pyo.Param()
    model.SI0 = pyo.Param()
    model.IAIPstar0 = pyo.Param()
    model.IIstar0 = pyo.Param()

    model.CS_AIP = pyo.Param()
    model.CS_I = pyo.Param()
    model.AI_Cost = pyo.Param()
    model.CP_I = pyo.Param()
    model.CP_AI = pyo.Param()
    model.SP_AI = pyo.Param()

    # model.TP = pyo.Var(model.L, within=pyo.NonNegativeReals)
    # model.TP_dual = pyo.Param(model.L, within=pyo.NonNegativeReals)
    model.rho = pyo.Param()

    model.PAI = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.PI = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.SAIP = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.SI = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.Cost_Eu = pyo.Var()
    if arg == "Augmented":

        def o(m):
            return m.Cost_Eu / 1e6 + m.rho * sum(
                (m.x[i] - m.z[i] ) ** 2 for i in m.z_I
            )

        model.obj = pyo.Objective(rule=o)
    else:

        def o_(m):
            return m.Cost_Eu / 1e6

        model.obj = pyo.Objective(rule=o_)

        def fix_TP(m, i):
            return m.z[i] == m.x[i]

        model.TP_constr = pyo.Constraint(model.z_I, rule=fix_TP)

    def CoEu(m):
        prod_cost = sum(m.CP_I * m.PI[t] for t in m.T) + sum(
            m.CP_AI * m.PAI[t] for t in m.T
        )
        store_cost = sum(m.CS_AIP * m.SAIP[t] for t in m.T) + sum(
            m.CS_I * m.SI[t] for t in m.T
        )
        sales = sum(m.SP_AI * (m.x[to_Tc(t)]+m.x[N_t+to_Tc(t)]) for t in m.T)
        return m.Cost_Eu == prod_cost + store_cost - sales

    model.EuCost = pyo.Constraint(rule=CoEu)

    def invAIP_t(m, t):
        if t > 1:
            return m.SAIP[t] == m.SAIP[t - 1] + m.PAI[t] - (m.x[to_Tc(t)] + m.x[to_Tc(t)+N_t]) ### here
        else:
            return pyo.Constraint.Skip

    def invAIP_0(m):
        return m.SAIP[1] == m.SAIP0 + m.PAI[1] - (m.x[1] + m.x[1+N_t]) ### here

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

    # Sales equals forecast

    def safeSI(m, t):
        if t > 4:
            return m.SI[t] >= m.IIstar0
        else:
            # return m.SI[t] >= 0
            return m.SI[t] >= m.IIstar0 / 4

    def safeSAIP(m, t):
        if t > 4:
            return m.SAIP[t] >= m.IAIPstar0
        else:
            # return m.SAIP[t] >= 0
            return m.SAIP[t] >= m.IAIPstar0 / 4

    model.safeII = pyo.Constraint(model.T, rule=safeSI)
    model.safeIAIP = pyo.Constraint(model.T, rule=safeSAIP)

    ins = model.create_instance(data)

    # set initial stuff

    # solver = pyo.SolverFactory("mosek")
    solver = pyo.SolverFactory("gurobi_direct")
    res = solver.solve(ins)

    return ins
    # return ins, res


def Asia(data, arg="Augmented"):

    N_t = data[None]['N_t'][None]

    to_Tc = lambda t: state_to_control_t(t, N_t, data[None]['T'][None])

    model = pyo.AbstractModel()
    model.Tc = pyo.Set()

    model.I = pyo.Set()
    model.z_I = pyo.Set()
    model.x = pyo.Var(model.I)
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)

    model.P = pyo.Set()
    model.L = pyo.Set()
    model.R = pyo.Set()
    model.T = pyo.Set()

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

    model.SAIS0 = pyo.Param(model.L)
    model.S0 = pyo.Param(model.P)
    model.IAISstar0 = pyo.Param(model.L)
    model.Istar0 = pyo.Param(model.P)
    model.CS_SAIS = pyo.Param(model.L)
    model.SP_AI = pyo.Param()

    model.F = pyo.Param(model.P, model.T)

    model.Prod = pyo.Var(model.P, model.L, model.T, within=pyo.NonNegativeReals)

    model.S = pyo.Var(model.P, model.T, within=pyo.NonNegativeReals)
    model.SAIS = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)

    # model.TP = pyo.Var(model.L, within=pyo.NonNegativeReals)
    # model.TP_dual = pyo.Param(model.L, within=pyo.NonNegativeReals)
    model.rho = pyo.Param()

    model.SA = pyo.Var(model.P, model.T, within=pyo.NonNegativeReals)
    model.Cost_As = pyo.Var()
    model.Cost_Central = pyo.Var()
    if arg == "Augmented":

        def o(m):
            return m.Cost_As / 1e6 + m.rho * sum((m.x[i] - m.z[i] ) ** 2 for i in m.z_I) 

        model.obj = pyo.Objective(rule=o)
    else:

        def o_(m):
            return m.Cost_As / 1e6

        model.obj = pyo.Objective(rule=o_)

        def fix_TP(m, i):
            return m.x[i] == m.z[i]

        model.TP_constr = pyo.Constraint(model.z_I, rule=fix_TP)

    def CoAs(m):
        prod_cost = sum(m.CP[p] * m.Prod[p, "Asia", t] for p in m.P for t in m.T)
        transp_cost = sum(m.CT["Asia"] * m.x[to_Tc(t)] for t in m.T) ###
        store_cost = (
            sum(m.CS_SAIS["Asia"] * m.SAIS["Asia", t] for t in m.T)
            + sum(m.CS["PA"] * m.S["PA", t] for t in m.T)
            + sum(m.CS["TEE"] * m.S["TEE", t] for t in m.T)
            + sum(m.CS["PB"] * m.S["PB", t] for t in m.T)
            + sum(m.CS["TGE"] * m.S["TGE", t] for t in m.T)
        )
        sales = (
            sum(m.Price["PA"] * m.SA["PA", t] for t in m.T)
            + sum(m.Price["TEE"] * m.SA["TEE", t] for t in m.T)
            + sum(m.Price["PB"] * m.SA["PB", t] for t in m.T)
            + sum(m.Price["TGE"] * m.SA["TGE", t] for t in m.T)
            - sum(m.SP_AI * m.x[to_Tc(t)] for t in m.T) ### here
        )
        return m.Cost_As == prod_cost + store_cost + transp_cost - sales

    model.AsCost = pyo.Constraint(rule=CoAs)

    def resource_constraint(m, r, t):
        return (
            m.U["Asia", r]
            * sum(
                m.Prod[p, "Asia", t] * m.Q[p] for p in m.P
            )
            <= m.A["Asia", r]
        )

    model.res_constr = pyo.Constraint(model.R, model.T, rule=resource_constraint)

    def inv_PA_t(m, t):
        if t > 1:
            return (
                m.S["PA", t]
                == m.S["PA", t - 1] - m.SA["PA", t] + m.Prod["PA", "Asia", t]
            )
        else:
            return pyo.Constraint.Skip

    def inv_PA_0(m):
        return m.S["PA", 1] == m.S0["PA"] - m.SA["PA", 1] + m.Prod["PA", "Asia", 1]

    model.S_PAt = pyo.Constraint(model.T, rule=inv_PA_t)
    model.S_PA0 = pyo.Constraint(rule=inv_PA_0)

    def inv_TEE_t(m, t):
        if t > 1:
            return (
                m.S["TEE", t]
                == m.S["TEE", t - 1] - m.SA["TEE", t] + m.Prod["TEE", "Asia", t]
            )
        else:
            return pyo.Constraint.Skip

    def inv_TEE_0(m):
        return m.S["TEE", 1] == m.S0["TEE"] - m.SA["TEE", 1] + m.Prod["TEE", "Asia", 1]

    model.S_TEEt = pyo.Constraint(model.T, rule=inv_TEE_t)
    model.S_TEE0 = pyo.Constraint(rule=inv_TEE_0)

    def inv_PB_t(m, t):
        if t > 1:
            return (
                m.S["PB", t]
                == m.S["PB", t - 1] - m.SA["PB", t] + m.Prod["PB", "Asia", t]
            )
        else:
            return pyo.Constraint.Skip

    def inv_PB_0(m):
        return m.S["PB", 1] == m.S0["PB"] - m.SA["PB", 1] + m.Prod["PB", "Asia", 1]

    model.S_PBt = pyo.Constraint(model.T, rule=inv_PB_t)
    model.S_PB0 = pyo.Constraint(rule=inv_PB_0)

    def inv_TGE_t(m, t):
        if t > 1:
            return (
                m.S["TGE", t]
                == m.S["TGE", t - 1] - m.SA["TGE", t] + m.Prod["TGE", "Asia", t]
            )
        else:
            return pyo.Constraint.Skip

    def inv_TGE_0(m):
        return m.S["TGE", 1] == m.S0["TGE"] - m.SA["TGE", 1] + m.Prod["TGE", "Asia", 1]

    model.S_TGEt = pyo.Constraint(model.T, rule=inv_TGE_t)
    model.S_TGE0 = pyo.Constraint(rule=inv_TGE_0)


    def invAIS_t(m, t):
        t_TP = t - m.LT['Asia']
        if t > 1:
            if t_TP > 0:
                return m.SAIS["Asia", t] == m.SAIS["Asia", t - 1] + m.x[to_Tc(t_TP)] - 1.1 * sum(m.Prod[p, "Asia", t] * m.Q[p] for p in m.P) ### here
            else:
                return m.SAIS["Asia", t] == m.SAIS["Asia", t - 1] - 1.1 * sum(m.Prod[p, "Asia", t] * m.Q[p] for p in m.P)
        else:
            return pyo.Constraint.Skip

    def invAIS_0(m): ### here
        return m.SAIS["Asia", 1] == m.SAIS0["Asia"] - 1.1 * sum(
            m.Prod[p, "Asia", 1] * m.Q[p] for p in m.P
        )

    model.SAIS_t = pyo.Constraint(model.T, rule=invAIS_t)
    model.SAIS_0 = pyo.Constraint(rule=invAIS_0)

    def Prod_UL(m, p, t):
        return m.Prod[p, "Asia", t] <= 500e3 * m.X["Asia", p]

    model.P_UL = pyo.Constraint(model.P, model.T, rule=Prod_UL)

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

    def safeSAIS(m, t):
        if t > 4:
            return m.SAIS["Asia", t] >= m.IAISstar0["Asia"]
        else:
            # return m.SAIS[l, t] >= 0
            return m.SAIS["Asia", t] >= m.IAISstar0["Asia"] / 4

    model.safeI = pyo.Constraint(model.P, model.T, rule=safeS)
    model.safeIAIS = pyo.Constraint(model.T, rule=safeSAIS)

    ins = model.create_instance(data)

    # set initial stuff

    # solver = pyo.SolverFactory('mosek')
    solver = pyo.SolverFactory("gurobi_direct")
    res = solver.solve(ins)

    return ins
    # return ins, res


def America(data, arg="Augmented"):

    N_t = data[None]['N_t'][None]

    to_Tc = lambda t: state_to_control_t(t, N_t, data[None]['T'][None])

    model = pyo.AbstractModel()
    model.Tc = pyo.Set()

    model.I = pyo.Set()
    model.z_I = pyo.Set()
    model.x = pyo.Var(model.I)
    model.z = pyo.Param(model.z_I)
    model.u = pyo.Param(model.z_I)

    model.P = pyo.Set()
    model.L = pyo.Set()
    model.R = pyo.Set()
    model.T = pyo.Set()

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

    model.SAIS0 = pyo.Param(model.L)
    model.S0 = pyo.Param(model.P)
    model.IAISstar0 = pyo.Param(model.L)
    model.Istar0 = pyo.Param(model.P)
    model.CS_SAIS = pyo.Param(model.L)
    model.SP_AI = pyo.Param()

    model.F = pyo.Param(model.P, model.T)

    model.Prod = pyo.Var(model.P, model.L, model.T, within=pyo.NonNegativeReals)

    model.S = pyo.Var(model.P, model.T, within=pyo.NonNegativeReals)
    model.SAIS = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)

    # model.TP_dual = pyo.Param(model.L, within=pyo.NonNegativeReals)
    # model.TP = pyo.Var(model.L, within=pyo.NonNegativeReals)
    model.rho = pyo.Param()

    model.SA = pyo.Var(model.P, model.T, within=pyo.NonNegativeReals)
    model.Cost_Am = pyo.Var()

    if arg == "Augmented":

        def o(m):
            return (
                m.Cost_Am / 1e6 + m.rho * sum((m.x[i] - m.z[i]  ) ** 2 for i in m.z_I)
            )

        model.obj = pyo.Objective(rule=o)
    else:

        def o_(m):
            return m.Cost_Am / 1e6

        model.obj = pyo.Objective(rule=o_)

        def fix_TP(m, i):
            return m.x[i] == m.z[i]

        model.TP_constr = pyo.Constraint(model.z_I, rule=fix_TP)

    def CoAm(m):
        prod_cost = sum(m.CP[p] * m.Prod[p, "America", t] for p in m.P for t in m.T)
        transp_cost = sum(m.CT["America"] * m.x[N_t+to_Tc(t)] for t in m.T) ### here
        store_cost = sum(m.CS_SAIS["America"] * m.SAIS["America", t] for t in m.T) + \
            sum(m.CS["PC"] * m.S["PC", t] for t in m.T) + \
            sum(m.CS["PD"] * m.S["PD", t] for t in m.T)
        sales = sum(m.Price["PC"] * m.SA["PC", t] for t in m.T) + sum(m.Price["PD"] * m.SA["PD", t] for t in m.T) - sum(
            m.SP_AI * m.x[N_t+to_Tc(t)] for t in m.T ### here
        )
        return m.Cost_Am == prod_cost + store_cost + transp_cost - sales

    model.AmCost = pyo.Constraint(rule=CoAm)

    def resource_constraint(m, r, t):
        return (
            m.U["America", r]
            * sum(
                m.Prod[p, "America", t]*m.Q[p] for p in m.P
            )
            <= m.A["America", r]
        )

    model.res_constr = pyo.Constraint(model.R, model.T, rule=resource_constraint)

    def inv_PC_t(m, t):
        if t > 1:
            return (
                m.S["PC", t]
                == m.S["PC", t - 1] - m.SA["PC", t] + m.Prod["PC", "America", t]
            )
        else:
            return pyo.Constraint.Skip

    def inv_PC_0(m):
        return m.S["PC", 1] == m.S0["PC"] - m.SA["PC", 1] + m.Prod["PC", "America", 1]

    model.S_PCt = pyo.Constraint(model.T, rule=inv_PC_t)
    model.S_PC0 = pyo.Constraint(rule=inv_PC_0)

    def inv_PD_t(m, t):
        if t > 1:
            return (
                m.S["PD", t]
                == m.S["PD", t - 1] - m.SA["PD", t] + m.Prod["PD", "America", t]
            )
        else:
            return pyo.Constraint.Skip

    def inv_PD_0(m):
        return m.S["PD", 1] == m.S0["PD"] - m.SA["PD", 1] + m.Prod["PD", "America", 1]

    model.S_PDt = pyo.Constraint(model.T, rule=inv_PD_t)
    model.S_PD0 = pyo.Constraint(rule=inv_PD_0)

    def invAIS_t(m, t):
        t_TP = t - m.LT['America']
        if t > 1:
            if t_TP >= 0:
                return m.SAIS["America", t] == m.SAIS["America", t - 1] + m.x[N_t+to_Tc(t_TP)] - 1.1 * sum(m.Prod[p, "America", t] * m.Q[p] for p in m.P) ### here
            else:
                return m.SAIS["America", t] == m.SAIS["America", t - 1] - 1.1 * sum(m.Prod[p, "America", t] * m.Q[p] for p in m.P) ### here
        else:
            return pyo.Constraint.Skip

    def invAIS_0(m):
        return m.SAIS["America", 1] == m.SAIS0["America"] - 1.1 * sum( ### here
            m.Prod[p, "America", 1] * m.Q[p] for p in m.P
        )

    model.SAIS_t = pyo.Constraint(model.T, rule=invAIS_t)
    model.SAIS_0 = pyo.Constraint(rule=invAIS_0)

    def Prod_UL(m, p, t):
        return m.Prod[p, "America", t] <= 500e3 * m.X["America", p]

    model.P_UL = pyo.Constraint(model.P, model.T, rule=Prod_UL)

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

    def safeSAIS(m, t):
        if t > 4:
            return m.SAIS["America", t] >= m.IAISstar0["America"]
        else:
            # return m.SAIS[l, t] >= 0
            return m.SAIS["America", t] >= m.IAISstar0["America"] / 4

    model.safeI = pyo.Constraint(model.P, model.T, rule=safeS)
    model.safeIAIS = pyo.Constraint(model.T, rule=safeSAIS)

    ins = model.create_instance(data)

    # set initial stuff

    # solver = pyo.SolverFactory("mosek")
    solver = pyo.SolverFactory("gurobi_direct")
    res = solver.solve(ins)

    return ins
    # return ins, res

def f_Europe(data, z_list, rho, global_ind, index, u_list = None, solver = False):
    
    temp_dict = {
        'z': {}, 'u': {},
        'z_I': {None: global_ind},
        'rho': {None: rho},
        'I': {None: index},
    }

    N_t = int(len(global_ind)/2)
    data[None].update(temp_dict)
    data[None].update({'N_t': {None: N_t}, 'Tc': {None: np.arange(1, 1+N_t)}})

    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0
    
    return Europe(data)


def f_Asia(data, z_list, rho, global_ind, index, u_list = None, solver = False):
    
    temp_dict = {
        'z': {}, 'u': {},
        'z_I': {None: global_ind},
        'rho': {None: rho},
        'I': {None: index},
    }
    
    N_t = int(len(global_ind)/2)
    data[None].update(temp_dict)
    data[None].update({'N_t': {None: N_t}, 'Tc': {None: np.arange(1, 1+N_t)}})

    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0
    
    return Asia(data)

def f_America(data, z_list, rho, global_ind, index, u_list = None, solver = False):
    
    temp_dict = {
        'z': {}, 'u': {},
        'z_I': {None: global_ind},
        'rho': {None: rho},
        'I': {None: index},
    }
    
    N_t = int(len(global_ind)/2)
    data[None].update(temp_dict)
    data[None].update({'N_t': {None: N_t}, 'Tc': {None: np.arange(1, 1+N_t)}})

    for idx in global_ind:
        data[None]['z'][idx] = z_list[idx][-1]
        if u_list is not None:
            data[None]['u'][idx] = u_list[idx][-1]
        else:
            data[None]['u'][idx] = 0
    
    return America(data)

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
                Demand[p][t - 1] = Forecast[p][t - 1]

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
    
    Dummy_Sales = {}
    Dummy_Sales["PA"] = np.zeros(N_t)
    Dummy_Sales["PB"] = np.zeros(N_t)
    Dummy_Sales["PC"] = np.zeros(N_t)
    Dummy_Sales["PD"] = np.zeros(N_t)
    Dummy_Sales["TEE"] = np.zeros(N_t)
    Dummy_Sales["TGE"] = np.zeros(N_t)

    for p in P_set:
        Storage[p][0] = max(0,data[None]["S0"][p] - min(Sales[p][0], Demand[p][0]) + Production[p][0])
        Dummy_Sales[p][0] =  min(Sales[p][0], Demand[p][0], data[None]["S0"][p]+Production[p][0])
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
                    Storage[p][t - 2] - min(Sales[p][t-1], Demand[p][t - 1]) + Production[p][t - 1]
                )
                Dummy_Sales[p][t-1] =  min(Sales[p][t-1], Demand[p][t-1], Storage[p][t - 2] +Production[p][0])
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
    return Storage, Demand, Dummy_Sales

list_fi = [
    partial(f_Europe, import_data), 
    partial(f_Asia, import_data), 
    partial(f_America, import_data),
    ]

# z_arr = [pyo.value(res.TP["Asia", t]) for t in res.Tc] + [pyo.value(res.TP["America", t]) for t in res.Tc]
# global_ind = [i+1 for i in range(len(z_arr))]
# z_list = {j: [z_arr[j-1]] for j in global_ind}
# results = [f(z_list, 1e6, global_ind, global_ind) for f in list_fi]
# objectives = [pyo.value(r.obj) for r in results]
# print(objectives)
# Asia = results[1]

# for i in Asia.I:
#     print(pyo.value(Asia.x[i]), pyo.value(Asia.z[i]), pyo.value(results[2].z[i]))
# for r in res.R:
#     for t in res.T: 
#         print(pyo.value(Asia.U["Asia", r]*sum(Asia.Prod[p, "Asia", t] * Asia.Q[p] for p in res.P)) <= pyo.value(Asia.A["Asia", r]))
#         print(pyo.value(res.U["Asia", r]*sum(res.Prod[p, "Asia", t] * res.Q[p] for p in res.P)) <= pyo.value(res.A["Asia", r]))
#         print("")


# for p in res.P:
#     print('Asia', pyo.value(p), [[(pyo.value(Asia.S[p, t]), pyo.value(res.S[p, t]))  for t in res.T]])
#     print('Asia', [[(pyo.value(Asia.Prod[p, "Asia", t]), pyo.value(res.Prod[p, "Asia", t]))  for t in res.T]])
#     print("")

# for p in res.P:
#     print('America', pyo.value(p), [[(pyo.value(results[2].S[p, t]), pyo.value(res.S[p, t]))  for t in res.T]])
#     print('America', [[(pyo.value(results[2].Prod[p, "America", t]), pyo.value(res.Prod[p, "America", t]))  for t in res.T]])
#     print("")

# print(pyo.value(res.Cost_Eu), pyo.value(res.Cost_As), pyo.value(res.Cost_Am))
# print((pyo.value(res.Cost_Eu) + pyo.value(res.Cost_As) + pyo.value(res.Cost_Am))/1e6, pyo.value(res.obj))
