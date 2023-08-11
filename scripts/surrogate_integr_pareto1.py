import time
import argparse
import json


import numpy as np
import pyomo.environ as pyo

import matplotlib.pyplot as plt

from openbox import Optimizer
from openbox import space as sp

from training_Sch_NN import main as training
from hierarchy.DFO_ext_all_integr import wrapper, wrapper_bi, wrapper_tri
from hierarchy.planning.Planning_Integrated import centralised_all
from data.planning.planning_sch_bilevel_lowdim import scheduling_data, data

def pyo2vec(res):
    
    x = [res.Prod['PA','Asia',t].value for t in res.T]
    x += [res.Prod['PB','Asia',t].value for t in res.T]
    x += [res.Prod['TEE','Asia',t].value for t in res.T]
    x += [res.Prod['TGE','Asia',t].value for t in res.T]
    x += [res.Prod['PC','America',t].value for t in res.T]
    x += [res.Prod['PD','America',t].value for t in res.T]
    
    x += [res.SA['PA',t].value for t in res.T]
    x += [res.SA['PB',t].value for t in res.T]
    x += [res.SA['TEE',t].value for t in res.T]
    x += [res.SA['TGE',t].value for t in res.T]
    x += [res.SA['PC',t].value for t in res.T]
    x += [res.SA['PD',t].value for t in res.T]

    x += [res.PI[t].value for t in res.T]
    x += [res.PAI[t].value for t in res.T]
    x += [res.TP['Asia', tc].value for tc in res.Tc]
    x += [res.TP['America', tc].value for tc in res.Tc]

    return x


def parse_args():
    parser = argparse.ArgumentParser(description="Run training")

    parser.add_argument("--model", type=str, default="RegNN", help = "Model to be trained - 'RegNN', 'ClassNN'")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--nodes1", type=int, default=50, help="Number of nodes in first hidden layer")
    parser.add_argument("--nodes2", type=int, default=20, help="Number of nodes in second hidden layer")
    parser.add_argument("--data", type=str, default="scheduling", help="Dataset - 'scheduling', 'integrated'")
    parser.add_argument("--save", type=bool, default=False, help="Flag indicating if model(s) should get saved")

    return parser


# for n1 in nodes1:
#     for n2 in nodes2:
#         parser = parse_args()
#         args = parser.parse_args(['--nodes1', str(n1), '--nodes2', str(n2), '--save', 'True', '--data', 'integrated'])
#         training(args)

#         t0 = time.perf_counter()
#         Nt=5
#         data[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
#         res = centralised_all(data, regression_type = 'NN', nodes=(n1,n2), time_limit=60)
#         t = time.perf_counter()
#         print(f"Objective: {pyo.value(res.obj)} in {(t - t0)/60} min")

#         dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
#         with open(dir, 'r+') as file:
#             file_data = json.load(file)
#             file_data["opt_time_s"] = t - t0
#             x = pyo2vec(res)
#             file_data["x"] = x

#             real = {
#                 'upper': {'time': None, 'obj': None},
#                 'bi': {'time': None, 'obj': None},
#                 'tri': {'time': None, 'obj': None},
#             }

#             t0 = time.perf_counter()
#             obj = wrapper(x, data, 1000)
#             t = time.perf_counter()
#             real['upper']['time'] = t - t0
#             real['upper']['obj'] = obj

#             t0 = time.perf_counter()
#             obj = wrapper_bi(x, data, 1000)
#             t = time.perf_counter()
#             real['bi']['time'] = t - t0
#             real['bi']['obj'] = obj

#             t0 = time.perf_counter()
#             obj = wrapper_tri(x, data, 1000)
#             t = time.perf_counter()
#             real['tri']['time'] = t - t0
#             real['tri']['obj'] = obj
#             file_data["hierarchy"] = real

#         with open(dir, 'w') as f:
#             json.dump(file_data, f)

#             ## add x for comparison with other method


def small(config):
    n1, n2 = config['nodes1'], config['nodes2']

    try:
        dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
        with open(dir, 'r+') as file:
            file_data = json.load(file)
    except:
        parser = parse_args()
        args = parser.parse_args(['--nodes1', str(n1), '--nodes2', str(n2), '--save', 'True', '--data', 'integrated'])
        training(args)

    t0 = time.perf_counter()
    Nt=5
    data[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
    res = centralised_all(data, regression_type = 'NN', nodes=(n1,n2), time_limit=60)
    t = time.perf_counter()
    print(f"Objective: {pyo.value(res.obj)} in {(t - t0)/60} min")

    dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
    with open(dir, 'r+') as file:
        file_data = json.load(file)
        file_data["opt_time_s"] = t - t0
        x = pyo2vec(res)
        file_data["x"] = x
        real = {
            'upper': {'time': None, 'obj': None},
            'bi': {'time': None, 'obj': None},
            'tri': {'time': None, 'obj': None},
        }
        t0 = time.perf_counter()
        obj = wrapper(x, data, 1000)
        t = time.perf_counter()
        real['upper']['time'] = t - t0
        real['upper']['obj'] = obj
        t0 = time.perf_counter()
        obj = wrapper_bi(x, data, 1000)
        t = time.perf_counter()
        real['bi']['time'] = t - t0
        real['bi']['obj'] = obj
        t0 = time.perf_counter()
        obj = wrapper_tri(x, data, 1000)
        t = time.perf_counter()
        real['tri']['time'] = t - t0
        real['tri']['obj'] = obj
        file_data["hierarchy"] = real
    with open(dir, 'w') as f:
        json.dump(file_data, f)

    return {'objectives': [file_data['opt_time_s'], file_data['hierarchy']['tri']['obj']]}



def medium(config):
    n1, n2 = config['nodes1'], config['nodes2']

    try:
        dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
        with open(dir, 'r+') as file:
            file_data = json.load(file)
    except:
        parser = parse_args()
        args = parser.parse_args(['--nodes1', str(n1), '--nodes2', str(n2), '--save', 'True', '--data', 'integrated'])
        training(args)

    t0 = time.perf_counter()
    Nt=5
    data[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
    res = centralised_all(data, size='medium', regression_type = 'NN', nodes=(n1,n2), time_limit=600)
    t = time.perf_counter()
    print(f"Objective: {pyo.value(res.obj)} in {(t - t0)/60} min")

    dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
    with open(dir, 'r+') as file:
        file_data = json.load(file)
        file_data["opt_time_m"] = t - t0
        x = pyo2vec(res)
        file_data["x"] = x
        real = {
            'upper': {'time': None, 'obj': None},
            'bi': {'time': None, 'obj': None},
            'tri': {'time': None, 'obj': None},
        }
        t0 = time.perf_counter()
        obj = wrapper(x, data, 1000)
        t = time.perf_counter()
        real['upper']['time'] = t - t0
        real['upper']['obj'] = obj
        t0 = time.perf_counter()
        obj = wrapper_bi(x, data, 1000)
        t = time.perf_counter()
        real['bi']['time'] = t - t0
        real['bi']['obj'] = obj
        t0 = time.perf_counter()
        obj = wrapper_tri(x, data, 1000)
        t = time.perf_counter()
        real['tri']['time'] = t - t0
        real['tri']['obj'] = obj
        file_data["medium"] = real
    with open(dir, 'w') as f:
        json.dump(file_data, f)

    return {'objectives': [file_data['opt_time_m'], file_data['medium']['tri']['obj']]}


def large(config):
    n1, n2 = config['nodes1'], config['nodes2']

    try:
        dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
        with open(dir, 'r+') as file:
            file_data = json.load(file)
    except:
        parser = parse_args()
        args = parser.parse_args(['--nodes1', str(n1), '--nodes2', str(n2), '--save', 'True', '--data', 'integrated'])
        training(args)

    t0 = time.perf_counter()
    Nt=5
    data[None].update({'N_t': {None: Nt}, 'Tc': {None: np.arange(1, 1+Nt)}})
    res = centralised_all(data, size='large', regression_type = 'NN', nodes=(n1,n2), time_limit=7200)
    t = time.perf_counter()
    print(f"Objective: {pyo.value(res.obj)} in {(t - t0)/60} min")

    dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
    with open(dir, 'r+') as file:
        file_data = json.load(file)
        file_data["opt_time_l"] = t - t0
        x = pyo2vec(res)
        file_data["x"] = x
        real = {
            'upper': {'time': None, 'obj': None},
            'bi': {'time': None, 'obj': None},
            'tri': {'time': None, 'obj': None},
        }
        t0 = time.perf_counter()
        obj = wrapper(x, data, 1000)
        t = time.perf_counter()
        real['upper']['time'] = t - t0
        real['upper']['obj'] = obj
        t0 = time.perf_counter()
        obj = wrapper_bi(x, data, 1000)
        t = time.perf_counter()
        real['bi']['time'] = t - t0
        real['bi']['obj'] = obj
        t0 = time.perf_counter()
        obj = wrapper_tri(x, data, 1000)
        t = time.perf_counter()
        real['tri']['time'] = t - t0
        real['tri']['obj'] = obj
        file_data["large"] = real
    with open(dir, 'w') as f:
        json.dump(file_data, f)

    return {'objectives': [file_data['opt_time_l'], file_data['large']['tri']['obj']]}



def main():
    

    nodes1 = [10, 20, 30, 40, 60, 80]
    nodes2 = [1, 5, 10, 20, 30, 40]

    json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')

    # nodes_dict = {}
    # for n1 in nodes1:
    #     for n2 in nodes2:
    #         dir = f"./results/Models/integrated_RegNN_{n1}_{n2}.json"
    #         with open(dir, 'r+') as file:
    #             file_data = json.load(file)
    #         nodes_dict[(n1,n2)] = {
    #             'test_score': file_data['test_score'],
    #             'opt_time_s': file_data['opt_time_s'],
    #             'hierarchy_s': file_data['hierarchy']['bi']['obj'],
    #         }

    # dominant_nodes3 = []
    # for nodes in nodes_dict:
    #     dummy = True
    #     for nodes2 in nodes_dict:
    #         if (nodes_dict[nodes2]['test_score'] < nodes_dict[nodes]['test_score']) and (nodes_dict[nodes2]['opt_time_s'] < nodes_dict[nodes]['opt_time_s']) and (nodes_dict[nodes2]['hierarchy_s'] < nodes_dict[nodes]['hierarchy_s']):
    #         # if (nodes_dict[nodes2]['opt_time_s'] < nodes_dict[nodes]['opt_time_s']) and (nodes_dict[nodes2]['hierarchy_s'] < nodes_dict[nodes]['hierarchy_s']):
    #             dummy = False
    #             break
    #     if dummy:
    #         dominant_nodes3 += [nodes]

    # print(dominant_nodes3, len(dominant_nodes3))

    # dominant_nodes2 = []
    # for nodes in nodes_dict:
    #     dummy = True
    #     for nodes2 in nodes_dict:
    #         # if (nodes_dict[nodes2]['test_score'] < nodes_dict[nodes]['test_score']) and (nodes_dict[nodes2]['opt_time_s'] < nodes_dict[nodes]['opt_time_s']) and (nodes_dict[nodes2]['hierarchy_s'] < nodes_dict[nodes]['hierarchy_s']):
    #         if (nodes_dict[nodes2]['opt_time_s'] < nodes_dict[nodes]['opt_time_s']) and (nodes_dict[nodes2]['hierarchy_s'] < nodes_dict[nodes]['hierarchy_s']):
    #             dummy = False
    #             break
    #     if dummy:
    #         dominant_nodes2 += [nodes]

    # print(dominant_nodes2, len(dominant_nodes2))

    # def f(x):
    #     config = {'nodes1': int(x[0]), 'nodes2': int(x[1])}
    #     try:
    #         return small(config)['objectives'][1]
    #     except:
    #         return 0
    
    # soln = pybobyqa.solve(
    #     f, [40, 20], bounds=[[5, 80], [1, 40]], 
    #     maxfun=50, objfun_has_noise=True,
    #     seek_global_minimum=True,
    #     scaling_within_bounds=True,
    #     print_progress=True
    # )
    # print(soln)

    # funcMulti = Function()
    # opt = opt_entmoot(
    #     funcMulti.get_bounds(),
    #     num_obj=2,
    #     n_initial_points=10,
    #     acq_optimizer='sampling',
    #     random_state=100
    # )

    # # main BO loop that derives pareto-optimal points
    # for i in range(40):
    #     next_x = opt.ask()
    #     next_y = funcMulti(next_x)
    #     opt.tell(next_x,next_y)
    #     print('Iteration'+str(i), next_x, next_y)


    space = sp.Space()
    nodes1 = sp.Int("nodes1", 5, 80)
    nodes2 = sp.Int("nodes2", 1, 40)
    space.add_variables([nodes1, nodes2])

    opt = Optimizer(
        small,
        space,
        num_objectives=2,
        max_runs=50,
        surrogate_type='gp',
        acq_optimizer_type='random_scipy',
        ref_point=[12, -2.95],
        time_limit_per_trial=400,
    )
    # opt = Optimizer(
    #     small,
    #     space,
    #     num_objectives=2,
    #     max_runs=50,
    #     surrogate_type='prf',
    #     acq_optimizer_type='local_random',
    #     ref_point=[12, -2.95],
    #     time_limit_per_trial=400,
    # )
    history = opt.run()
    # print(history)

    history = opt.get_history()
    try:
        print(history)
        text_file = open("results/Optima/surrogates_integr_small_opt.txt", "w")
        n = text_file.write(str(history))
        text_file.close()
    except Exception as e:
        print('Failed to save optima: ', e)

    history.plot_pareto_front()
    plt.show()
    plt.savefig('results/Figures/pareto_integr_small.svg')

    # history.plot_hypervolumes(logy=True)
    # plt.show()
    # plt.savefig('results/Figures/hypervolume_small.svg')

    # space = sp.Space()
    # nodes1 = sp.Int("nodes1", 5, 80)
    # nodes2 = sp.Int("nodes2", 1, 40)
    # space.add_variables([nodes1, nodes2])

    # opt = Optimizer(
    #     medium,
    #     space,
    #     num_objectives=2,
    #     max_runs=20,
    #     surrogate_type='gp',
    #     acq_optimizer_type='random_scipy',
    #     ref_point=[905, 5],
    #     time_limit_per_trial=900,
    # )
    # # opt = Optimizer(
    # #     small,
    # #     space,
    # #     num_objectives=2,
    # #     max_runs=20,
    # #     surrogate_type='prf',
    # #     acq_optimizer_type='local_random',
    # #     ref_point=[905, 5],
    # #     time_limit_per_trial=900,
    # # )
    # history = opt.run()
    # # print(history)
    
    # history.plot_pareto_front()
    # plt.show()
    # plt.savefig('results/Figures/pareto_medium1.svg')

    # history.plot_hypervolumes(logy=True)
    # plt.show()
    # plt.savefig('results/Figures/hypervolume_medium1.svg')

    # history = opt.get_history()
    # try:
    #     print(history)
    # except:
    #     print("Can't plot history for some reason")

    # history.plot_pareto_front()
    # plt.show()
    # plt.savefig('results/Figures/pareto_medium2.svg')

    # history.plot_hypervolumes(logy=True)
    # plt.show()
    # plt.savefig('results/Figures/hypervolume_medium2.svg')

    # space = sp.Space()
    # nodes1 = sp.Int("nodes1", 5, 80)
    # nodes2 = sp.Int("nodes2", 1, 40)
    # space.add_variables([nodes1, nodes2])

    # opt = Optimizer(
    #     large,
    #     space,
    #     num_objectives=2,
    #     max_runs=12,
    #     surrogate_type='gp',
    #     acq_optimizer_type='random_scipy',
    #     ref_point=[8000, 5],
    #     time_limit_per_trial=8100,
    # )
    # # opt = Optimizer(
    # #     small,
    # #     space,
    # #     num_objectives=2,
    # #     max_runs=20,
    # #     surrogate_type='prf',
    # #     acq_optimizer_type='local_random',
    # #     ref_point=[905, 5],
    # #     time_limit_per_trial=900,
    # # )
    # history = opt.run()
    # # print(history)
    
    # history.plot_pareto_front()
    # plt.show()
    # plt.savefig('results/Figures/pareto_integr_large1.svg')

    # history.plot_hypervolumes(logy=True)
    # plt.show()
    # plt.savefig('results/Figures/hypervolume_integr_large1.svg')

    # history = opt.get_history()
    # try:
    #     print(history)
    #     text_file = open("results/Optima/surrogates_integr_large_opt.txt", "w")
    #     n = text_file.write(str(history))
    #     text_file.close()
    # except Exception as e:
    #     print("Can't plot history for some reason: ", e)

    # history.plot_pareto_front()
    # plt.show()
    # plt.savefig('results/Figures/pareto_integr_large2.svg')

    # history.plot_hypervolumes(logy=True)
    # plt.show()
    # plt.savefig('results/Figures/hypervolume_integr_large2.svg')


if __name__=='__main__':
    main()

