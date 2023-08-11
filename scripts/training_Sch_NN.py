import warnings
import argparse
import time
import multiprocessing as mp

import json

# from time import perf_counter, sleep
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython.display import Image

import torch
import torch.nn as nn
# import torch.nn.functional as F
import lightgbm as lgb

from functools import partial

from omlt.io import write_onnx_model_with_bounds

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from torch.utils.data import DataLoader, TensorDataset

from onnxmltools.convert.lightgbm.convert import convert
from skl2onnx.common.data_types import FloatTensorType

# from omlt.io import (
#     write_onnx_model_with_bounds, 
#     load_onnx_neural_network,
# )

from data.planning.planning_sch_bilevel_lowdim import data, scheduling_data

class Net(nn.Module):
    def __init__(self, l1, l2):
        super(Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, 1),
        )
    def forward(self, x):
        # x = self.flatten(x)
        out = self.linear_relu_stack(x)
        # out = F.log_softmax(out)
        return out

class Net2(nn.Module):
    def __init__(self, l1, l2):
        super(Net2, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, 1),
        )
    def forward(self, x):
        # x = self.flatten(x)
        out = self.linear_relu_stack(x)
        # out = torch.sigmoid(out)
        return out

def moving_average(array, N_av=10):
    N = len(array)
    dummy = np.zeros(N)
    for i in range(N):
        dummy[i] = np.mean(array[i - min(9, i):i+1])
    return dummy

def save_NN(model, scaled_bounds, dir, train_sc, test_sc, train_t):
    x = torch.randn(32, 4)
    try:
        model.forward(x)
        torch.onnx.export(
            model,
            x,
            dir+'.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        )
        print(f"Wrote PyTorch model to {dir+'.onnx'}")
    except:
        model.forward(x)
        torch.onnx.export(
            model,
            x,
            '.'+dir+'.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }
        )
        print(f"Wrote PyTorch model to {'.'+dir+'.onnx'}")
    write_onnx_model_with_bounds(dir, None, scaled_bounds)

    save_obj = {
    'train_score': train_sc,
    'test_score': test_sc,
    'train_time': train_t,
    # 'opt_time_S': None,
    # 'opt_time_M': None,
    # 'opt_time_L': None, 
    }
    json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
    with open(dir+'.json', 'w') as f:
        json.dump(save_obj, f)
    

def classNN(data_model, EPOCHS, nodes1, nodes2):
    X_train, X_test, y_train, y_test = data_model
    model = Net2(nodes1,nodes2)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    training_loss = np.zeros(EPOCHS)
    training_ac = np.zeros(EPOCHS)
    test_ac = np.zeros(EPOCHS)
    test_loss = np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        count = 0
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            y_batch_pred = model(x_batch)
            loss = criterion(y_batch_pred, y_batch.view(*y_batch_pred.shape))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = np.round(torch.sigmoid(y_batch_pred).detach().numpy())
            target = np.round(y_batch.detach().numpy())
            training_ac[epoch] += np.sum(pred==target)
            training_loss[epoch] += loss.item()
        training_loss[epoch] = training_loss[epoch]/len(y_train)
        training_ac[epoch] = training_ac[epoch]/len(y_train)
        y_test_pred = model(torch.as_tensor(X_test, dtype=torch.float32))
        test_loss[epoch] += criterion(y_test_pred, torch.as_tensor(y_test, dtype=torch.float32).view(*y_test_pred.shape)).item()/len(y_test)
        pred = np.round(torch.sigmoid(y_test_pred).detach().numpy())
        target = np.round(y_test)
        test_ac[epoch] += np.sum(pred==target)/len(y_test)
        # if epoch % 50 == 0:
        #     print(f"Epoch {epoch} accuracy : Training: {training_ac[epoch]}, Test: {test_ac[epoch]}")
        #     print(f"Epoch {epoch} loss : Training: {training_loss[epoch]}, Test: {test_loss[epoch]}")
    y_test_pred = model(torch.as_tensor(X_test, dtype=torch.float32))
    y_train_pred = model(torch.as_tensor(X_train, dtype=torch.float32))
    test_accuracy = np.sum(np.round(torch.sigmoid(y_test_pred).detach().numpy())==np.round(y_test))/len(y_test)
    train_accuracy = np.sum(np.round(torch.sigmoid(y_train_pred).detach().numpy())==np.round(y_train))/len(y_train)
    # print(f"Training accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")

    return train_accuracy, test_accuracy, model

def regNN(data_model, EPOCHS, nodes1, nodes2):
    X_train, X_test, y_train, y_test = data_model
    # model = Net(5, 5)
    model = Net(nodes1, nodes2)
    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    #Split DataSet 
    dataset = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
    # test_dataset = TensorDataset(torch.as_tensort(X_test, dtype=torch.float32), torch.as_tensor(y_test, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    training_loss = np.zeros(EPOCHS)
    test_loss = np.zeros(EPOCHS)
    for epoch in range(EPOCHS):
        count = 0
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            y_batch_pred = model(x_batch)
            loss = loss_function(y_batch_pred, y_batch.view(*y_batch_pred.shape))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss[epoch] += loss.item()
        training_loss[epoch] = training_loss[epoch]/len(y_train)
        y_test_pred = model(torch.as_tensor(X_test, dtype=torch.float32))
        test_loss[epoch] += loss_function(y_test_pred, torch.as_tensor(y_test, dtype=torch.float32).view(*y_test_pred.shape)).item()/len(y_test)
        # if epoch % 50 == 0:
        #     print(f"Epoch {epoch} loss : Training: {training_loss[epoch]}, Test: {test_loss[epoch]}")
    # #Evaluating in the test Set
    X_in = torch.as_tensor(X_test, dtype=torch.float32)
    X_train_in = torch.as_tensor(X_train, dtype=torch.float32)
    y_pred_test = model.forward(X_in) 
    y_pred_train = model.forward(X_train_in)
    # plt.scatter(y_test, y_pred_test.detach().numpy(), c='r')
    # plt.scatter(y_train, y_pred_train.detach().numpy(), c='k')
    # plt.show()

    mse_relu_train = mean_squared_error(y_train, y_pred_train.detach().numpy())       
    print('mse relu in the training set %.6f' %mse_relu_train)
    mse_relu_test = mean_squared_error(y_test, y_pred_test.detach().numpy())       
    print('mse relu in the test set %.6f' %mse_relu_test)

    return mse_relu_train, mse_relu_test, model

def run_model(data_model, EPOCHS, nodes1, nodes2, model):
    dfin_scaled, dfout_scaled, dfout_class = data_model
    if model == 'RegNN':
        data_next = train_test_split(dfin_scaled.values, dfout_scaled.values, test_size=0.15, random_state=8, shuffle=True)
        res = regNN(data_next, EPOCHS, nodes1, nodes2)
    elif model == 'ClassNN': 
        data_next = train_test_split(dfin_scaled.values, dfout_class.values, test_size=0.15, random_state=8, shuffle=True) #### ClassNN <- used dfin rather than dfin_scaled?
        res = classNN(data_next, EPOCHS, nodes1, nodes2)
    return res

def main(args):

    prod_list = [p for p in data[None]['P'][None] if p in scheduling_data[None]['states'][None]]
    
    dataset = args.data

    if dataset == 'scheduling':
        try:
            dir = './data/scheduling/scheduling'
            df = pd.read_csv(dir)
        except:
            dir = '../data/scheduling/scheduling'
            df = pd.read_csv(dir)
    else:
        try:
            dir = './data/scheduling/scheduling_integr'
            df = pd.read_csv(dir)
        except:
            dir = '../data/scheduling/scheduling_integr'
            df = pd.read_csv(dir)

    inputs = [p for p in prod_list]
    outputs = ['cost']
    out_class = ['feas']

    dfin = df[inputs]
    dfout = df[outputs]
    dfout_class = df[out_class]


    lb = np.min(dfin.values, axis=0)
    ub = np.max(dfin.values, axis=0)
    input_bounds = [(l, u) for l, u in zip(lb, ub)]
    # print(input_bounds)

    #Scaling
    x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
    y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()
    # y_off_class, y_fac_class = dfout_class.mean().to_dict(), dfout_class.std().to_dict()

    dfin_scaled = (dfin - dfin.mean()).divide(dfin.std())
    dfout_scaled = (dfout - dfout.mean()).divide(dfout.std())
    # dfout_class = (dfout_class - dfout_class.mean()).divide(dfout_class.std())

    #Save the scaling parameters of the inputs for OMLT
    scaled_lb = dfin.min()[inputs].values
    scaled_ub = dfin.max()[inputs].values
    scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}

    # X_class, Y_class = dfin.values, dfout_class.values

    model_type = args.model
    EPOCHS = args.epochs
    SAVE = args.save
    nodes1 = args.nodes1
    nodes2 = args.nodes2
    data_model = dfin_scaled, dfout_scaled, dfout_class

    wrapper = partial(run_model, data_model, EPOCHS, nodes1, nodes2)

    if model_type != 'all':
        t0 = time.perf_counter()
        res = wrapper(model_type)
        t = time.perf_counter()
        print(f"{model_type}: {res} in {t-t0:.3f} seconds")

        if SAVE:
            dir = f"./results/Models/{dataset}_{model_type}_{nodes1}_{nodes2}"
            save_NN(res[2], scaled_input_bounds, dir, res[0], res[1], t-t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training")

    parser.add_argument("--model", type=str, default="RegNN", help = "Model to be trained - 'RegNN', 'ClassNN'")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--nodes1", type=int, default=50, help="Number of nodes in first hidden layer")
    parser.add_argument("--nodes2", type=int, default=20, help="Number of nodes in second hidden layer")
    parser.add_argument("--data", type=str, default="scheduling", help="Dataset - 'scheduling', 'integrated'")
    parser.add_argument("--save", type=bool, default=False, help="Flag indicating if model(s) should get saved")

    args = parser.parse_args()
    main(args)


# plt.scatter(y_test_cl, y_pred_test_cl, c='r')
# plt.scatter(y_train_cl, y_pred_train_cl, c='k')
# plt.show()


# def split_ouputs(Y):    
#     tf= np.zeros((Y.shape[0]))
#     Q = np.zeros((Y.shape[0]))        
#     for i in range(Y.shape[0]):
#         tf[i] = Y[i,0]
#         Q[i]  = Y[i,1]
#     return ([tf,Q])    


# #Rescaling inputs
# a = X_test.shape[0]
# X_test = (X_test*x_factor['conc_end'] + x_offset['conc_end']).reshape(a,1)
# tf_test = split_ouputs(y_test)[0]*y_factor['tf'] + y_offset['tf']
# Q_test = split_ouputs(y_test)[1]*y_factor['utility'] + y_offset['utility']
# tf_relu = split_ouputs(y_pred_test_relu)[0]*y_factor['tf'] + y_offset['tf']
# Q_relu = split_ouputs(y_pred_test_relu)[1]*y_factor['utility'] + y_offset['utility']

# a = X_train.shape[0]
# X_train = (X_train*x_factor['conc_end'] + x_offset['conc_end']).reshape(a,1)
# tf_train = split_ouputs(y_train)[0]*y_factor['tf'] + y_offset['tf']
# Q_train = split_ouputs(y_train)[1]*y_factor['utility'] + y_offset['utility']
# tf_relu_train = split_ouputs(y_pred_train_relu)[0]*y_factor['tf'] + y_offset['tf']
# Q_relu_train = split_ouputs(y_pred_train_relu)[1]*y_factor['utility'] + y_offset['utility']

# #Plotting
# fig1 = plt.figure(figsize =(7,5))
# gs = fig1.add_gridspec(2, hspace = 0)
# axs = gs.subplots(sharex = True)
# msize = 7.5
# alp = 0.65
# m = 1
# axs[0].plot(X_test,tf_test, label = 'Sampled Test Points', marker = 'x', color = 'r', linewidth = 0, ms = msize ,mew = 2) 
# axs[0].plot(X_test, tf_relu, label = 'ReLU NN', marker = '^',color = 'b', linewidth = 0, alpha = alp, mew = m)
# # axs[0].set_ylim(0,dfout.max()['tf'])
# axs[0].grid(alpha =0.3 ,ls ='--')
# axs[0].set_ylabel('Time [h]')
# axs[0].legend()

# axs[0].plot(X_train,tf_train, label = 'Sampled Training Points', marker = 'x', color = 'orange', linewidth = 0, ms = msize/2 ,mew = 2/2) 
# axs[0].plot(X_train, tf_relu_train, label = 'ReLU NN', marker = '^',color = 'c', linewidth = 0, alpha = alp, mew = m/2)
# # axs[0].set_ylim(0,dfout.max()['tf'])
# axs[0].grid(alpha =0.3 ,ls ='--')
# axs[0].set_ylabel('Time [h]')
# axs[0].legend()

# axs[1].plot(X_test,Q_test, label = 'Sampled Points', marker = 'x',color = 'r', linewidth = 0, ms = msize ,mew = 2)
# axs[1].plot(X_test, Q_relu, label = 'ReLU NN', marker = '^',color = 'b', linewidth = 0, alpha = alp, mew = m)
# # axs[1].set_xlabel('Concentration $\\rm[m^3]$')
# axs[1].set_xlabel('Production $\\rm[kg]$')
# # axs[1].set_ylim(None, dfout.max()['utility'])
# axs[1].set_ylabel('Utility [uU]')
# axs[1].grid(alpha =0.3 ,ls ='--')
# axs[1].legend()

# axs[1].plot(X_train,Q_train, label = 'Sampled Points', marker = 'x',color = 'orange', linewidth = 0, ms = msize/2 ,mew = 2/2)
# axs[1].plot(X_train, Q_relu_train, label = 'ReLU NN', marker = '^',color = 'c', linewidth = 0, alpha = alp, mew = m/2)
# axs[1].set_xlabel('Production $\\rm[kg]$')
# # axs[1].set_ylim(None, dfout.max()['utility'])
# axs[1].set_ylabel('Utility [uU]')
# axs[1].grid(alpha =0.3 ,ls ='--')
# axs[1].legend()


# # fig1.savefig('test_set_CS1.png', dpi = 600,format = 'png',bbox_inches  = 'tight')  

# plt.show()

# x = torch.randn(10, 4)
# f_before = model.forward(x)
# pytorch_model = None

# try:
#     dir = './results/Models/CS2_ReLU_'+mat+'.onnx'
#     torch.onnx.export(
#         model,
#         x,
#         dir,
#         input_names=['input'],
#         output_names=['output'],
#         dynamic_axes={
#             'input': {0: 'batch_size'},
#             'output': {0: 'batch_size'},
#         }
#     )
#     # torch.onnx.export(
#     #     model,
#     #     x,
#     #     dir,
#     #     input_names=['Volume'],
#     #     output_names=['Final Time', 'Utility'],
#     #     dynamic_axes={
#     #         'Volume': {0: 'batch_size'},
#     #         'Final Time': {0: 'batch_size'},
#     #         'Utility': {0: 'batch_size'}
#     #     }
#     # )
#     write_onnx_model_with_bounds(dir, None, scaled_input_bounds)
#     print(f"Wrote PyTorch model to {dir}")
# except:
#     dir = '../results/Models/CS2_ReLU_'+mat+'.onnx'
#     torch.onnx.export(
#         model,
#         x,
#         dir,
#         input_names=['conc_end'],
#         output_names=['Final Time', 'Utility'],
#         dynamic_axes={
#             'conc_end': {0: 'batch_size'},
#             'Final Time': {0: 'batch_size'},
#             'Utility': {0: 'batch_size'}
#         }
#     )
#     write_onnx_model_with_bounds(dir, None, scaled_input_bounds)
#     print(f"Wrote PyTorch model to {dir}")
