import warnings
import argparse
# from time import perf_counter, sleep
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import torch
import torch.nn as nn
# import torch.nn.functional as F
import lightgbm as lgb
from torch.utils.data import DataLoader, TensorDataset

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

def main(args):

    prod_list = [p for p in data[None]['P'][None] if p in scheduling_data[None]['states'][None]]

    try:
        dir = './data/scheduling/scheduling'
        df = pd.read_csv(dir)
    except:
        dir = '../data/scheduling/scheduling'
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
    print(input_bounds)

    #Scaling
    x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
    y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()
    # y_off_class, y_fac_class = dfout_class.mean().to_dict(), dfout_class.std().to_dict()

    dfin = (dfin - dfin.mean()).divide(dfin.std())
    dfout = (dfout - dfout.mean()).divide(dfout.std())
    # dfout_class = (dfout_class - dfout_class.mean()).divide(dfout_class.std())

    #Save the scaling parameters of the inputs for OMLT
    scaled_lb = dfin.min()[inputs].values
    scaled_ub = dfin.max()[inputs].values
    scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}

    # X_class, Y_class = dfin.values, dfout_class.values

    model_type = args.model
    EPOCHS = args.epochs

    if model_type == "ClassNN" or model_type == "all":
        #Split DataSet 
        X_train, X_test, y_train, y_test = train_test_split(dfin.values, dfout_class.values, test_size=0.15, random_state=8)

        model = Net2(50,20)
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

            if epoch % 50 == 0:
                print(f"Epoch {epoch} accuracy : Training: {training_ac[epoch]}, Test: {test_ac[epoch]}")
                print(f"Epoch {epoch} loss : Training: {training_loss[epoch]}, Test: {test_loss[epoch]}")

        y_test_pred = model(torch.as_tensor(X_test, dtype=torch.float32))
        y_train_pred = model(torch.as_tensor(X_train, dtype=torch.float32))
        test_accuracy = np.sum(np.round(torch.sigmoid(y_test_pred).detach().numpy())==np.round(y_test))/len(y_test)
        train_accuracy = np.sum(np.round(torch.sigmoid(y_train_pred).detach().numpy())==np.round(y_train))/len(y_train)
        print(f"Training accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")


    if model_type == "RegNN" or model_type == "all":

        X, Y = dfin.values, dfout.values

        # model = Net(5, 5)
        model = Net(50, 20)
        loss_function = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
        # optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

        #Split DataSet 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=8, shuffle=True)

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

            if epoch % 50 == 0:
                print(f"Epoch {epoch} loss : Training: {training_loss[epoch]}, Test: {test_loss[epoch]}")

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

    # plt.plot(range(EPOCHS), training_loss, '--k', ms = 3)
    # plt.plot(range(EPOCHS), moving_average(training_loss), '-k')
    # plt.plot(range(EPOCHS), moving_average(test_loss), '-r')
    # plt.plot(range(EPOCHS), test_loss, '--r', ms = 3)
    # plt.yscale('log')
    # plt.show()

    if model_type == "RegTree" or model_type == "all":

        X, Y = dfin.values, dfout.values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=8, shuffle=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PARAMS = {
                'objective': 'regression',
                'metric': 'mse',
                'boosting': 'gbdt',
                'num_trees': 50,
                'max_depth': 3,
                'min_data_in_leaf': 2,
                'random_state': 100,
                'verbose': -1
                }
            train_data = lgb.Dataset(X_train, 
                                     label=y_train,
                                     params={'verbose': -1})

            model = lgb.train(PARAMS, 
                              train_data,
                              verbose_eval=False)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        print(f"LGB regression MSE: Training: {mse_train}, Testing: {mse_test}")

        # plt.scatter(y_test, y_pred_test, c='r')
        # plt.scatter(y_train, y_pred_train, c='k')
        # plt.show()

    if model_type == "ClassTree" or model_type == "all":

        X, Y_cl = dfin.values, dfout_class.values
        X_train, X_test, y_train_cl, y_test_cl = train_test_split(X, Y_cl, test_size=0.15, random_state=8, shuffle=True)


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            PARAMS = {
                'objective': 'binary',
                'metric': 'binary_logloss',
            'boosting': 'gbdt',
            'num_trees': 50,
            'max_depth': 3,
            'min_data_in_leaf': 2,
            'random_state': 100,
            'verbose': -1
            }
            
            train_data = lgb.Dataset(X_train, 
                                 label=y_train_cl,
                                 params={'verbose': -1})

            model = lgb.train(PARAMS, 
                          train_data,
                          verbose_eval=False)

        y_pred_train_cl = model.predict(X_train)
        y_pred_test_cl = model.predict(X_test)
        y_pred_test_cl1 =  np.round(y_pred_test_cl)
        ac_train = accuracy_score(y_train_cl, np.round(y_pred_train_cl))
        ac_test = accuracy_score(y_test_cl, np.round(y_pred_test_cl))
        print(f"LGB classification accuracy: Training: {ac_train}, Testing: {ac_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training")

    parser.add_argument("--model", type=str, default="all", help = "Model to be trained - 'all', 'RegNN', 'ClassNN', 'RegTree', 'ClassTree'")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")

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
