from time import perf_counter, sleep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from omlt.io import (
    write_onnx_model_with_bounds, 
    load_onnx_neural_network,
)

class PyTorchModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense_0 = nn.Linear(input_dim, 5)
        self.dense_1 = nn.Linear(5, 5)
        self.out = nn.Linear(5, output_dim)

    def forward(self, x):
        x = F.relu(self.dense_0(x))
        x = F.relu(self.dense_1(x))
        x = self.out(x)
        return x

mat = 'TI'

try:
    dir = './data/CS2_sampling/Batch_Reactor_NN_'+mat
    df = pd.read_csv(dir)
except:
    dir = './data/CS2_sampling/Batch_Reactor_NN_'+mat
    df = pd.read_csv(dir)

EPOCHS = 300

inputs = ['conc_end']
outputs = ['tf','utility']

dfin = df[inputs]
dfout = df[outputs]


lb = np.min(dfin.values, axis=0)
ub = np.max(dfin.values, axis=0)
input_bounds = [(l, u) for l, u in zip(lb, ub)]
print(input_bounds)

#Scaling
x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()

dfin = (dfin - dfin.mean()).divide(dfin.std())
dfout = (dfout - dfout.mean()).divide(dfout.std())

#Save the scaling parameters of the inputs for OMLT
scaled_lb = dfin.min()[inputs].values
scaled_ub = dfin.max()[inputs].values
scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}

X, Y = dfin.values, dfout.values

# #Split DataSet 
# X_train, X_test, y_train, y_test = train_test_split(dfin.values, dfout.values, test_size=0.15, random_state=8)

model = PyTorchModel(len(inputs), len(outputs))
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=10)

for epoch in range(EPOCHS):
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):
        y_batch_pred = model(x_batch)
        loss = loss_function(y_batch_pred, y_batch.view(*y_batch_pred.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch number: {epoch} loss : {loss.item()}")

### TensorFlow

# #Neural Network Arquitecture
# nn_relu = keras.Sequential(name = 'Batch_Reactor_Relu')
# nn_relu.add(keras.layers.Dense(units = 5, input_dim = len(inputs), activation = 'relu', use_bias = True))
# nn_relu.add(keras.layers.Dense(units = 5, activation = 'relu', use_bias = True))
# nn_relu.add(keras.layers.Dense(units = len(outputs)))
# nn_relu.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])

# nn_sigmoid = keras.Sequential(name = 'Batch_Reactor_Sigmoid')
# nn_sigmoid.add(keras.layers.Dense(units = 5, input_dim = len(inputs), activation = 'sigmoid', use_bias = True))
# nn_sigmoid.add(keras.layers.Dense(units = 5, activation = 'sigmoid', use_bias = True))
# nn_sigmoid.add(keras.layers.Dense(units = len(outputs)))
# nn_sigmoid.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])

#Train Neural Network
# start_relu = perf_counter()
# history_relu = nn_relu.fit(X_train,y_train, epochs = ep, validation_split = 0.15, batch_size = 4)
# end_relu = perf_counter()

# # nn_relu.save('Batch_Reactor_NN_Relu_f2')
# start_sigmoid = perf_counter()
# history_sigmoid = nn_sigmoid.fit(X_train,y_train, epochs = ep, validation_split = 0.15, batch_size = 4)
# end_sigmoid = perf_counter()

# nn_sigmoid.save('Batch_Reactor_NN_Sigmoid_f2')

###

#Split DataSet 
X_train, X_test, y_train, y_test = train_test_split(dfin.values, dfout.values, test_size=0.15, random_state=8)
X_in = torch.as_tensor(X_test, dtype=torch.float32)
X_train_in = torch.as_tensor(X_train, dtype=torch.float32)

#Evaluating in the test Set

y_pred_test_relu = model.forward(X_in) 
y_pred_train_relu = model.forward(X_train_in)
       
# mse_sigmoid = mean_squared_error(y_test, y_pred_test_sigmoid)       
# print('mse sigmoid in the test set %.6f' %mse_sigmoid) 
# mse_relu = mean_squared_error(y_test, y_pred_test_relu)       
# print('mse relu in the test set %.6f' %mse_relu)

# # summarize history for loss
# fig,ax= plt.subplots()
# ax.plot(history_relu.history['loss'], label = 'Training ReLU NN', linestyle = '--', color = 'b')
# ax.plot(history_relu.history['val_loss'],label = 'Validation ReLU NN',linestyle = '--', color = 'royalblue')
# ax.plot(history_sigmoid.history['loss'], label = 'Training Sigmoid NN',linestyle = 'dashdot', color = 'r')
# ax.plot(history_sigmoid.history['val_loss'],label = 'Validation Sigmoid NN',linestyle = 'dashdot', color = 'lightcoral')
# ax.set_ylabel('Loss [MSE]')
# ax.set_xlabel('Epochs')
# ax.legend(loc='upper right')
# ax.grid(alpha =0.3 ,ls ='--')
# ax.set_ylim(0,None)
# ax.set_xlim(0,ep)
# ax.legend()
# # fig.savefig('lerning_curves_CS1.png', dpi = 600,format = 'png',bbox_inches  = 'tight')  

def split_ouputs(Y):    
    tf= np.zeros((Y.shape[0]))
    Q = np.zeros((Y.shape[0]))        
    for i in range(Y.shape[0]):
        tf[i] = Y[i,0]
        Q[i]  = Y[i,1]
    return ([tf,Q])    

#Rescaling inputs
a = X_test.shape[0]
X_test = (X_test*x_factor['conc_end'] + x_offset['conc_end']).reshape(a,1)
tf_test = split_ouputs(y_test)[0]*y_factor['tf'] + y_offset['tf']
Q_test = split_ouputs(y_test)[1]*y_factor['utility'] + y_offset['utility']
tf_relu = split_ouputs(y_pred_test_relu)[0]*y_factor['tf'] + y_offset['tf']
Q_relu = split_ouputs(y_pred_test_relu)[1]*y_factor['utility'] + y_offset['utility']

a = X_train.shape[0]
X_train = (X_train*x_factor['conc_end'] + x_offset['conc_end']).reshape(a,1)
tf_train = split_ouputs(y_train)[0]*y_factor['tf'] + y_offset['tf']
Q_train = split_ouputs(y_train)[1]*y_factor['utility'] + y_offset['utility']
tf_relu_train = split_ouputs(y_pred_train_relu)[0]*y_factor['tf'] + y_offset['tf']
Q_relu_train = split_ouputs(y_pred_train_relu)[1]*y_factor['utility'] + y_offset['utility']

#Plotting
fig1 = plt.figure(figsize =(7,5))
gs = fig1.add_gridspec(2, hspace = 0)
axs = gs.subplots(sharex = True)
msize = 7.5
alp = 0.65
m = 1
axs[0].plot(X_test,tf_test, label = 'Sampled Test Points', marker = 'x', color = 'r', linewidth = 0, ms = msize ,mew = 2) 
axs[0].plot(X_test, tf_relu, label = 'ReLU NN', marker = '^',color = 'b', linewidth = 0, alpha = alp, mew = m)
# axs[0].set_ylim(0,dfout.max()['tf'])
axs[0].grid(alpha =0.3 ,ls ='--')
axs[0].set_ylabel('Time [h]')
axs[0].legend()

axs[0].plot(X_train,tf_train, label = 'Sampled Training Points', marker = 'x', color = 'orange', linewidth = 0, ms = msize/2 ,mew = 2/2) 
axs[0].plot(X_train, tf_relu_train, label = 'ReLU NN', marker = '^',color = 'c', linewidth = 0, alpha = alp, mew = m/2)
# axs[0].set_ylim(0,dfout.max()['tf'])
axs[0].grid(alpha =0.3 ,ls ='--')
axs[0].set_ylabel('Time [h]')
axs[0].legend()

axs[1].plot(X_test,Q_test, label = 'Sampled Points', marker = 'x',color = 'r', linewidth = 0, ms = msize ,mew = 2)
axs[1].plot(X_test, Q_relu, label = 'ReLU NN', marker = '^',color = 'b', linewidth = 0, alpha = alp, mew = m)
# axs[1].set_xlabel('Concentration $\\rm[m^3]$')
axs[1].set_xlabel('Production $\\rm[kg]$')
# axs[1].set_ylim(None, dfout.max()['utility'])
axs[1].set_ylabel('Utility [uU]')
axs[1].grid(alpha =0.3 ,ls ='--')
axs[1].legend()

axs[1].plot(X_train,Q_train, label = 'Sampled Points', marker = 'x',color = 'orange', linewidth = 0, ms = msize/2 ,mew = 2/2)
axs[1].plot(X_train, Q_relu_train, label = 'ReLU NN', marker = '^',color = 'c', linewidth = 0, alpha = alp, mew = m/2)
axs[1].set_xlabel('Production $\\rm[kg]$')
# axs[1].set_ylim(None, dfout.max()['utility'])
axs[1].set_ylabel('Utility [uU]')
axs[1].grid(alpha =0.3 ,ls ='--')
axs[1].legend()


# fig1.savefig('test_set_CS1.png', dpi = 600,format = 'png',bbox_inches  = 'tight')  

plt.show()

x = torch.randn(10, 1)
f_before = model.forward(x)
pytorch_model = None

try:
    dir = './results/Models/CS2_ReLU_'+mat+'.onnx'
    torch.onnx.export(
        model,
        x,
        dir,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    # torch.onnx.export(
    #     model,
    #     x,
    #     dir,
    #     input_names=['Volume'],
    #     output_names=['Final Time', 'Utility'],
    #     dynamic_axes={
    #         'Volume': {0: 'batch_size'},
    #         'Final Time': {0: 'batch_size'},
    #         'Utility': {0: 'batch_size'}
    #     }
    # )
    write_onnx_model_with_bounds(dir, None, scaled_input_bounds)
    print(f"Wrote PyTorch model to {dir}")
except:
    dir = '../results/Models/CS2_ReLU_'+mat+'.onnx'
    torch.onnx.export(
        model,
        x,
        dir,
        input_names=['conc_end'],
        output_names=['Final Time', 'Utility'],
        dynamic_axes={
            'conc_end': {0: 'batch_size'},
            'Final Time': {0: 'batch_size'},
            'Utility': {0: 'batch_size'}
        }
    )
    write_onnx_model_with_bounds(dir, None, scaled_input_bounds)
    print(f"Wrote PyTorch model to {dir}")
