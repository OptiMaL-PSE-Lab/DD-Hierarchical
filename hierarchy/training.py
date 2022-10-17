from time import perf_counter, sleep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow.keras as keras

df = pd.read_csv('Batch_Reactor_NN')
inputs = ['volume']
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


#Neural Network Arquitecture
nn_relu = keras.Sequential(name = 'Batch_Reactor_Relu')
nn_relu.add(keras.layers.Dense(units = 5, input_dim = len(inputs), activation = 'relu', use_bias = True))
nn_relu.add(keras.layers.Dense(units = 5, activation = 'relu', use_bias = True))
nn_relu.add(keras.layers.Dense(units = len(outputs)))
nn_relu.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])

nn_sigmoid = keras.Sequential(name = 'Batch_Reactor_Sigmoid')
nn_sigmoid.add(keras.layers.Dense(units = 5, input_dim = len(inputs), activation = 'sigmoid', use_bias = True))
nn_sigmoid.add(keras.layers.Dense(units = 5, activation = 'sigmoid', use_bias = True))
nn_sigmoid.add(keras.layers.Dense(units = len(outputs)))
nn_sigmoid.compile(loss = 'mse', optimizer= 'adam', metrics = ['mae'])

#Split DataSet 
X_train, X_test, y_train, y_test = train_test_split(dfin.values, dfout.values, test_size=0.15, random_state=8)

ep = 300
#Train Neural Network
start_relu = perf_counter()
history_relu = nn_relu.fit(X_train,y_train, epochs = ep, validation_split = 0.15, batch_size = 4)
end_relu = perf_counter()

# nn_relu.save('Batch_Reactor_NN_Relu_f2')
start_sigmoid = perf_counter()
history_sigmoid = nn_sigmoid.fit(X_train,y_train, epochs = ep, validation_split = 0.15, batch_size = 4)
end_sigmoid = perf_counter()

# nn_sigmoid.save('Batch_Reactor_NN_Sigmoid_f2')

#Evaluating in the test Set
y_pred_test_relu = nn_relu.predict(X_test) 
y_pred_test_sigmoid = nn_sigmoid.predict(X_test)
       
mse_sigmoid = mean_squared_error(y_test, y_pred_test_sigmoid)       
print('mse sigmoid in the test set %.6f' %mse_sigmoid) 
mse_relu = mean_squared_error(y_test, y_pred_test_relu)       
print('mse relu in the test set %.6f' %mse_relu)

# summarize history for loss
fig,ax= plt.subplots()
ax.plot(history_relu.history['loss'], label = 'Training ReLU NN', linestyle = '--', color = 'b')
ax.plot(history_relu.history['val_loss'],label = 'Validation ReLU NN',linestyle = '--', color = 'royalblue')
ax.plot(history_sigmoid.history['loss'], label = 'Training Sigmoid NN',linestyle = 'dashdot', color = 'r')
ax.plot(history_sigmoid.history['val_loss'],label = 'Validation Sigmoid NN',linestyle = 'dashdot', color = 'lightcoral')
ax.set_ylabel('Loss [MSE]')
ax.set_xlabel('Epochs')
ax.legend(loc='upper right')
ax.grid(alpha =0.3 ,ls ='--')
ax.set_ylim(0,None)
ax.set_xlim(0,ep)
ax.legend()
# fig.savefig('lerning_curves_CS1.png', dpi = 600,format = 'png',bbox_inches  = 'tight')  

def split_ouputs(Y):    
    tf= np.zeros((Y.shape[0]))
    Q = np.zeros((Y.shape[0]))        
    for i in range(Y.shape[0]):
        tf[i] = Y[i,0]
        Q[i]  = Y[i,1]
    return ([tf,Q])    
#Rescaling inputs
a = X_test.shape[0]
X_test = (X_test*x_factor['volume'] + x_offset['volume']).reshape(a,1)
tf_test = split_ouputs(y_test)[0]*y_factor['tf'] + y_offset['tf']
Q_test = split_ouputs(y_test)[1]*y_factor['utility'] + y_offset['utility']
tf_relu = split_ouputs(y_pred_test_relu)[0]*y_factor['tf'] + y_offset['tf']
Q_relu = split_ouputs(y_pred_test_relu)[1]*y_factor['utility'] + y_offset['utility']
tf_sigmoid = split_ouputs(y_pred_test_sigmoid)[0]*y_factor['tf'] + y_offset['tf']
Q_sigmoid = split_ouputs(y_pred_test_sigmoid)[1]*y_factor['utility'] + y_offset['utility']

#Plotting
fig1 = plt.figure(figsize =(7,5))
gs = fig1.add_gridspec(2, hspace = 0)
axs = gs.subplots(sharex = True)
msize = 7.5
alp = 0.65
m = 1
axs[0].plot(X_test,tf_test, label = 'Sampled Test Points', marker = 'x', color = 'r', linewidth = 0, ms = msize ,mew = 2) 
axs[0].plot(X_test, tf_relu, label = 'ReLU NN', marker = '^',color = 'b', linewidth = 0, alpha = alp, mew = m)
axs[0].plot(X_test, tf_sigmoid, label = 'Sigmoid NN', marker = 's',color = 'g', linewidth = 0, alpha = alp, mew = m)
axs[0].set_ylim(0,4)
axs[0].grid(alpha =0.3 ,ls ='--')
axs[0].set_ylabel('Time [h]')
axs[0].legend()

axs[1].plot(X_test,Q_test, label = 'Sampled Points', marker = 'x',color = 'r', linewidth = 0, ms = msize ,mew = 2)
axs[1].plot(X_test, Q_relu, label = 'ReLU NN', marker = '^',color = 'b', linewidth = 0, alpha = alp, mew = m)
axs[1].plot(X_test, Q_sigmoid, label = 'Sigmoid NN', marker = 's',color = 'g', linewidth = 0, alpha = alp, mew = m)
axs[1].set_xlabel('Volume $\\rm[m^3]$')
axs[1].set_ylim(None, 220)
axs[1].set_ylabel('Utility [uU]')
axs[1].grid(alpha =0.3 ,ls ='--')
axs[1].legend()
# fig1.savefig('test_set_CS1.png', dpi = 600,format = 'png',bbox_inches  = 'tight')  
print(start_relu-end_relu)
print(start_sigmoid-end_sigmoid)