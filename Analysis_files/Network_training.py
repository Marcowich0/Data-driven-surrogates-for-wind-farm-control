# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:12:02 2022

@author: Marc
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import csv
from tensorflow.keras.callbacks import TensorBoard
import datetime
import tensorflow as tf
import math

# =============================================================================
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
# tf.debugging.set_log_device_placement(False)
# =============================================================================


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def weibullmedian(a,k):
    return (a*math.log(2)**(1/k))
    #return k
    #return a*math.gamma(1+1/k)

Directory_data = r'C:\Users\marc0\Desktop\Bachelor\Machinelearning\\'

neuronnumber = 10
epochnumber = 2000
plt.rcParams.update({'font.size': 12})



##############################################################################
#                                                                            #
#                                   Power                                    #
#                                                                            #
##############################################################################

################################File loading##################################
inputslist = []
outputlist = []
with open(Directory_data+"TrainingDataInputs") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        inputslist.append(row)
        
with open(Directory_data+"TrainingDataoutputs") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        outputlist.append(row)


inputs  = np.array(inputslist)
outputs = np.array(outputlist)
inputs, outputs = unison_shuffled_copies(inputs, outputs)

train_index       =  np.int(0.6*np.size(inputs[:,0]))
validation_index  =  np.int(0.8*np.size(inputs[:,0]))

################################Data#####################################
scaler   = MinMaxScaler(feature_range=(0,1))
x_scaled = scaler.fit_transform(inputs)
y_scaled = scaler.fit_transform(outputs)
x_train = x_scaled[:train_index,:]
y_train = y_scaled[:train_index,:]
x_validation = x_scaled[train_index:validation_index,:]
y_validation = y_scaled[train_index:validation_index,:]
x_test = x_scaled[validation_index:,:]
y_test = y_scaled[validation_index:,:]

power_test_in = inputs[validation_index:,:]
power_test_out = outputs[validation_index:,:]
power_x_test = x_test

#############################Model#######################################
K.clear_session()
model= Sequential()
model.add(Dense(neuronnumber, input_dim = x_train.shape[1], kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(neuronnumber, activation='relu'))
model.add(Dense(2))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
tbGraph = TensorBoard(log_dir = r'C:\Users\marc0\Desktop\Bachelor\Machinelearning\logs\{now}',
                      histogram_freq=64, write_graph=True, write_images=True)
history=model.fit(x_train, y_train, epochs=epochnumber,
                  batch_size=16, 
                  verbose=2, 
                  validation_data=(x_validation,y_validation),
                  callbacks=[tbGraph])

print(history.history.keys())


###########################Plots#########################################

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

plt.rcParams.update({'font.size': 22})
fig, ax = plt.subplots(2, 2, figsize=(26,16))
ax[0, 0].hist(inputs[:train_index,0], bins=7, color='gray',edgecolor='black', alpha=0.2)
ax[0, 0].hist(inputs[train_index:validation_index,0], bins=7, color='red', alpha=.7)
ax[0, 0].hist(inputs[validation_index:,0], bins=7, alpha=.5, edgecolor='black', color='yellow')
ax[0, 0].set_title("Yaw 1")
ax[0, 0].set_xlabel("Yaw angle")
ax[0, 0].set_ylabel("amount")
ax[0, 1].hist(inputs[:train_index,1], bins=10, color='gray',edgecolor='black', alpha=0.2)
ax[0, 1].hist(inputs[train_index:validation_index,1], bins=10, color='red', alpha=.7)
ax[0, 1].hist(inputs[validation_index:,1], bins=10, alpha=.5, edgecolor='black', color='yellow')
ax[0, 1].set_title("Omega")
ax[0, 1].set_xlabel("Rotational speed")
ax[0, 1].set_ylabel("amount")
ax[1, 0].hist(inputs[:train_index,2]/100, bins=10, color='gray',edgecolor='black', alpha=0.2)
ax[1, 0].hist(inputs[train_index:validation_index,2]/100, bins=10, color='red', alpha=.7)
ax[1, 0].hist(inputs[validation_index:,2]/100, bins=10, alpha=.5, edgecolor='black', color='yellow')
ax[1, 0].set_title("Pitch")
ax[1, 0].set_xlabel("Pitch angle")
ax[1, 0].set_ylabel("amount")
ax[1, 1].hist(inputs[:train_index,3], bins=15, color='gray',edgecolor='black', alpha=0.2)
ax[1, 1].hist(inputs[train_index:validation_index,3], bins=15, color='red', alpha=.7)
ax[1, 1].hist(inputs[validation_index:,3], bins=15, alpha=.5, edgecolor='black', color='yellow')
ax[1, 1].set_title("Yaw 2")
ax[1, 1].set_xlabel("Yaw angle")
ax[1, 1].set_ylabel("amount")

ax[0, 0].legend(["Training","Validation","Test"],loc="best")
ax[0, 1].legend(["Training","Validation","Test"],loc="best")
ax[1, 0].legend(["Training","Validation","Test"],loc="best")
ax[1, 1].legend(["Training","Validation","Test"],loc="best")
plt.show()
      
      
errorlist = np.array([])

for i in range(np.size(x_test[:,1])):
    a_predict = scaler.inverse_transform(model.predict(x_test[i:i+1,:]))[0,1]
    k_predict = scaler.inverse_transform(model.predict(x_test[i:i+1,:]))[0,0]
    
    a_true = scaler.inverse_transform(y_test[i:i+1,:])[0,1]
    k_true = scaler.inverse_transform(y_test[i:i+1,:])[0,0]
    
    diff = weibullmedian(a_predict,k_predict)-weibullmedian(a_true,k_true)
    pro = diff/weibullmedian(a_true,k_true)
    print(pro)
    errorlist = np.append(errorlist, pro)

error_power = errorlist

plt.hist(errorlist,bins=30, density=True)
plt.xlim=(-1,1)
plt.show()


print(inputs[validation_index:,:][0])
print(scaler.inverse_transform(model.predict(x_test[0:1,:]))[0])
print(scaler.inverse_transform(y_test[0:1,:])[0])
      
print(x_test[0:1,:])


##############################################################################
#                                                                            #
#                                   Flap1                                    #
#                                                                            #
##############################################################################

################################File loading##################################
inputslist = []
outputlist = []
with open(Directory_data+"T_InputDEL_DELFlap") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        inputslist.append(row)
        
with open(Directory_data+"T_OutputDEL_DELFlap") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        outputlist.append(row)


inputs  = np.array(inputslist)
outputs = np.array(outputlist)
inputs, outputs = unison_shuffled_copies(inputs, outputs)

####################################Data#####################################
scaler   = MinMaxScaler(feature_range=(0,1))
x_scaled = scaler.fit_transform(inputs)
y_scaled = scaler.fit_transform(outputs)
x_train = x_scaled[:train_index,:]
y_train = y_scaled[:train_index,:]
x_validation = x_scaled[train_index:validation_index,:]
y_validation = y_scaled[train_index:validation_index,:]
x_test = x_scaled[validation_index:,:]
y_test = y_scaled[validation_index:,:]

#################################Model#######################################
K.clear_session()
model_flap= Sequential()
model_flap.add(Dense(neuronnumber, input_dim = x_train.shape[1], kernel_initializer='he_uniform', activation='relu'))
model_flap.add(Dense(neuronnumber, activation='relu'))
model_flap.add(Dense(2))
model_flap.summary()

model_flap.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
tbGraph = TensorBoard(log_dir = r'C:\Users\marc0\Desktop\Bachelor\Machinelearning\logs\{now}',
                      histogram_freq=64, write_graph=True, write_images=True)
history=model_flap.fit(x_train, y_train, epochs=epochnumber,
                  batch_size=16, 
                  verbose=2, 
                  validation_data=(x_validation,y_validation),
                  callbacks=[tbGraph])

print(history.history.keys())

################################Plots#######################################

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.legend()
plt.show()


errorlist = np.array([])

for i in range(np.size(x_test[:,1])):
    a_predict = scaler.inverse_transform(model_flap.predict(x_test[i:i+1,:]))[0,1]
    k_predict = scaler.inverse_transform(model_flap.predict(x_test[i:i+1,:]))[0,0]
    
    a_true = scaler.inverse_transform(y_test[i:i+1,:])[0,1]
    k_true = scaler.inverse_transform(y_test[i:i+1,:])[0,0]
    
    diff = weibullmedian(a_predict,k_predict)-weibullmedian(a_true,k_true)
    pro = diff/weibullmedian(a_true,k_true)
    print(pro)
    errorlist = np.append(errorlist, pro)

plt.hist(errorlist,bins=30, density=True)
plt.xlim=(-1,1)
plt.show()


error_flap = errorlist



##############################################################################
#                                                                            #
#                                   Tow                                      #
#                                                                            #
##############################################################################

################################File loading##################################
inputslist = []
outputlist = []
with open(Directory_data+"T_InputDEL_DELBtow") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        inputslist.append(row)
        
with open(Directory_data+"T_OutputDEL_DELBtow") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        outputlist.append(row)


inputs  = np.array(inputslist)
outputs = np.array(outputlist)
inputs, outputs = unison_shuffled_copies(inputs, outputs)

####################################Data#####################################
scaler   = MinMaxScaler(feature_range=(0,1))
x_scaled = scaler.fit_transform(inputs)
y_scaled = scaler.fit_transform(outputs)
x_train = x_scaled[:train_index,:]
y_train = y_scaled[:train_index,:]
x_validation = x_scaled[train_index:validation_index,:]
y_validation = y_scaled[train_index:validation_index,:]
x_test = x_scaled[validation_index:,:]
y_test = y_scaled[validation_index:,:]

#################################Model#######################################
K.clear_session()
model_tow= Sequential()
model_tow.add(Dense(neuronnumber, input_dim = x_train.shape[1], kernel_initializer='he_uniform', activation='relu'))
model_tow.add(Dense(neuronnumber, activation='relu'))
model_tow.add(Dense(2))
model_tow.summary()

model_tow.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
tbGraph = TensorBoard(log_dir = r'C:\Users\marc0\Desktop\Bachelor\Machinelearning\logs\{now}',
                      histogram_freq=64, write_graph=True, write_images=True)
history=model_tow.fit(x_train, y_train, epochs=epochnumber,
                  batch_size=16, 
                  verbose=2, 
                  validation_data=(x_validation,y_validation),
                  callbacks=[tbGraph])

print(history.history.keys())


################################Plots#######################################

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.legend()
plt.show()




errorlist = np.array([])

for i in range(np.size(x_test[:,1])):
    a_predict = scaler.inverse_transform(model_tow.predict(x_test[i:i+1,:]))[0,1]
    k_predict = scaler.inverse_transform(model_tow.predict(x_test[i:i+1,:]))[0,0]
    
    a_true = scaler.inverse_transform(y_test[i:i+1,:])[0,1]
    k_true = scaler.inverse_transform(y_test[i:i+1,:])[0,0]
    
    diff = weibullmedian(a_predict,k_predict)-weibullmedian(a_true,k_true)
    pro = diff/weibullmedian(a_true,k_true)
    print(pro)
    errorlist = np.append(errorlist, pro)

plt.hist(errorlist,30, density=True)
plt.show()


error_tow = errorlist



################################Plots#######################################

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.legend()
plt.show()



print("Loss function")
print(np.array(history.history['loss'])[np.size(np.array(history.history['loss']))-1])


# =============================================================================
# errorlist = np.array([])
# 
# for i in range(np.size(x_test[:,1])):
#     diff = scaler.inverse_transform(model_tow.predict(x_test[i:i+1,:]))[0,1]-scaler.inverse_transform(y_test[i:i+1,:])[0,1]
#     pro = diff/scaler.inverse_transform(y_test[i:i+1,:])[0,1]
#     print(pro)
#     errorlist = np.append(errorlist, pro)
# 
# plt.hist(errorlist,30, density=True)
# plt.show()
# 
# =============================================================================


plt.rcParams.update({'font.size': 16})
data = [error_power*100, error_flap*100 , error_tow*100]
fig, ax = plt.subplots(figsize=(16,10))
ax.set_title("Prediction accuracy")
ax.boxplot(data, labels=["Power","Flap", "Tow"], medianprops={"linewidth": 2}, boxprops={"linewidth": 3}, whiskerprops={"linewidth": 3}, capprops={"linewidth": 3})
ax.set_ylabel("Deviation from expected value [%]")
plt.ylim(-2.25, 2.25)
#plt.yticks(np.arange(-2,2.5,0.5))
plt.yticks([-2,-1,-0.5,-0.25,0,0.25,0.5,1,2])
plt.grid(axis = 'y')
plt.show()
# =============================================================================
# 
# 
# model_power_json = model.to_json()
# with open("model_power.json", "w") as json_file:
#     json_file.write(model_power_json)
# # serialize weights to HDF5
# model.save_weights("model_power.h5")
# print("Saved model to disk")
# 
# 
# model_flap_json = model_flap.to_json()
# with open("model_flap.json", "w") as json_file:
#     json_file.write(model_flap_json)
# # serialize weights to HDF5
# model_flap.save_weights("model_flap.h5")
# print("Saved model to disk")
# 
# model_tow_json = model_tow.to_json()
# with open("model_tow.json", "w") as json_file:
#     json_file.write(model_tow_json)
# # serialize weights to HDF5
# model_tow.save_weights("model_tow.h5")
# print("Saved model to disk")
# 
# 
# 
# =============================================================================
