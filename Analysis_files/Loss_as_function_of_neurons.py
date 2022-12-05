# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:25:07 2022

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


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

Directory_data = r'C:\Users\marc0\Desktop\Bachelor\Machinelearning\\'

neuronnumber = 10
epochnumber = 300
plt.rcParams.update({'font.size': 20})



for l in range(30):
    neuronarr= np.array([])
    errorarr = np.array([])
    inputslist = []
    outputlist = []
    for i in range(2,14,1):
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
        
        error = np.array(history.history['loss'])[np.size(np.array(history.history['loss']))-1]
        errorarr = np.append(errorarr, error)
        neuronarr = np.append(neuronarr,i)
        
    if (l==0):
        errormatrix = errorarr
    else:
        errormatrix = np.vstack((errormatrix,errorarr))
        
print(errormatrix)

meanerror = np.array([])
meanstd = np.array([])

print(np.size(errormatrix[1,:]))
for i in range(np.size(errormatrix[1,:])):
    meanerror = np.append(meanerror, np.mean(errormatrix[:,i]))
    meanstd = np.append(meanstd,np.std(errormatrix[:,i])/math.sqrt(10))


print(meanerror)
print(meanstd)

plt.figure(figsize=(20,10))
plt.errorbar(neuronarr, meanerror,  yerr=meanstd, linewidth=6.0, marker='s', mfc='red',
         mec='black', ms=10, mew=4, color='red', ecolor='black')
plt.xticks(range(2,14,1))
plt.title("Loss as a function of neurons (Mean Squared Error) - Power")
plt.xlabel("Neurons pr. layer")
plt.ylabel("Loss (MSE)")
plt.show()
