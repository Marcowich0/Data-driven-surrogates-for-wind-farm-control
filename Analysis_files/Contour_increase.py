

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import csv
from tensorflow.keras.callbacks import TensorBoard
import datetime
import tensorflow as tf
import math
import matplotlib as mpl

# =============================================================================
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
# tf.debugging.set_log_device_placement(False)
# 
# =============================================================================
def mirror(M):
    temp = np.zeros(np.shape(M))
    print(np.size(M[:,0]))
    print(np.size(M[0,:]))
    for i in range(np.size(M[:,0])):
        print(i)
        temp[i,:] = M[np.size(M[:,0])-1-i,:]
    return temp

def weibullmedian(a,k):
    return (a*math.log(2)**(1/k))

######################Load models##########################

Directory_data = r'C:\Users\marc0\Desktop\Bachelor\Machinelearning\\'

json_file = open('model_power.json', 'r')
loaded_model_power_json = json_file.read()
json_file.close()
model_power = model_from_json(loaded_model_power_json)
model_power.load_weights("model_power.h5")
print("Loaded model from disk")


json_file = open('model_flap.json', 'r')
loaded_model_flap_json = json_file.read()
json_file.close()
model_flap = model_from_json(loaded_model_flap_json)
model_flap.load_weights("model_flap.h5")
print("Loaded model from disk")


json_file = open('model_tow.json', 'r')
loaded_model_tow_json = json_file.read()
json_file.close()
model_tow = model_from_json(loaded_model_tow_json)
model_tow.load_weights("model_tow.h5")
print("Loaded model from disk")


####################Load training data sample for scalings####################

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

scaler   = MinMaxScaler(feature_range=(0,1))
x_scaled = scaler.fit_transform(inputs)


##################Sample Prediction##################
Predict = scaler.transform([[0,0,90,0]])

y1_range = range(0,31,1)
y2_range = range(-30,31,1)

Power_matrix = np.zeros(( np.size(y1_range) , np.size(y2_range) ))
flap_matrix = np.zeros(( np.size(y1_range) , np.size(y2_range) ))
tow_matrix = np.zeros(( np.size(y1_range) , np.size(y2_range) ))
load_matrix = np.zeros(( np.size(y1_range) , np.size(y2_range) ))

y_scaled = scaler.fit_transform(outputs)
a_normal_power = scaler.inverse_transform(model_power.predict(Predict))[0][1]
k_normal_power = scaler.inverse_transform(model_power.predict(Predict))[0][0]

a_normal_flap = scaler.inverse_transform(model_flap.predict(Predict))[0][1]
k_normal_flap = scaler.inverse_transform(model_flap.predict(Predict))[0][0]

a_normal_tow = scaler.inverse_transform(model_tow.predict(Predict))[0][1]
k_normal_tow = scaler.inverse_transform(model_tow.predict(Predict))[0][0]

normal = weibullmedian(a_normal_power,k_normal_power)
normal_flap = weibullmedian(a_normal_flap,k_normal_flap)
normal_tow = weibullmedian(a_normal_tow,k_normal_tow)

x_scaled = scaler.fit_transform(inputs)

config = np.array([0,0,90,0])
y1_index = 0
for y1 in y1_range:
    y2_index = 0
    for y2 in y2_range:
        maxpower = 0
        load_flap = 0
        load_tow = 0
        for o1 in range(90,93,10):
            print("##########################################")
            print(y1_index)
            print("##########################################")
            for p1 in range(-00,300,1000):
                x_scaled = scaler.fit_transform(inputs)
                config = scaler.transform(np.array([[y1,p1,o1,y2]]))
                y_scaled = scaler.fit_transform(outputs)
                a_power = scaler.inverse_transform(model_power.predict(config))[0][1]
                k_power = scaler.inverse_transform(model_power.predict(config))[0][0]
                power = weibullmedian(a_power,k_power)
                if (power>maxpower):
                    maxpower = power
                    
                    a_flap = scaler.inverse_transform(model_flap.predict(config))[0][1]
                    k_flap = scaler.inverse_transform(model_flap.predict(config))[0][0]
                    
                    a_tow = scaler.inverse_transform(model_tow.predict(config))[0][1]
                    k_tow = scaler.inverse_transform(model_tow.predict(config))[0][0]
                    
                    load_flap = weibullmedian(a_flap,k_flap)
                    load_tow = weibullmedian(a_tow,k_tow)
                    
                    flap_increase = (load_flap-normal_flap)/normal_flap
                    tow_increase = (load_tow-normal_tow)/normal_tow
                    
                    flap_matrix[y1_index,y2_index] = flap_increase
                    tow_matrix[y1_index,y2_index] = tow_increase
                    
        if (flap_increase>0.1 or tow_increase>0.1):
            load_matrix[y1_index,y2_index] = 1
        Power_matrix[y1_index,y2_index] = maxpower
        y2_index = y2_index + 1
    y1_index = y1_index + 1
    
power_increase = (Power_matrix-normal)/normal
# =============================================================================
# vminval=-0.05
# vmaxval=0.25
# cmap = mpl.cm.turbo
# norm = mpl.colors.Normalize(vmin=vminval, vmax=vmaxval)
# 
# plt.figure(figsize=(20,12))
# plt.rcParams.update({'font.size': 28})
# 
# plt.contourf(y2_range,range(-30,2,2), mirror(power_increase),30, cmap=cmap, vmin=vminval, vmax=vmaxval)
# plt.colorbar()
# plt.show()
# 
# print(np.shape(range(-30,31,1)))
# =============================================================================

vminval=-0.25
vmaxval=0.60
cmap = mpl.cm.turbo
norm = mpl.colors.Normalize(vmin=vminval, vmax=vmaxval)

fig, ax = plt.subplots(1, 3, figsize=(54,14))
plt.rcParams.update({'font.size': 34})
power_increase = (Power_matrix-normal)/normal
ax[0].contourf(y2_range,range(-30,1,1), mirror(power_increase),30, cmap=cmap, vmin=vminval, vmax=vmaxval)
ax[0].set_xticks(range(-30,40,10))
ax[0].set_yticks(range(-30,5,5))
ax[0].set_xlabel("$\u03C8_2$")
ax[0].set_ylabel("$\u03C8_1$")
ax[0].set_title("Power increase [%]")
ax[1].contourf(y2_range,range(-30,1,1), mirror(flap_matrix),30, cmap=cmap, vmin=vminval, vmax=vmaxval)
ax[1].set_xlabel("$\u03C8_2$")
ax[1].set_ylabel("$\u03C8_1$")
ax[1].set_title("Flap load increase [%]")
ax[2].contourf(y2_range,range(-30,1,1), mirror(tow_matrix),30, cmap=cmap, vmin=vminval, vmax=vmaxval)
ax[2].set_xlabel("$\u03C8_2$")
ax[2].set_ylabel("$\u03C8_1$")
ax[2].set_title("Tower load increase [%]")
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax[:])
plt.show()

print(np.greater(flap_matrix,0.1))

fig, ax = plt.subplots(figsize=(14,14))
cp = ax.contourf(y2_range,range(-30,2,2), np.greater(mirror(flap_matrix),0.1),1)
ax.clabel(cp, inline=True, 
          fontsize=10)
plt.show()

print(np.greater(mirror(flap_matrix),0.1))
# =============================================================================
# config_scaled=scaler.transform(config)
# 
# #Define output scaling
# y_scaled = scaler.fit_transform(outputs)
# 
# ###############Calculating loads for normal operation######################
# normalload_flap = scaler.inverse_transform(model_flap.predict(config_scaled[0:1,:]))[0][1]
# normalload_tow = scaler.inverse_transform(model_tow.predict(config_scaled[0:1,:]))[0][1]
# 
# #Defining constrains
# loadconstraints = np.arange(-0.25,0.25,0.005)
# maxpower_arr = np.zeros(np.size(loadconstraints))
# index_arr = np.zeros(np.size(loadconstraints))
# 
# for i in range(np.size(config[:,0])-2):
#     power = scaler.inverse_transform(model_power.predict(config_scaled[i+1:i+2,:]))[0][1]
#     load_flap =  scaler.inverse_transform(model_flap.predict(config_scaled[i+1:i+2,:]))[0][1]
#     load_tow =  scaler.inverse_transform(model_tow.predict(config_scaled[i+1:i+2,:]))[0][1]
#    
#     #Checking constrain conditions
#     for m in range(np.size(loadconstraints)):   
#         if (power>maxpower_arr[m] and ((load_flap-normalload_flap)/normalload_flap)<loadconstraints[m] and ((load_tow-normalload_tow)/normalload_tow)<loadconstraints[m]):
#             index_arr[m] = i
#             maxpower_arr[m] = power
#         
# 
# print("MAX POWER")
# print("Normal operation power")
# print(scaler.inverse_transform(model_power.predict(config_scaled[0:1,:]))[0][1])
# print("Maximum power")
# print(config[int(index_arr[0]):int(index_arr[0])+1,:])
# 
# #Normal power production
# normal = scaler.inverse_transform(model_power.predict(config_scaled[0:1,:]))[0][1]
# power_increse = (maxpower_arr-normal)/normal
# 
# plt.figure(figsize=(20,10))
# plt.rcParams.update({'font.size': 22})
# plt.plot(loadconstraints*100,power_increse*100, color='darkred', linewidth=3.0)
# plt.title("Posible gain as a function of load constrains")
# plt.xlabel("Load increase [%]")
# plt.ylabel("Power increase [%]")
# plt.xlim(-11,25)
# plt.ylim(0.25)
# plt.show()
# 
# print(scaler.inverse_transform(model_power.predict(Predict))[0][0])
# print(scaler.inverse_transform(model_power.predict(Predict))[0][1])
# 
# 
# =============================================================================
