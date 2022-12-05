# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:25:31 2022

@author: Marc
"""


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

def weibullmedian(a,k):
    return (a*math.log(2)**(1/k))

######################Load models##########################

Directory_data = r'C:\Users\marc0\Desktop\Bachelor\Machinelearning2\\'

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
Predict = scaler.transform([[25,-50,75,15]])
count=0

#################Control setting used in optimization######################
config = np.array([0,0,90,0])
for y1 in range(0,31,2):
    for p1 in range(-50,301,50):
        for o1 in range(60,93,15):
            for y2 in range(-30,31,2):
                config = np.vstack((config,np.array([y1,p1,o1,y2])))



config_scaled=scaler.transform(config)

#Define output scaling
y_scaled = scaler.fit_transform(outputs)

###############Calculating loads for normal operation######################

a_normalload_flap = scaler.inverse_transform(model_flap.predict(config_scaled[0:1,:]))[0][1]
k_normalload_flap = scaler.inverse_transform(model_flap.predict(config_scaled[0:1,:]))[0][0]

a_normalload_tow = scaler.inverse_transform(model_tow.predict(config_scaled[0:1,:]))[0][1]
k_normalload_tow = scaler.inverse_transform(model_tow.predict(config_scaled[0:1,:]))[0][0]

normalload_flap = weibullmedian(a_normalload_flap, k_normalload_flap)
normalload_tow = weibullmedian(a_normalload_tow, k_normalload_tow)

#Defining constrains
loadconstraints = np.arange(-0.05,0.15,0.005)
maxpower_arr = np.zeros(np.size(loadconstraints))
std_arr = np.zeros(np.size(loadconstraints))
index_arr = np.zeros(np.size(loadconstraints))

for i in range(np.size(config[:,0])-2):
    a_power = scaler.inverse_transform(model_power.predict(config_scaled[i+1:i+2,:]))[0][1]
    k_power = scaler.inverse_transform(model_power.predict(config_scaled[i+1:i+2,:]))[0][0]
    
    a_flap =  scaler.inverse_transform(model_flap.predict(config_scaled[i+1:i+2,:]))[0][1]
    k_flap =  scaler.inverse_transform(model_flap.predict(config_scaled[i+1:i+2,:]))[0][0]
    
    a_tow =  scaler.inverse_transform(model_tow.predict(config_scaled[i+1:i+2,:]))[0][1]
    k_tow =  scaler.inverse_transform(model_tow.predict(config_scaled[i+1:i+2,:]))[0][0]
   
    power = weibullmedian(a_power,k_power)  
    load_flap = weibullmedian(a_flap,k_flap) 
    load_tow = weibullmedian(a_tow,k_tow)  
   
    count = count + 1
    
    print("Progress: " + str(count) + "/" + str(np.size(config[:,0])))
    #Checking constrain conditions
    for m in range(np.size(loadconstraints)):   
        if (power>maxpower_arr[m] and ((load_flap-normalload_flap)/normalload_flap)<=loadconstraints[m] and ((load_tow-normalload_tow)/normalload_tow)<=loadconstraints[m]):
            index_arr[m] = i
            maxpower_arr[m] = power
            std_arr[m] = a_power*math.sqrt(math.gamma(1+2/k_power)-(math.gamma(1+1/k_power))**2)
        

print("MAX POWER")
print("Normal operation power")
print(scaler.inverse_transform(model_power.predict(config_scaled[0:1,:]))[0][1])
print("Maximum power")
print(config[int(index_arr[0]):int(index_arr[0])+1,:])

#Normal power production
k_normal = scaler.inverse_transform(model_power.predict(config_scaled[0:1,:]))[0][0]
a_normal = scaler.inverse_transform(model_power.predict(config_scaled[0:1,:]))[0][1]
normal = weibullmedian(a_normal,k_normal)
delta_u = a_normal*math.sqrt(math.gamma(1+2/k_normal)-(math.gamma(1+1/k_normal))**2)

power_increse = (maxpower_arr-normal)/normal

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})
plt.plot(loadconstraints*100,power_increse*100, color='darkred', linewidth=5.0)
plt.title("Posible gain as a function of load constrains")
plt.xlabel("Load increase [%]")
plt.ylabel("Power increase [%]")
plt.xlim(-5,15)
plt.ylim(0,10)
plt.show()

y_scaled = scaler.fit_transform(outputs)
print(scaler.inverse_transform(model_power.predict(Predict))[0][0])
print(scaler.inverse_transform(model_power.predict(Predict))[0][1])

x_scaled = scaler.fit_transform(inputs)

load0=0
load0_index=int(index_arr[int(load0*200+10)])
print(load0_index)

load2=0.025
load2_index=int(index_arr[int(load2*200+10)])
print(load2_index)


load5=0.05
load5_index=int(index_arr[int(load5*200+10)])
print(load5_index)

load7=0.075
load7_index=int(index_arr[int(load7*200+10)])
print(load7_index)


load10=0.1
load10_index=int(index_arr[int(load10*200+10)])
print(load10_index)


print("0% load constraint")
print(config[load0_index,:])
delta_p=std_arr[int(load0*200+10)]
print(str(round(100*power_increse[int(load0*200+10)],1))  + " +- " +  str(((delta_p**2+delta_u**2)/(power_increse[int(load0*200+10)]-normal)**2+(delta_u/normal)**2)*100))

print("2.5% load constraint")
print(config[load2_index,:])
delta_p=std_arr[int(load2*200+10)]
print(str(round(100*power_increse[int(load2*200+10)],1))  + " +- " +  str(((delta_p**2+delta_u**2)/(power_increse[int(load2*200+10)]-normal)**2+(delta_u/normal)**2)*100))


print("5% load constraint")
print(config[load5_index,:])
delta_p=std_arr[int(load5*200+10)]
print(str(round(100*power_increse[int(load5*200+10)],1))  + " +- " +  str(((delta_p**2+delta_u**2)/(power_increse[int(load5*200+10)]-normal)**2+(delta_u/normal)**2)*100))

print("7.5% load constraint")
print(config[load7_index,:])
delta_p=std_arr[int(load7*200+10)]
print(str(round(100*power_increse[int(load7*200+10)],1))  + " +- " +  str(((delta_p**2+delta_u**2)/(power_increse[int(load7*200+10)]-normal)**2+(delta_u/normal)**2)*100))

print("2.5% load constraint")
print(config[load10_index,:])
delta_p=std_arr[int(load10*200+10)]
print(str(round(100*power_increse[int(load10*200+10)],1))  + " +- " +  str(((delta_p**2+delta_u**2)/(power_increse[int(load10*200+10)]-normal)**2+(delta_u/normal)**2)*100))


#print(scaler.inverse_transform(model_power.predict(config_scaled))[0][1])

#print(scaler.inverse_transform(model_flap.predict(config_scaled))[0][1])

#print(scaler.inverse_transform(model_tow.predict(config_scaled))[0][1])

print(power_increse*100)
np.savetxt('power_increase_6R_unlocked.out', power_increse*100, delimiter=',')
np.savetxt('loadconstrains_6R_unlocked.out', loadconstraints*100, delimiter=',')


load10_index=int(index_arr[int(np.size(index_arr)-1)])
print(config[load10_index])
print(load10_index)