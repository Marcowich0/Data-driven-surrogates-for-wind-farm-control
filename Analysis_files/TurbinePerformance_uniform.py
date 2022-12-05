# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import matplotlib.pyplot as plt


def get(var):
    index = np.where(Sensors==var)
    return dataraw[:,index[0]]


"""
Spyder Editor

This is a temporary script file.
"""
Directory_sweep = r'C:\Users\marc0\Desktop\Bachelor\sweep\\' 
Directory_Matlab = r'C:\Users\marc0\Desktop\Bachelor\Matlab\\'


U=np.arange(5,26)

Sensorstemp=open(Directory_Matlab+"SENSORnames").read().splitlines()
Sensors=np.array(Sensorstemp)
print(Sensors)

R     = 198/2
P0    = 10000
Ngear = 50
rho   = 1.225



#Isolerer datafiler til seperat array
datafiles = []
files = np.array(os.listdir(Directory_sweep))
for j in range(len(files)):
    if "sweep.0" in files[j] and ".tim" in files[j]:
        datafiles.append(files[j])

print(datafiles)


for i in range(len(datafiles)):
    dataraw=np.loadtxt(Directory_sweep+datafiles[i])
    
    Pitch    = get("Pitch")
    Omega    = get("Omega")
    Power    = get("Power")
    Thrust   = get("ThrustN")
    Velocity = get("Vhub")
    
    CP = 1000*Power/(1/2*rho*Velocity**3*math.pi*R**2)
    CT = 1000*Thrust/(1/2*rho*Velocity**2*math.pi*R**2)
    Pitch = get("Pitch")
    Omega = get("Omega")
    Power

V=get("Vhub")

cpsmooth     = np.array([])
ctsmooth     = np.array([])
powersmooth  = np.array([])
pitchsmooth  = np.array([])
omegasmooth  = np.array([])
Thrustsmooth = np.array([])


for j in range(len(U)):
    temp1=np.array([])
    temp2=np.array([])
    temp3=np.array([])
    temp4=np.array([])
    temp5=np.array([])
    temp6=np.array([])
    print(U[j])
    for i in range(len(V)):
        if V[i]>(U[j]-1/2) and V[i]<(U[j]+1/2):
            temp1=np.append(temp1, CP[i])
            temp2=np.append(temp2, CT[i])
            temp3=np.append(temp3, Power[i])
            temp4=np.append(temp4, Pitch[i])
            temp5=np.append(temp5, Omega[i])
            temp6=np.append(temp6, Thrust[i])
            
            
    cpsmooth     = np.append(cpsmooth, np.mean(temp1))
    ctsmooth     = np.append(ctsmooth, np.mean(temp2))
    powersmooth  = np.append(powersmooth, np.mean(temp3))
    pitchsmooth  = np.append(pitchsmooth, np.mean(temp4))
    omegasmooth  = np.append(omegasmooth, np.mean(temp5))
    Thrustsmooth = np.append(Thrustsmooth, np.mean(temp6))


plt.plot(U,cpsmooth)
plt.ylabel('CP')
plt.xlabel('U')
plt.show()

thickness = 4

fig, ax = plt.subplots(1, 2, figsize=(16,6))
ax[0].plot(U,omegasmooth, '#FC7634', linewidth=thickness)
ax[0].set_title("Rotational speed")
ax[0].set_xlabel("U [m/s]")
ax[0].set_ylabel("\u03A9 [rad/s]")


ax[1].plot(U,pitchsmooth, '#008835', linewidth=thickness) 
ax[1].set_title("Pitch")
ax[1].set_xlabel("U [m/s]")
ax[1].set_ylabel("\u03B8 [rad]")
plt.savefig('rotpitch.svg',format='svg', dpi=1200)
plt.show()


fig, ax = plt.subplots(2, 3, figsize=(16,10))


ax[0, 0].plot(U,powersmooth, 'r')
ax[0, 0].set_title("Power")
ax[0, 0].set_xlabel("Velocity [m/s]")
ax[0, 0].set_ylabel("Power [W]")
ax[0, 1].plot(U,Thrustsmooth, 'b')
ax[0, 1].set_title("Thrust")
ax[0, 1].set_xlabel("Velocity [m/s]")
ax[0, 1].set_ylabel("Thrust [N]")
ax[0, 2].plot(U,pitchsmooth, 'g') 
ax[0, 2].set_title("Pitch")
ax[0, 2].set_xlabel("Velocity [m/s]")
ax[0, 2].set_ylabel("Pitch [rad]")
ax[1, 0].plot(U,cpsmooth, 'c') 
ax[1, 0].set_title("CP")
ax[1, 0].set_xlabel("Velocity [m/s]")
ax[1, 0].set_ylabel("CP []")
ax[1, 1].plot(U,ctsmooth, 'm') 
ax[1, 1].set_title("CT")
ax[1, 1].set_xlabel("Velocity [m/s]")
ax[1, 1].set_ylabel("CT []")
ax[1, 2].plot(U,omegasmooth, 'y')
ax[1, 2].set_title("Omega")
ax[1, 2].set_xlabel("Velocity [m/s]")
ax[1, 2].set_ylabel("RPM [rad/s]")
plt.show()

