# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

def f(x, y):
    return cp_arr[x,y]

def get(var):
    index = np.where(Sensors==var)
    return dataraw[:,index[0]]
"""
Spyder Editor

This is a temporary script file.
"""
Directory_sweep = r'C:\Users\marc0\Desktop\Bachelor\sweep\\'
Directory_Matlab = r'C:\Users\marc0\Desktop\Bachelor\Matlab\\'
Directory_surfaces = r'C:\Users\marc0\Desktop\Bachelor\cpctfiles\\'


U=np.array([8])

Sensorstemp=open(Directory_Matlab+"SENSORnames").read().splitlines()
Sensors=np.array(Sensorstemp)

R     = 198/2
P0    = 10000
Ngear = 50
rho   = 1.225



#Isolerer datafiler til seperat array
datafolders = []
folders = np.array(os.listdir(Directory_surfaces))
print(folders[1][1])
for j in range(len(folders)):
    if folders[j][0]=="o" and folders[j][4]=="p":
        datafolders.append(folders[j])

print("Datafolders")
print(datafolders)
datafiles=[]
for i in range(len(datafolders)):
    files = np.array(os.listdir(Directory_surfaces+datafolders[i]))
    
    for j in range(len(files)):
        if "sweep.0" in files[j] and ".tim" in files[j]:
            datafiles.append(datafolders[i] + "/" + files[j])

print("Datafiles")
print(datafiles)


# Making array for data CP
x_arr= np.array([])
y_arr= np.array([])
z_arr= np.array([])

for i in range(len(datafolders)):
    x_arr = np.append(x_arr,int(datafolders[i][1:4]))
    y_arr = np.append(y_arr,int(datafolders[i][5:8]))
    z_arr = np.append(z_arr,int(datafolders[i][9:12]))

x_range=int(np.max(x_arr)-np.min(x_arr))+1
y_range=int(np.max(y_arr)-np.min(y_arr))+1
z_range=int(np.max(z_arr)-np.min(z_arr))+1

print("RANGES")
print(x_range)
print(y_range)
print(z_range)
cp_arr=np.zeros((x_range, y_range, z_range))
ct_arr=np.zeros((x_range, y_range, z_range))

omega_arr=np.zeros(x_range)
pitch_arr=np.zeros(y_range)
yaw_arr=np.zeros(z_range)

for k in range(len(datafiles)):
    dataraw=np.loadtxt(Directory_surfaces+datafiles[k])
    
    Thrust   = get("ThrustN")
    Velocity = get("Vhub")
    Omega = get("Omega")
    TorqueN = get("TorqueN")
    Pitch = get("Pitch")
    Yaw = get("TiltAngle")
    
    
    Power = Omega*TorqueN
    CP = 1000*Power/(1/2*rho*Velocity**3*math.pi*R**2)
    CT = 1000*Thrust/(1/2*rho*Velocity**2*math.pi*R**2)
    
    omega_index  = int(datafiles[k][1:4])-1
    pitch_index  = int(datafiles[k][5:8])-1   
    yaw_index   = int(datafiles[k][9:12])-1   
    
    cp_arr[omega_index , pitch_index , yaw_index] = np.mean(CP)
    ct_arr[omega_index , pitch_index , yaw_index] = np.mean(CT)
    omega_arr[omega_index] = round(np.mean(Omega),2)
    pitch_arr[pitch_index] = round(np.mean(Pitch),2)
    yaw_arr  [yaw_index]   = round(np.mean(Yaw),2)



#fig, ax = plt.subplots(2,2, figsize=(12,7))

print("SHAPE")
print(np.shape(cp_arr[:,:,:]))
print(np.size(omega_arr))
print(np.size(pitch_arr))
print(np.size(yaw_arr))
print(yaw_arr)

yaw_arr = np.arange(-30,32,2)
print(omega_arr)
print(pitch_arr)
print(yaw_arr)
print(cp_arr[1,:,:])

for i in range(1,1):
    plt.contourf(pitch_arr,omega_arr,cp_arr[:,:,i], 30)
    plt.colorbar()
    plt.title('CP - Yaw=' + str(yaw_arr[i]))
    plt.xlabel('$\Theta$')
    plt.ylabel('$\Omega$')
    plt.show()
    
    plt.contourf( yaw_arr,pitch_arr,cp_arr[i,:,:],30)
    plt.colorbar()
    plt.title('CP - Omega=' + str(omega_arr[i]))
    plt.xlabel('$\Phi$')
    plt.ylabel('$\Theta$')
    plt.show()
    
    plt.contourf( yaw_arr,omega_arr,cp_arr[:,i,:],30)
    plt.colorbar()
    plt.title('CP - Pitch=' + str(pitch_arr[i]))
    plt.xlabel('$\Phi$')
    plt.ylabel('$\Omega$')
    plt.show()


vminval=0.35
vmaxval=0.5
cmap = mpl.cm.gist_rainbow
norm = mpl.colors.Normalize(vmin=vminval, vmax=vmaxval)


plt.figure(figsize=(20,12))
plt.rcParams.update({'font.size': 22})

plt.contourf( yaw_arr,omega_arr,cp_arr[:,6,:],20, cmap=cmap, vmin=vminval, vmax=vmaxval)
plt.title('CP - Pitch=' + str(pitch_arr[6]))
plt.xlabel('$\Psi$')
plt.ylabel('$\Omega$')

plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))

plt.show()


vminval=0.35
vmaxval=0.5
cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=vminval, vmax=vmaxval)

fig, ax = plt.subplots(1, 2, figsize=(32,8))


ax[0].contourf( yaw_arr,omega_arr,cp_arr[:,6,:],25, cmap=cmap, vmin=vminval, vmax=vmaxval)
ax[0].set_title('CP - Pitch=' + str(pitch_arr[6]))
ax[0].set_xlabel('$\Psi$')
ax[0].set_ylabel('$\Omega$')
ax[1].contourf( pitch_arr,omega_arr,cp_arr[:,:,15],25, cmap=cmap, vmin=vminval, vmax=vmaxval)
ax[1].set_title('CP - yaw=' + str(yaw_arr[15]))
ax[1].set_xlabel('\u03B8')
ax[1].set_ylabel('$\Omega$')

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax[:])

plt.show()





vminval=0.5
vmaxval=1.2
cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=vminval, vmax=vmaxval)

fig, ax = plt.subplots(1, 2, figsize=(32,8))


ax[0].contourf( yaw_arr,omega_arr,ct_arr[:,6,:],25, cmap=cmap, vmin=vminval, vmax=vmaxval)
ax[0].set_title('CT - Pitch=' + str(pitch_arr[6]))
ax[0].set_xlabel('$\Psi$')
ax[0].set_ylabel('$\Omega$')
ax[1].contourf( pitch_arr,omega_arr,ct_arr[:,:,15],25, cmap=cmap, vmin=vminval, vmax=vmaxval)
ax[1].set_title('CT - yaw=' + str(yaw_arr[15]))
ax[1].set_xlabel('\u03B8')
ax[1].set_ylabel('$\Omega$')

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax[:])

plt.show()
