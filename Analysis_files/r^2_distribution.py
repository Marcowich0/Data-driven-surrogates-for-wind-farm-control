# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:46:42 2022

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn

power = np.loadtxt(r'C:\Users\marc0\Desktop\Bachelor\R^2\R2values_power')
flap = np.loadtxt(r'C:/Users/marc0/Desktop/Bachelor/R^2/R2values_DELFlap')
tow = np.loadtxt(r'C:\Users\marc0\Desktop\Bachelor\R^2\R2values_DELBtow')


plt.hist(power,40, density=True)
plt.xlabel("R^2 values")
plt.title("Weibull accuracy - power")
plt.show()

plt.hist(flap,40)
plt.xlabel("R^2 values")
plt.title("Weibull accuracy - flaps")
plt.show()

plt.hist(tow,40)
plt.xlabel("R^2 values")
plt.title("Weibull accuracy - tower")
plt.show()


plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 24})
plt.hist(flap,bins = 50, color='gray',edgecolor='black', alpha=0.2)
plt.hist(tow,bins = 50, color='red', alpha=.7)
plt.hist(power,bins = 30, alpha=.5, edgecolor='black', color='yellow')
plt.xlabel("$R^2$ values")
plt.title("Weibull accuracy")
plt.legend(["Flaps","Tower","Power"], loc='upper left')
plt.show()


plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 24})
plt.hist(flap,bins = 50, color='gray',edgecolor='black', alpha=0.2, density=True, stacked=True)
plt.hist(tow,bins = 50, color='red', alpha=.7, density=True, stacked=True)
plt.hist(power,bins = 30, alpha=.5, edgecolor='black', color='yellow', density=True, stacked=True)
#plt.xlim(0.75, 1)
plt.xlabel("$R^2$ values")
plt.title("Weibull accuracy")
plt.legend(["Flaps","Tower","Power"], loc='upper left')
plt.show()


plt.figure(figsize=(20,10))
seaborn.histplot(flap,stat='probability', color='gray',edgecolor='black', alpha=0.2, bins=60)
seaborn.histplot(tow,stat='probability', color='red', alpha=.7, bins=60)
seaborn.histplot(power,stat='probability', alpha=.5, edgecolor='black', color='yellow', bins=20)
#plt.xlim(0.75, 1)
plt.xlabel("$R^2$ value")
plt.ylabel("Probability")
plt.title("Weibull $R^2$-distribution")
plt.legend(["Flaps","Tower","Power"], loc='upper right')
plt.show()

