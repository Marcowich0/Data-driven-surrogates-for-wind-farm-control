# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:04:31 2022

@author: marc0
"""

import numpy as np
import matplotlib.pyplot as plt


power_6R = np.loadtxt('power_increase_6R_unlocked.out', delimiter=',')
power_14R = np.loadtxt('power_increase_14R_unlocked.out', delimiter=',')
power_14R[power_14R<0]=0

loadconstrain = np.loadtxt('constraints_14R_unlocked.out', delimiter=',')

power_6R_locked = np.loadtxt('power_increase_6R_locked.out', delimiter=',')
power_6R_locked[power_6R_locked<0]=0

power_14R_locked = np.loadtxt('power_increase_14R_locked.out', delimiter=',')
power_14R_locked[power_14R_locked<0]=0

print(power_6R_locked)


plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 26})
plt.plot(loadconstrain*100,power_6R*100, color='darkred', linewidth=5.0)
plt.plot(loadconstrain*100,power_6R_locked, color='darkred', linewidth=5.0, linestyle='dashed')

plt.plot(loadconstrain*100,power_14R*100, color='darkblue', linewidth=5.0)
plt.plot(loadconstrain*100,power_14R_locked*100, color='darkblue', linewidth=5.0, linestyle='dashed')



plt.title("Posible gain as a function of load constrains")
plt.xlabel("Load constraint [%]")
plt.ylabel("Power increase [%]")
plt.xlim(0,10)
plt.ylim(0,10)
plt.grid()
plt.legend(["6R   - optimized","6R   - locked","14R - optimized","14R - locked"])
plt.show()


