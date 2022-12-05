# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:05:53 2022

@author: Marc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 14:27:06 2022

@author: Marc
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import weibull_min
import scipy as s
from reliability.Fitters import Fit_Weibull_2P
from reliability.Probability_plotting import plot_points


k1=7.930070359182593
a1=9656.52484053884

k2=7.7712393
a2=9398.578

RES1=0
TOT1=0
RES2=0
TOT2=0

y = np.array([
    7699.271070784525,
    7740.269054974002,
    7792.14778478944,
    7808.586747854407,
    7954.197073487228,
    7975.974468670362,
    7989.286509102337,
    8256.870100611395,
    8362.814826518485,
    8711.881375041425,
    8775.956382663848,
    8899.73320760528,
    9146.547163790641,
    9329.481957013884,
    10265.232376378493,
    10368.976291200503,
    10523.515211593622,
    10621.788831689617,
    10721.91808284098,
    10802.129714364894,
    11489.49052764985])

shape, loc, scale = weibull_min.fit(y, floc=0)


print("weibull")
print(shape)
print(scale)

x = np.arange(1/np.size(y),1+1/np.size(y),1/np.size(y))

print(weibull_min.fit(y, floc=0))

# =============================================================================
# plt.figure(figsize=(20,10))
# weibull_fit = Fit_Weibull_2P(failures=y,show_probability_plot=False,print_results=False)
# weibull_fit.distribution.SF(label='Fitted Distribution',color='steelblue')
# plot_points(failures=y,func='SF',label='failure data',color='red',alpha=0.7)
# plt.xlim(4000,13000)
# plt.legend()
# plt.show()
# =============================================================================

for u in range(np.size(y)):
    RES1 = RES1 + abs( ((1-math.exp(-(y[u]/a1)**k1)) -  x[u]))
    TOT1 = TOT1 + abs(x[u]-0.5)
    
    RES2 = RES2 + abs( ((1-math.exp(-(y[u]/a2)**k2)) -  x[u]))
    TOT2 = TOT2 + abs(x[u]-0.5)
    
R1=((TOT1**2-RES1**2)/TOT1**2)
R2=((TOT2**2-RES2**2)/TOT2**2)

weibull_1 = np.array([])
weibull_2 = np.array([])
for i in range(np.size(y)):
    weibull_1=np.append(weibull_1, (1-math.exp(-(y[i]/a1)**k1)))
    weibull_2=np.append(weibull_2, (1-math.exp(-(y[i]/a2)**k2)))
    
print("R^2 weibull 1 = " + str(np.corrcoef(x,weibull_1)[0,1]**2))    
print("R^2 weibull 2 = " + str(np.corrcoef(x,weibull_2)[0,1]**2))   

print(x)
print(weibull_1)

print("R**2_1 = " + str(R1))
print("R**2_2 = " + str(R2))


x1 = np.arange(y[0]-100,y[np.size(y)-1]+100,1)

x2=np.array([])
x3=np.array([])
for i in range(np.size(x1)):
    x2 = np.append(x2,1-math.exp(-(x1[i]/a1)**k1))
    x3 = np.append(x3,1-math.exp(-(x1[i]/a2)**k2))

plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 22})
plt.plot(x1,x2, color='black', linewidth=6.0)
plt.plot(x1,x3, color='blue', linewidth=6.0)
plt.scatter(y,x, color='#990000', s=200)
plt.title("Example of Weibull fit (CDF) - Power")
plt.legend(["Weibull fit", "Network prediction", "Mean values"])
plt.xlabel("Power")
plt.ylabel("Cumulative distribution")
plt.annotate("$R^2$ = {:.3f}".format(R2), (8800, 0.75))
plt.annotate("$R^2$ = {:.3f}".format(R1), (10000, 0.5))

plt.show()