import os
import numpy as np
from scipy.stats import weibull_min
import math

Directory_data = "/zhome/87/3/146402/Desktop/Bachelor/OutputFilesDEL2/"
Directory_output = "/zhome/87/3/146402/Desktop/Bachelor/Training/"

datafolders=np.array(os.listdir(Directory_data))
for k in range(np.size(datafolders)):


    datafiles=np.array(os.listdir(Directory_data + datafolders[k]))
    print(datafiles)
    inputvec = np.array([])
    outputvec = np.array([])
    R2output = np.array([])

    for i in range(np.size(datafiles)):
        print("i = " + str(i))
        print(datafiles[i])
        print("k = " + str(k))
        print(datafolders[k])
        data = np.loadtxt(Directory_data+datafolders[k] + "/" + datafiles[i])
        shape, loc, scale = weibull_min.fit(data, floc=0)

        for j in range(len(datafiles[i])):
            if (datafiles[i][j:j+2]=="y1"):
                yaw1=datafiles[i][j+3:j+5]
            if (datafiles[i][j:j+2]=="p1"):
                pitch1=datafiles[i][j+3:j+6]
            if (datafiles[i][j:j+2]=="o1"):
                omega1=datafiles[i][j+3:j+6]
            if (datafiles[i][j]=="s"):
                distance=datafiles[i][j+2:j+4]
            if (datafiles[i][j:j+2]=="y2"):
                yaw2=datafiles[i][j+3:j+6]
            if (datafiles[i][j:j+2]=="p2"):
                pitch2=datafiles[i][j+3:j+5]
            if (datafiles[i][j:j+2]=="o2"):
                omega2=datafiles[i][j+3:j+5]

        inputstr = yaw1 + "," + pitch1 + "," + omega1 + "," + yaw2 
        inputvec = np.append(inputvec, inputstr)

        outputstr = str(shape) + "," + str(scale)
        outputvec = np.append(outputvec, outputstr)


        RES = 0
        TOT = 0
        x = np.arange(1/np.size(data),1+1/np.size(data),1/np.size(data))
        for u in range(np.size(data)):
            RES = RES + abs( ((1-math.exp(-(data[u]/scale)**shape)) -  x[u]))
            TOT = TOT + abs(x[u]-0.5)

        R2=((TOT**2-RES**2)/TOT**2)
        R2output=np.append(R2output,R2)

    f = open(Directory_output + "T_InputDEL_" + str(datafolders[k]),"w+")
    for i in range(np.size(inputvec)):
        f.write(str(inputvec[i]) + "\n")
    f.close()

    f = open(Directory_output + "T_OutputDEL_" + str(datafolders[k]),"w+")
    for i in range(np.size(outputvec)):
        f.write(str(outputvec[i]) + "\n")
    f.close()


    f = open("/zhome/87/3/146402/Desktop/Bachelor/R2values_"+ str(datafolders[k]),"w+")
    for i in range(np.size(R2output)):
        f.write(str(R2output[i]) + "\n")
    f.close()