
import os
import numpy as np
from wetb.fatigue_tools.fatigue import eq_load
import math

print("##########################################################################################")
print("#                             Running mean calculater3                                   #")
print("##########################################################################################")

def get(var):
    index = np.where(Sensors==var)
    return dataraw[:,index[0]]

def get2(var):
    index = np.where(Sensors==var)
    return datapower2[:,index[0]]


Directory_working = "/work3/s193961/WorkingData/"
Directory_output = "/zhome/87/3/146402/Desktop/Bachelor/OutputFiles/"
Directory_outputDEL = "/zhome/87/3/146402/Desktop/Bachelor/OutputFilesDEL2/"
Directory_Matlab = "/zhome/87/3/146402/Desktop/Bachelor/Matlab/"

Sensorstemp=open(Directory_Matlab+"SENSORnames").read().splitlines()
Sensors=np.array(Sensorstemp)

#######################    SUB FOLDERS   ################################
casefolders = np.array(os.listdir(Directory_working))
for u in range(np.size(casefolders)):
    Directory_distance = Directory_working + casefolders[u] + "/"
    distancefolders = np.array(os.listdir(Directory_distance))
    for t in range(np.size(distancefolders)):
        Directory_WT = Directory_distance + distancefolders[t] + "/"
        alldownstreamfolders = np.array(os.listdir(Directory_WT))
        downstreamfolders = np.array([])
        for e in range(np.size(alldownstreamfolders)):
            if (alldownstreamfolders[e][0]=="y"):
                downstreamfolders = np.append(downstreamfolders,alldownstreamfolders[e])

        for r in range(np.size(downstreamfolders)):
            Directory_les = Directory_WT + downstreamfolders[r] + "/"


            MeanPower= np.array([])
            MeanDELflap= np.array([])
            MeanDELtow= np.array([])

            datafolders = []
            folders = np.array(os.listdir(Directory_les))
            for j in range(len(folders)):
                if "LES" in folders[j]: 
                    datafolders.append(folders[j])

            for i in range(len(datafolders)):
                datafiles = []
                files = np.array(os.listdir(Directory_les + datafolders[i]))
                for k in range(len(files)):
                    if "turb.0" in files[k] and ".tim" in files[k]:
                        datafiles.append(files[k])

                dataraw=np.loadtxt(Directory_les + datafolders[i] + "/" + datafiles[0])
                datapower2 = np.loadtxt("/work3/s193961/NoWT1/WT02/" + casefolders[u] + "/" + datafolders[i] + "/" + datafiles[0])
                Power = get("Power")
                Power = Power[~np.isnan(Power)]
                Powerfirst = get2("TorqueN")*get2("Omega")
                Powerfirst = Powerfirst[~np.isnan(Powerfirst)]
                print(casefolders[u])
                print(np.mean(Powerfirst))

                total_power = np.mean(Power) + np.mean(Powerfirst)
                MeanPower = np.append(MeanPower,total_power)
                print(casefolders[t])
                print(downstreamfolders[r])
                print(np.mean(Powerfirst))
                print(np.mean(Power))

                
                flap1 = get("FlapM1")
                flap1 = flap1[~np.isnan(flap1)]
                Loadsflap1 = eq_load(flap1, m=10, neq=6000)[0][0]
                
                flap1_first = get2("FlapM1")
                flap1_first = flap1_first[~np.isnan(flap1_first)]
                Loadsflap1_first = eq_load(flap1_first, m=10, neq=6000)[0][0]
                
                flap1_total = Loadsflap1_first + Loadsflap1
                MeanDELflap= np.append(MeanDELflap,flap1_total)

                flap2 = get("FlapM2")
                flap2 = flap2[~np.isnan(flap2)]
                Loadsflap2 = eq_load(flap2, m=10, neq=6000)[0][0]
                
                flap2_first = get2("FlapM2")
                flap2_first = flap2_first[~np.isnan(flap2_first)]
                Loadsflap2_first = eq_load(flap2_first, m=10, neq=6000)[0][0]
                
                flap2_total = Loadsflap2_first + Loadsflap2
                MeanDELflap = np.append(MeanDELflap,flap2_total)

                flap3 = get("FlapM3")
                flap3 = flap3[~np.isnan(flap3)]
                Loadsflap3 = eq_load(flap3, m=10, neq=6000)[0][0]
                
                
                flap3_first = get2("FlapM3")
                flap3_first = flap3_first[~np.isnan(flap3_first)]
                Loadsflap3_first = eq_load(flap3_first, m=10, neq=6000)[0][0]
                
                flap3_total = Loadsflap3_first + Loadsflap3
                MeanDELflap = np.append(MeanDELflap,flap3_total)

                tow_y = get("BtowMyTb")
                tow_y = tow_y[~np.isnan(tow_y)]
                tow_z = get("BtowMzTb")
                tow_z = tow_z[~np.isnan(tow_z)]
                
                tow_y_first = get2("BtowMyTb")
                tow_y_first = tow_y_first[~np.isnan(tow_y_first)]
                tow_z_first = get2("BtowMzTb")
                tow_z_first = tow_z_first[~np.isnan(tow_z_first)]
                
                tow = np.array([])
                tow_first = np.array([])
                for i in range(np.size(tow_y)):
                    tow = np.append(tow, math.sqrt(tow_y[i]**2+tow_z[i]**2))
                    tow_first = np.append(tow_first, math.sqrt(tow_y_first[i]**2+tow_z_first[i]**2))

                Loadstow = eq_load(tow, m=5, neq=6000)[0][0]
                Loadstow_first = eq_load(tow_first, m=5, neq=6000)[0][0]
                loadstow_total = Loadstow + Loadstow_first
                

                MeanDELtow = np.append(MeanDELtow,loadstow_total)


            MeanPower=np.sort(MeanPower)
            MeanDELflap = np.sort(MeanDELflap)
            MeanDELtow = np.sort(MeanDELtow)

            if (casefolders[u][13]=="n"):
                outputname = "y1_" + casefolders[u][30:32] + "_o1_" + casefolders[u][23:26] + "_p1_-" + casefolders[u][15:17]
            else:
                outputname = "y1_" + casefolders[u][29:31] + "_o1_" + casefolders[u][22:25] + "_p1_" + casefolders[u][13:16]
            
            outputname = outputname + "_s_" + distancefolders[t][2:4] + "_y2_" + downstreamfolders[r][1:4] + "_p2_" + downstreamfolders[r][6:8] + "_o2_" + downstreamfolders[r][10:12]

            
            f = open(Directory_output + outputname,"w+")
            for i in range(len(MeanPower)):
                f.write(str(MeanPower[i]) + "\n")
            f.close()

            
            f = open(Directory_outputDEL + "DELFlap/" + outputname,"w+")
            for i in range(len(MeanDELflap)):
                f.write(str(MeanDELflap[i]) + "\n")
            f.close()

            f = open(Directory_outputDEL + "DELBtow/" + outputname,"w+")
            for i in range(len(MeanDELtow)):
                f.write(str(MeanDELtow[i]) + "\n")
            f.close()


