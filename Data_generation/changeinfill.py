import numpy as np
import os

Directory_names = "/work3/s193961/WorkingData/"

casefolders = np.array(os.listdir(Directory_names))
for i in range(np.size(casefolders)):
    WTfolders = np.array(os.listdir(Directory_names + casefolders[i]))
    for j in range(np.size(WTfolders)):
        T2_folders_temp = np.array(os.listdir(Directory_names + casefolders[i] + "/" + WTfolders[j]))
        T2_folders = np.array([])
        print(T2_folders_temp)
        for k in range(np.size(T2_folders_temp)):
            if (T2_folders_temp[k][0] == "y"):
                T2_folders = np.append(T2_folders, T2_folders_temp[k] )
                
        for k in range(np.size(T2_folders)):

            #print(T2_folders)
            print(T2_folders[k])
            print(int(T2_folders[k][1:4]))
            print(int(T2_folders[k][6:8])/10)
            print(int(T2_folders[k][10:12])/100)
            yaw = int(T2_folders[k][1:4])
            pitch = int(T2_folders[k][6:8])/10
            omega = int(T2_folders[k][10:12])/100
    
            infilldir = Directory_names + casefolders[i] + "/" + WTfolders[j] + "/" + T2_folders[k] + "/base/Infile.pas"
            reading_file = open(infilldir, "r", encoding="ISO-8859-1")
            new_file_content = ""
            for line in reading_file:
                stripped_line = line.strip()
                new_line = stripped_line.replace("yawchange", str(yaw))
                new_file_content += new_line +"\n"
            reading_file.close()
            
            writing_file = open(infilldir, "w", encoding="ISO-8859-1")
            writing_file.write(new_file_content)
            writing_file.close()
            
            
            reading_file = open(infilldir, "r", encoding="ISO-8859-1")
            new_file_content = ""
            for line in reading_file:
                stripped_line = line.strip()
                new_line = stripped_line.replace("pitchchange", str(pitch))
                new_file_content += new_line +"\n"
            reading_file.close()
            
            writing_file = open(infilldir, "w", encoding="ISO-8859-1")
            writing_file.write(new_file_content)
            writing_file.close()
            
            reading_file = open(infilldir, "r", encoding="ISO-8859-1")
            new_file_content = ""
            for line in reading_file:
                stripped_line = line.strip()
                new_line = stripped_line.replace("omegachange", str(omega))
                new_file_content += new_line +"\n"
            reading_file.close()
            
            writing_file = open(infilldir, "w", encoding="ISO-8859-1")
            writing_file.write(new_file_content)
            writing_file.close()
            
