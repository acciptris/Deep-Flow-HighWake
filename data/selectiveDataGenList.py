import os, sys
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime

data_dir = ["./test_uniform_unknown_135/"]
negative_percent_threshold = 0.0

if len(sys.argv)>=3:
    data_dir = sys.argv[1]
    data_dir = [data_dir]
    negative_percent_threshold = float(sys.argv[2])

print("selectiveDataGenList\n",data_dir,negative_percent_threshold)

files = []
for directory in data_dir:    
    files = files + [directory+f for f in os.listdir(directory)]
if len(files)==0:
	print("error - no data found ")
	exit(1)
files.sort()

airfoil_list = []
u_list = []
v_list = []
vel_list = []
angleOfAttack_list = []
negativeUPercent_list = []
negativeUPercentAboveThreshold_list = []
file_name_list = []

start_time = time.time()
for file_path in files:

    with np.load(file_path) as npfile:
        data = npfile['a']

        u_field = data[4]
        u_field = u_field.flatten()
        
        negative_percent = 0
        for u in u_field:
            if u<0:
                negative_percent += 1
        negative_percent /= len(u_field)

        negativeUPercent_list.append(negative_percent)

        if(negative_percent >= negative_percent_threshold or negative_percent_threshold == 0.0):
            file_name = file_path.rpartition('/')[2]
            name_split = file_name.split(sep='_')
            airfoil_name = name_split[0] + '.dat'
            u_vel = float(name_split[1])/100
            v_vel = float(name_split[2][:-4])/100
            
            negativeUPercentAboveThreshold_list.append(negative_percent)
            airfoil_list.append(airfoil_name)
            u_list.append(u_vel)
            v_list.append(v_vel)
            file_name_list.append(file_path)
    
    if(files.index(file_path) % 1000 == 0):
        ind = files.index(file_path)
        time_taken = time.time() - start_time
        #print(ind)
        est_time = (time_taken / (ind + 1)) * (len(files)-ind-1)
        print("ETC: {}".format(datetime.fromtimestamp(time.time()+est_time)))

for i in range(len(airfoil_list)):
    velocity = (u_list[i]**2+v_list[i]**2)**0.5
    vel_list.append(velocity)
    angleOfAttack = math.degrees(math.atan(v_list[i]/u_list[i]))
    angleOfAttack_list.append(angleOfAttack)

if len(data_dir) > 1:
    dir_name_split = '/data_combined/'.split(sep='/')
else:
    dir_name_split = data_dir[0].split(sep='/')
directory_characteristic_name = ""
for i in range(1,len(dir_name_split)-1):
    directory_characteristic_name += dir_name_split[i] + "_"

np.savetxt(directory_characteristic_name+"negativeUPercent.txt",negativeUPercent_list)

data_dict = {'airfoil':airfoil_list,'u_vel':u_list,'v_vel':v_list,'velocity':vel_list,'angleOfAttack':angleOfAttack_list,'negativePercent':negativeUPercentAboveThreshold_list,'fileName':file_name_list}
df = pd.DataFrame(data=data_dict)
df.to_pickle(directory_characteristic_name+str(negative_percent_threshold)+"_pdDataframe.pkl")

