import pandas as pd
import numpy as np
import shutil, os, utils

##################################################################################################

df_dir = "./data_combined_0.0_pdDataframe.pkl"
# src_dir = "./data_combined/"
dst_dir = "./data_uniform_27x200_141/"

sample_per_bin = 200
num_bins = 41

##################################################################################################
bin_values = np.linspace(0,0.2,num_bins)
bin_values = bin_values[:-14]
bin_values = np.append(bin_values,0.2)
bins = pd.IntervalIndex.from_arrays(bin_values[:-1],bin_values[1:],closed='left')

df = pd.read_pickle(df_dir)
df = df.assign(df_split = pd.cut(df['negativePercent'],bins))

out_df = df.groupby('df_split').sample(n=sample_per_bin)

utils.makeDirs([dst_dir])
for file in out_df.fileName:
    shutil.copy2(file,dst_dir)

os.system("python selectiveDataGenList.py " + dst_dir + " 0.0")

