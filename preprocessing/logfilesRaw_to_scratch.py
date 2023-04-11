from fileinput import filename
import pandas as pd
import numpy as np
import glob
import csv # for loading the dictionary (include_dict.csv)
import shutil
import os

#Copy logfiles from /aux to project scratch folder 
data_orig = '/aux/MINDLAB2022_MR-semantics-of-depression/DCT-main/data'
data_dest = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/logfiles_raw'

copiedFiles = os.listdir(data_dest)
newFiles = os.listdir(data_orig)
toCopyFiles = list(set(newFiles)-set(copiedFiles))

#Create directory. If folder already exists, delete first 
#if os.path.exists(data_dest):      
#    shutil.rmtree(data_dest, ignore_errors=True)  

#Copy all files to the destination directory 
for file in toCopyFiles: 
    orig_dir = os.path.join(data_orig +"/"+ file)
    dest_dir = os.path.join(data_dest + "/"+file)
    shutil.copy(orig_dir, dest_dir)
