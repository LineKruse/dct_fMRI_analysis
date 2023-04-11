#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:17:07 2022

@author: sirid & emma - adapted by Line 
"""
import pandas as pd
from warnings import filterwarnings
import glob
import re
import os
from itertools import chain

os.chdir('/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/')
from warnings import filterwarnings

#To import stormdb-python module directly from the folder (pip install not working)
import sys
sys.path.append('/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/stormdb-python')
import stormdb

# define subject list: 'subjects'
#data_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS' # define path to logfiles folder (including only relevant logfiles)
data_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS' # define path to logfiles folder (including only relevant logfiles)
search_str = data_path + '/sub*' # literally just adding '/*.csv' to data_path
file_list = glob.glob(search_str) # list of all logfiles (including paths)

regex = re.compile(r'\d+$')
sub_list = [regex.findall(sub) for sub in file_list]
sub_list = list(chain.from_iterable(sub_list))

#sub_list = ['0025','0067'] # just for testing 

# define project name and path to current wd
proj_name = 'MINDLAB2022_MR-semantics-of-depression' 
qsub_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line' # wdir
    
### CLUSTERBATCH
from stormdb.cluster import ClusterBatch

cb = ClusterBatch(proj_name)

for subject in sub_list:
    submit_cmd = 'python /projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/run_fmriprep_all.py '+ subject # remembter the space between .py and ' !!!!! 
    cb.add_job(cmd=submit_cmd, queue='highmem.q',n_threads = 6, cleanup = False)

cb.submit()

"""
### CLUSTERJOB 
from stormdb.cluster import ClusterJob

for subject in subjects:

    cmd = 'python run_fmriprep_all.py ' + subject # for christ sake please remember ' ' after the .py
    cj = ClusterJob(cmd=cmd,
                    queue='highmem.q',
                    n_threads=6,
                    job_name='py-wrapper_' + subject,
                    proj_name=proj_name,
                    working_dir=qsub_path)
    cj.submit()
"""