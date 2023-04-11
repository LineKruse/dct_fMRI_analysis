#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:17:07 2022

@author: line - adapted from sirid & emma
"""
import pandas as pd
from warnings import filterwarnings
import os

# define subject list: 'subjects'
dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/'
sublist = os.listdir(dir)
sublist = [sub for sub in sublist if sub.startswith('s')]
len(sublist)

#Subjects to exclude as they fell asleep - had too many missing repsonses in several runs 
exclude = ['sub-0020','sub-0038','sub-0059']
sublist = [sub for sub in sublist if not sub in exclude]

#sublist = ['sub-0053', 'sub-0050', 'sub-0057', 'sub-0058', 'sub-0018', 'sub-0021', 'sub-0027']

#Sub-0053 (py-wrapper 6456705): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0050: (py-wrapper 6456706): ValueError: array must not contain infs or NaNs
#Sub-0057: (py-wrapper 6456707): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0058: (py-wrapper 6456708): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0018: (py-wrapper 6456709): missing event file for block 1, missing all physiological files, incomplete BOLD file for run 1
#Sub-0021: (py-wrapper 6456710): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0027: (py-wrapper 6456711): no mask provided for intersection - fMRIprep failed - no files 


# define project name and path to current wd
proj_name = 'MINDLAB2022_MR-semantics-of-depression' 
qsub_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/analysis' # wdir
    
### CLUSTERBATCH
#To import stormdb-python module directly from the folder (pip install not working)
import sys
sys.path.append('/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/stormdb-python')
import stormdb
from stormdb.cluster import ClusterBatch

cb = ClusterBatch(proj_name)

for subject in sublist:
    submit_cmd = 'python /projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/analysis/first_level_models.py '+ subject # remembter the space between .py and ' !!!!! 
    cb.add_job(cmd=submit_cmd, queue='highmem.q',n_threads = 6, cleanup = False)

cb.submit()

"""
### CLUSTERJOB 
from stormdb.cluster import ClusterJob

for subject in subjects:

    cmd = 'python first_level_fit_function.py ' + subject # for christ sake please remember ' ' after the .py
    cj = ClusterJob(cmd=cmd,
                    queue='highmem.q',
                    n_threads=6,
                    job_name='py-wrapper_' + subject,
                    proj_name=proj_name,
                    working_dir=qsub_path)
    cj.submit()
"""