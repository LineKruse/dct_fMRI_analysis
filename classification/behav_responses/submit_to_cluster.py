#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
os.chdir('/users/line/dct_fMRI_analysis/classification/')
from warnings import filterwarnings

#To import stormdb-python module directly from the folder (pip install not working)
import sys
sys.path.append('/users/line/dct_fMRI_analysis/classification/stormdb-python')
import stormdb
from stormdb.cluster import ClusterBatch

### OBS see Gemma's original script for the original full version of this script "scripts/bachelor_scripts/gemmas_script.py"

project = 'MINDLAB2022_MR-semantics-of-depression' # setting the project we are working on 
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2' # (we don't know why this is here)

cb = ClusterBatch(project) # making our ClusterBatch object

# submitting a file to cluster
script = 'step1_fit_flm.py'
#queue = 'short.q' #For short jobs, fast queue 
queue = 'highmem.q' #For big jobs, runtime = infinite 
#queue = 'highmem_short.q' #For big (short) jobs, runtime = 12 hours 

path = os.path.join('python /users/line/dct_fMRI_analysis/classification/behav_responses/'+script)
submit_cmd = path # choosing the script to be submitted
cb.add_job(cmd=submit_cmd, queue=queue,n_threads=12,cleanup=False) # cmd = job to be added, queue = the cluster queue (see wiki), n_threads = number of servers to use
cb.submit()

