#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import os
os.chdir('/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/')
from warnings import filterwarnings

#To import stormdb-python module directly from the folder (pip install not working)
import sys
sys.path.append('/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/stormdb-python')
import stormdb
from stormdb.cluster import ClusterBatch

### OBS see Gemma's original script for the original full version of this script "scripts/bachelor_scripts/gemmas_script.py"

project = 'MINDLAB2022_MR-semantics-of-depression' # setting the project we are working on 
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2' # (we don't know why this is here)

cb = ClusterBatch(project) # making our ClusterBatch object

# submitting a file to cluster
#Step 1) script = 'raw_to_scratch2.py'
#Step 2) script = 'check_files_dimensions.py' 
#Step 2) script = 'logfilesRaw_to_scratch.py'
#Step 3) script = 'editSeriesDescription_phaseFiles.py'
#Step 4) script = 'bidscoin_converter.py'
#Step 5) script = 'convert_logfiles_to_BIDS.py" 
#Step 6) script = 'add_events_jsonFiles_allSubjects.py" 
#Step 7) run_fmriprep_parallel.py (run that script directly from terminal - will subject to cluster seaprately for each subject)
script = 'analysis/clusterInfo_from_permutation_image.py'
#queue = 'short.q' #For short jobs, fast queue 
#queue = 'highmem.q' #For big jobs, runtime = infinite 
queue = 'highmem_short.q' #For big (short) jobs, runtime = 12 hours 

path = os.path.join('python /projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/'+script)
submit_cmd = path # choosing the script to be submitted
cb.add_job(cmd=submit_cmd, queue=queue,n_threads=12,cleanup=False) # cmd = job to be added, queue = the cluster queue (see wiki), n_threads = number of servers to use
cb.submit()





"""
# (fix and) include the chunk below when hacing to run on all subjects

subNs = [79] #range(11,21) #25,26,27,29,31

cb = ClusterBatch(project)
for s in subNs:
    #sub = sub_codes[s-1]
    submit_cmd = 'python {}APR6h_decoding_localizer_source_gemma.py {}'.format(script_dir,s)
    cb.add_job(cmd=submit_cmd, queue='all.q',n_threads = 8, cleanup = False)
"""
