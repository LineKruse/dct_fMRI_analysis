# -*- coding: utf-8 -*-

# import packages

from enum import unique
import pandas as pd
import numpy as np
import glob
import csv # for loading the dictionary (include_dict.csv)
import os
# load logfiles 

output_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/'
data_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/logfiles_raw' # define path to logfiles folder (including only relevant logfiles)
search_str = data_path + '/*.csv' # literally just adding '/*.csv' to data_path
file_list = glob.glob(search_str) # list of all logfiles (including paths)
#file_list = ['/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/logfiles_raw/130.csv', '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/logfiles_raw/131.csv', '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/logfiles_raw/132.csv']

exclude = "pilot"
exclude2 = "test"
for file in file_list: 
    if exclude in file: 
        print("Removing " + file)
        file_list.remove(file)
    if exclude2 in file: 
        print("Removing " + file)
        file_list.remove(file)

# loop through each file

for i in range(len(file_list)):
    print("Running subject " + str(i) + " of " + str(len(file_list)))
    
    file_name = file_list[i] # extracting filename (including path)
    #print(file_name)
    logfile = pd.read_csv(file_name) # load the logfile (converting them into padas dataframe)

    # add 'task' column
    task = np.repeat('DCT', len(logfile)) # a list that repeats 'task_name' as many times as the length of the logfile
    logfile['scan_task'] = task # creating task-column to dataframe from list


    # add 'series' column (using include_dict)
    csv_filename = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/SUBID_to_series_dict.csv'
    reader = csv.reader(open('/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/SUBID_to_series_dict.csv', 'r'))

    d = {}
    for row in reader:
        k, v = row
        d[k] = v
        keys_values = d.items()
        dict_sub = {str(key):int(value) for key, value in keys_values}
    
    #Check if SUBID is below or above 100 (indicating group) - if lower, add "0" or "00" in front, and group = 0, if higher group = 1 
    SUBID = str(logfile['ID'][0])
    if len(SUBID)<3:
        if len(SUBID)<2:
            newSUBID = str('00'+SUBID)
            logfile['ID'] = np.repeat(newSUBID, len(logfile))
        else: 
            newSUBID = str('0'+SUBID)
            logfile['ID'] = np.repeat(newSUBID, len(logfile))
        group = np.repeat(0,len(logfile))
    else: 
        group = np.repeat(1,len(logfile))
    
    logfile['group'] = group
    
    #Extract series name matching subject ID - used to rename files 
    series = np.repeat(dict_sub[str(logfile['ID'][0])], len(logfile)) # using the dictionary to create a list with series number, inserting their sub-ID as dictionary KEY
    logfile['series'] = series # creating 'series'-column to dataframe from list

    # renaming 'onset' column (BIDS conventions)
    logfile.rename(columns={'stim_onset': 'onset'}, inplace=True)
    
    # creating 'duration' column (BIDS conventions)
    duration_list = np.repeat(1, len(logfile))
    # for i in range(len(logfile)-1):
    #     some_number = logfile['onset'].iloc[i+1]-logfile['onset'].iloc[i] # calculating duration manually from onset time
    #     duration_list.append(some_number)
    #duration_list.insert(len(duration_list), 6.05) # potentially replace '6.05' with 'duration_list[-1]' # adding the last duration time manually as there's no info about it in log-file 
    logfile['duration'] = duration_list
    
    # changing positions of columns (onset should be first and duration second) (BIDS conventions)
    logfile=logfile.iloc[:, [9,14,0,1,2,3,4,5,6,7,8,10,11,12,13]] # you have to print the logfile you just created to get the index position of the different columns 

    # splitting dataframe to three (dependent on run) and saving as .tsv files

    #First check if all three blocks are present in logfile 
    nblocks = logfile.block.unique()
    fullSet = [1,2,3]
    missing = list[set(fullSet)-set(nblocks)]

    if len(nblocks)<3:
        print("Block number " + str(missing) + " is missing for subject " + str(logfile['ID'][0]))

    for k in nblocks: # looping through number 1-3 (because we have run: 1, 2, 3)
        #print(k)
        logfile_subset = logfile[logfile['block'] == k] # subsetting dataframe for each run
     
        # renaming the file in order with BIDS conventions 
        events_file_name = (
            'sub-'+ str('%04d' % logfile['series'].iloc[0]) + 
            '_task-' + str(logfile_subset['scan_task'].iloc[0]) + 
            '_run-'+ str(logfile_subset['block'].iloc[0]) + 
            '_events.tsv')
        
        # saving the file as .tsv in the BIDS folder
        BIDS_path = (
            output_path + # this path should lead to the BIDS folder with the BIDS converted data
            'sub-' + str('%04d' % logfile['series'].iloc[0]) +
            '/func/')
        events_file_full_name = (BIDS_path + events_file_name) 
        logfile_subset.to_csv(events_file_full_name, na_rep='n/a', index=False, sep="\t", encoding='utf-8') # forcing empty cells to be 'n/a' (BIDS convention), index=False is just to NOT have row names


"""
notes:
- obs* notice the difference in indexing between the original dataframe (logfile) logfile['sub'][0] and the subsettet version (logfile_subset)

- if you want to check the dataframe, BEFORE SUBSETTING then use this chunk 

    events_file_name=('sub-'+ str('%04d' % logfile['sub'][0]) + '_task-' + str(logfile['task'][0]) + '_events.tsv') # test*
    ouput_path='/projects/MINDLAB2022_MR-semantics-of-depression/scratch/bachelor_scratch/event_files_2_subjects/' # test*
    events_full_file_name=(ouput_path+events_file_name) # test*
    logfile.to_csv(events_full_file_name, index=False, sep="\t", encoding='utf-8') # test*
"""