#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Apr 11 13:33:48 2022

@author: Line (adapted from Emma and Sirids script)

Insert the line below to run the script
python /projects/MINDLAB2022_MR-semantics-of-depression/scripts/bachelor_scripts/controls_to_scratch_bachelor_script.py
"""

user = 'line'
project = 'MINDLAB2022_MR-semantics-of-depression'
Modality='MR'
Task='DCT'

import os
import shutil
import glob
import fnmatch
import re

# series_number har vi i en CSV fil
# så skal vi matche alle mapper i /raw/, der starter med series_number fra den liste 

# list of participants to exclude # including 38 subjects
exclude=(['0001','0002','0003','0004','0005','0006','0055'])
BOLD_phase_files = (['008','012','016'])

#Define paths 
user_path = '/users/line'
project_path = '/projects/MINDLAB2022_MR-semantics-of-depression'
raw_folder = '/projects/MINDLAB2022_MR-semantics-of-depression/raw'
scr_raw_folder = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw'
#bids_folder = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS'

#Dete any raw folder on scratch 
#if os.path.exists(scr_raw_folder):        # hvis mappe scr_raw_folder:
#    shutil.rmtree(scr_raw_folder, ignore_errors=True)  # delete entire directory tree, errors resulting from failed removals will be ignored
    #os.makedirs(scr_raw_folder)    # create directory 
test = os.listdir(raw_folder)
#Move raw files to scratch 
sub_list=os.listdir(raw_folder)
new_sub_list = []
for sub in sub_list:
    #subject level
    sub_raw_dir= os.path.join(raw_folder, sub)
    sub_scr_raw_dir= os.path.join(scr_raw_folder, sub)
    check_path = scr_raw_folder + '/sub-'+sub #To check if subject has already been copied 
    if not sub in exclude:
        if not os.path.exists(check_path):
            print('copying subject: ', sub)  
            shutil.copytree(sub_raw_dir,sub_scr_raw_dir)#If not already copied, copy subject folder
            new_sub_list.append(sub) #Append subject to new_sub_list - rest of the script operates only on these 
        else: print(sub, " already copied - skipping")
    else: print(sub, " in exclude - skipping")
print('Raw folder copied')

#Change file names (???) 
#sub_list=os.listdir(scr_raw_folder)  # defines sub_list as a list containing the names of the entries in the directory scr_raw_folder. if we didn't have the "cd_cmd" command and "os.system(cd_cmd)" from above, we should add the full path to the scr_raw_folder

for sub in new_sub_list: 
    #subject level
    subdir= os.path.join(scr_raw_folder, sub) # joining 2 path components. the os.path.join() concatenates  scr_raw_folder and sub in a path: 
    if sub in exclude:  # * why this again?
        print('removing subject: ', sub)
        shutil.rmtree(subdir, ignore_errors=True)
        
    else:
        subdirnew= os.path.join(scr_raw_folder, 'sub-'+sub) # adding sub number to folder name (so it says sub-0009 and not 0009) 
        #rename subdirs to fit with bidsmapper 
        os.rename(subdir,subdirnew) # renaming directory (but subdir still withou 00xx)
        subdir=subdirnew # * why this step? (this time subdir is defined as sub-00xx)
        sesdir_list=os.listdir(subdir) # defines sesdir_list as a list containing the names of the entries in the directory subdir. subdir is scr_raw_folder/sub-00xx
        #initiate numerator
        sesnum=1
        for sesdirs in sesdir_list:
            #Session level
       
            sesdir= os.path.join(subdir, sesdirs)

            #print(sesdir)
            typedir_list=os.listdir(sesdir)
            for typedirs in typedir_list:
                #Type level (MR/SR/MEG)
                typedir= os.path.join(sesdir, typedirs)
                #Only keep folders from expected modality
                if typedirs !=Modality:
                    tempdir= os.path.join(sesdir, typedir)
                    print('removing: ', typedir)
                    shutil.rmtree(tempdir, ignore_errors=True)
                    typedir_list2=os.listdir(sesdir)
                    if len(typedir_list2)==0:
                        print('removing emty ses: ', sesdir)
                        shutil.rmtree(sesdir, ignore_errors=True)
                else:    
                    scandir_list=os.listdir(typedir)
           
                    for scandirs in scandir_list:
                        #Remove the Physiolog folders * maybe we want to keep physiolog 
                        #if 'PhysioLog' in scandirs: # remove if we don't want to remove physiolog
                        #    print('removing: ', scandirs) # remove if we don't want to remove physiolog
                        #    tempdir= os.path.join(typedir, scandirs) # remove if we don't want to remove physiolog
                        #    shutil.rmtree(tempdir, ignore_errors=True) # remove if we don't want to remove physiolog 
                        #else:
                            #Rename folders to fit conventions
                            #scandirs_new=scandirs
                            #scandirs_new=scandirs_new.replace('T1_mprage','T1w') # 
                            #scandirs_new=scandirs_new.replace('EPI_sequence_words','task-faceword_bold') #  
                            
                            #Scan level (e.g.scout/BOLD)
                            #scandir_old= os.path.join(typedir, scandirs)
                        scandir= os.path.join(typedir, scandirs)
                            #os.rename(scandir_old,scandir)
                            # Move scans up one level
                        shutil.move(scandir,sesdir)
                    os.rmdir(typedir)
            if os.path.exists(sesdir):
                sesdir_old=sesdir
                #rename sessions to number
                sesdir= os.path.join(subdir, 'ses-'+ '00'+str(sesnum))
                #rename subdirs
                os.rename(sesdir_old,sesdir)
                sesnum=sesnum+1
                scandir_list=os.listdir(sesdir)
                for scandirs in scandir_list:
                    
                    #Remove functional BOLD_phase files (because they have the same name as the functional BOLD files)
                    #if scandirs.startswith(BOLD_phase_files):
                    if any (scandirs.startswith(id) for id in BOLD_phase_files):
                        scandirs_old = os.path.join(sesdir, scandirs)
                        scandirs_new = os.path.join(sesdir, scandirs+"_phase")
                        os.rename(scandirs_old, scandirs_new)
                        scandir=scandirs_new
                    else: 
                        scandir=(os.path.join(sesdir,scandirs))

                    print(scandir)
                    filedir_list=os.listdir(scandir)
                    filedir= os.path.join(scandir, filedir_list[0])
                    files=os.listdir(filedir)
                    #Move all files one directory up
                    for file in files:
                        filepath= os.path.join(filedir, file)
                        shutil.move(filepath,scandir)
                    #Remove old directory
                    os.rmdir(filedir)


