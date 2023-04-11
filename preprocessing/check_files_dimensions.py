import glob
import os
import sys
sys.path.append('/users/line/miniconda3/envs/env_bidscoin3/lib/python3.10/site-packages')
import pydicom

#data_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw' # define path to logfiles folder (including only relevant logfiles)
#search_str = data_path + '/*sub*' # literally just adding '/*.csv' to data_path
#sub_list = glob.glob(search_str) # list of all logfiles (including paths)
sub_list = ['/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw/sub-0085', '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw/sub-0086', '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw/sub-0087']

#Loop through subjects and check if they have the correct number of files (=17)
for subject in sub_list:
    #Add subject's series number to subject list 
    serNum =  subject.rsplit('/', 1)[1]
    serNum = serNum.rsplit('-')[1]
    print("---------------- Subject with series number " + serNum + " -------------------")

    #Check number of files 
    path = subject + "/ses-001"
    files = os.listdir(path)
    nFiles = len(files)
    if not nFiles==17: 
        print("Found " + str(nFiles) + " files")
        print(files)
    
    #Check if all files have the correct dimensions 
    for file in files: 
        filePath = path + "/" + file

        if file.startswith("001"):
            trueN = 128 
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")
        
        if file.startswith("002"):
            trueN = 5 
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete") 

        if file.startswith("003"):
            trueN = 3
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("004"):
            trueN = 3
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("005"):
            trueN = 192
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")
        
        if file.startswith("006"):
            trueN = 2 
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("007"):
            trueN = 582
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")
        
        if file.startswith("008"):
            trueN = 582 
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("010"):
            trueN = 2
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("011"):
            trueN = 582
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("012"):
            trueN = 582
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("014"):
            trueN = 2
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")
        
        if file.startswith("015"):
            trueN = 582
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")

        if file.startswith("016"):
            trueN = 582
            nImages = len(os.listdir(filePath))
            if not nImages==trueN: 
                print("Directory " + file + " is incomplete")