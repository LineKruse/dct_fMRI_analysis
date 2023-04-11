import sys
sys.path.append('/users/line/miniconda3/envs/env_bidscoin3/lib/python3.10/site-packages')
import pydicom
import glob

"""
bold = pydicom.filereader.dcmread('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw2sub/sub-0007/ses-001/007.fMRI_MB4_p2_1.8iso_run1/PROJ0497_SUBJ0007_SER007_ACQ00001_IMG00001_057989912596.dcm')
print(bold)
#Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'M', 'MB', 'ND', 'NORM', 'MOSAIC']
#Series Description                  LO: 'fMRI_MB4_p2_1.8iso_run1'
#Protocol Name                       LO: 'fMRI_MB4_p2_1.8iso_run1'

phase = pydicom.filereader.dcmread('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw2sub/sub-0007/ses-001/008.fMRI_MB4_p2_1.8iso_run1_phase/PROJ0497_SUBJ0007_SER008_ACQ00001_IMG00001_058429312603.dcm')
print(phase)
#Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'P', 'MB', 'ND', 'MOSAIC']
# Series Description                  LO: 'fMRI_MB4_p2_1.8iso_run1'
# Protocol Name                       LO: 'fMRI_MB4_p2_1.8iso_run1'

sbref = pydicom.filereader.dcmread('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw2sub/sub-0007/ses-001/010.fMRI_MB4_p2_1.8iso_run2_SBRef/PROJ0497_SUBJ0007_SER010_ACQ00001_IMG00001_182273847984.dcm')
print(sbref)
#Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM', 'MOSAIC']
#Series Description                  LO: 'fMRI_MB4_p2_1.8iso_run2_SBRef'
# Protocol Name                       LO: 'fMRI_MB4_p2_1.8iso_run2'

phys = pydicom.filereader.dcmread('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw2sub/sub-0007/ses-001/009.fMRI_MB4_p2_1.8iso_run1_PhysioLog/PROJ0497_SUBJ0007_SER009_ACQ04798_IMG00001_773079946921.dcm')
print(phys)
#Image Type                          CS: ['ORIGINAL', 'PRIMARY', 'RAWDATA', 'PHYSIO']
#Series Description                  LO: 'fMRI_MB4_p2_1.8iso_run1_PhysioLog'
#Protocol Name                       LO: 'fMRI_MB4_p2_1.8iso_run1' """

#Loop through subjects and phase files and edit "seresDescription" in header and save again to same path 
#data_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw' # define path to logfiles folder (including only relevant logfiles)
#search_str = data_path + '/*sub*' # literally just adding '/*.csv' to data_path
#sub_list = glob.glob(search_str) # list of all logfiles (including paths)
sub_list = ['/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw/sub-0085','/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw/sub-0086', '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw/sub-0087']
for subject in sub_list: 
    phase_path = subject
    phase_search = phase_path + '/ses*/*phase*'
    phase_list = glob.glob(phase_search)

    for dir in phase_list:
        subfiles = glob.glob(dir + '/*')

        for subfile in subfiles:
            file = pydicom.filereader.dcmread(subfile)
            desc = file.SeriesDescription
            substr = "_phase"
            if not substr in desc: #only do it on new files that have not already been edited 
                newDesc = desc + '_phase'
                file.SeriesDescription = newDesc
                file.save_as(subfile)

