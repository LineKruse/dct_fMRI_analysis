# this script is an attempt to make fmriprep on each subejct using parallel processing 
# this script should be run my the 'master_sirid.py' script where a function should be defined that loops over this script taking all participants

from sys import argv
import os

def fmriprep_func(subject):
    print('Beginning running fmriprep.')
    os.system('udocker run \
                -v /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/:/in \
                -v /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/derivatives:/out \
                -v /projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/FreeSurfer_license:/fs \
                -v /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line:/work \
                nipreps/fmriprep:22.0.2 /in /out participant \
                --participant-label sub-{} \
                --fs-no-reconall --fs-license-file /fs/license.txt -w /work'.format(subject))

def smoothing_func(subject):
    os.system('cd /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/derivatives_smooth/sub-{subject}/func \
        for run in 1 2 3; do \
            3dmerge -1blur_fwhm 4.0 -doall -prefix r${run}_blur.nii \
            sub-{subject}_task-DCT_run-${run}_echo-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz \
        done'.format(subject))

#Removed (was after --participant argument):                 --ignore sbref \ 

subject = argv[1]
    
fmriprep_func(subject)
#smoothing_func(subject)