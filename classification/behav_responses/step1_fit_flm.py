import os 
import nilearn 
from glob import glob 
from sys import argv
import pandas as pd
from nilearn import masking
from nilearn.image import load_img
import matplotlib.pyplot as plt 
from datetime import datetime

#%%

###########################################################################################################
#                                    Define sublist and data paths                                        #
###########################################################################################################

#Define subject list 
dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/'
sublist = os.listdir(dir)
sublist = [sub for sub in sublist if sub.startswith('s')]

#Subjects to exclude 
exclude = ['sub-0020','sub-0038','sub-0059', 'sub-0053', 'sub-0050', 'sub-0057', 'sub-0058', 'sub-0018', 'sub-0021', 'sub-0027', 'sub-0067']
sublist = [sub for sub in sublist if not sub in exclude]

#Crashed at some point, running the last 3 subjects 
sublist = ['sub-0085','sub-0086','sub-0087']

print('Fitting FL models to subjects: '+str(sublist))

#Sub-0020: had too many missing repsonses in several runs - probably fell asleep
#Sub-0038: had too many missing repsonses in several runs - probably fell asleep
#Sub-0059: had too many missing repsonses in several runs - probably fell asleep 
#Sub-0053: (py-wrapper 6456705): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0050: (py-wrapper 6456706): ValueError: array must not contain infs or NaNs
#Sub-0057: (py-wrapper 6456707): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0058: (py-wrapper 6456708): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0018: (py-wrapper 6456709): missing event file for block 1, missing all physiological files, incomplete BOLD file for run 1
#Sub-0021: (py-wrapper 6456710): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0027: (py-wrapper 6456711): no mask provided for intersection - fMRIprep failed - no files 
#Sub-0067: missing eventfile for session 3 

#Data specific params 
space = 'MNI152NLin2009cAsym'

###########################################################################################################
#                          Loop over subs and compute FL models and contrasts                             #
###########################################################################################################

#Exclude subs already run
completed = ['sub-0025']
sublist = [sub for sub in sublist if sub not in completed]

#Functional files 
for sub in sublist: 
    start = datetime.now()

    print('--------------------------------- Running subject '+sub+'----------------------------------')

    #------------------------------------- Load subject files --------------------------------------#

    func_dir = os.path.join(dir, 'derivatives',sub,'func')
    anat_dir = os.path.join(dir, 'derivatives', sub, 'anat')

    funcs = sorted(glob(os.path.join(func_dir, f'*space-{space}*-preproc*.nii.gz')))
    anat = sorted(glob(os.path.join(anat_dir, f'*space-{space}*-preproc*.nii.gz')))

    confs, events, masks = [], [], []
    conf_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
    for func in funcs:

        mask_path = func.replace(f'_echo-1_space-{space}_desc-preproc_bold.nii.gz', f'_space-{space}_desc-brain_mask.nii.gz')
        masks.append(mask_path)

        # Find the associated confounds file
        conf_path = func.replace(f'_echo-1_space-{space}_desc-preproc_bold.nii.gz', '_desc-confounds_timeseries.tsv')
        conf_df = pd.read_csv(conf_path, sep='\t').loc[:, conf_cols]
        confs.append(conf_df)
            
        # Find the associated events file
        event_path = func.replace('BIDS/derivatives/','BIDS/')
        event_path = event_path.replace(f'_echo-1_space-{space}_desc-preproc_bold.nii.gz', '_events.tsv')
        event_df = pd.read_csv(event_path, sep=',', index_col=0)
        events.append(event_df)
    
    print('Found '+ str(len(funcs)) + ' functional files')
    print('Found '+ str(len(confs)) + ' confound files')
    print('Found '+ str(len(events)) + ' event files')

    #If multiple masks, create intersection 
    mask_img = masking.intersect_masks(masks, threshold=0.8)

    #------------------------------------- Define and plot design matrix --------------------------------------#
    #Will get one column per trial 
    # --> one beta-coef per voxel per trial 

    from nilearn.glm.first_level import make_first_level_design_matrix
    import numpy as np
    tbt_dm=[]

    for i in range(len(events)):
        #n=number of events
        n=events[i].shape[0]
        #582 timeopints 
        t_fmri = np.linspace(0, 582,582,endpoint=False)

        # We have to create a dataframe with onsets/durations/trial_types
        trials = pd.DataFrame(events[i], columns=['onset', 'duration'])
        #To string on resp, so nan responses are also kept as string 
        trials.loc[:, 'trial_type'] = ['t_'+str(ii).zfill(3) + '_' + str(events[i]['choice'][ii-1]) for ii in range(1, n+1)]

        # lsa_dm = least squares all design matrix
        tbt_dm.append(make_first_level_design_matrix(
            frame_times=t_fmri,  
            events=trials,
            add_regs=confs[i], #Add the confounds from fmriprep
            hrf_model='glover',
            drift_model='cosine'
        )) 
    
    print('Finished defining ' + str(len(tbt_dm)) + ' trial-by-trial design matrices')

    #Plot design matrix 
    from nilearn.plotting import plot_design_matrix
    for i in range(len(events)):
        plot_design_matrix(tbt_dm[i]);

    #---------------- Get correlation structure of design matrix and save plots  --------------------------#
    import seaborn as sns
    for i in range(0, len(tbt_dm)):
        dm_corr = tbt_dm[i].corr()
        f, ax=plt.subplots(figsize=(10,10))
        p1 = sns.heatmap(dm_corr, ax=ax)
        fig = p1.get_figure()
        fig.savefig(os.path.join("/users/line/dct_fMRI_analysis/classification/behav_responses/output/dm_correlation_matrices/"+sub+f'_dm_corr_run-{i+1}.png'))

    print('Saved design matrix correlation structure images')

    #---------------- Define and fit first-level models --------------------------#
    import nibabel as nib
    from nilearn.glm.first_level import FirstLevelModel
    flm = FirstLevelModel(
        t_r=nib.load(funcs[0]).header['pixdim'][4],
        slice_time_ref=0.5,
        mask_img=mask_img,
        #noise_model, drift_model, and hrf_model defaults fit our data, so not specified
    )

    print('Fitting first-level models')

    #Fit model. When design matrices are passed, they take precedence over events and confounds
    flm.fit(run_imgs=funcs, events=events, confounds=confs, design_matrices=tbt_dm)

    #Saving the model object and design matrix 
    import pickle 
    f = open(os.path.join('/users/line/dct_fMRI_analysis/classification/behav_responses/output/flm_objects/'+f'{sub}_flm.pkl'), 'wb')
    pickle.dump([flm, tbt_dm], f)
    f.close()
    
    print('Saved flm objects and design matrix')

    #---------------- Compute contrasts and store zmaps --------------------------#

    # Load model objects and design matrix :
    f = open(os.path.join('/users/line/dct_fMRI_analysis/classification/behav_responses/output/flm_objects/'+f'{sub}_flm.pkl'), 'rb')
    model, tbt_dm = pickle.load(f)
    f.close()

    #Define and compute contrast 
    z_maps = []
    conditions_label = []

    for i in range(len(events)):
        print('Computing contrast zmaps for session ' + str(i+1))

        N=events[i].shape[0]
        #Make an identity matrix with N= number of trials
        contrasts=np.eye(N)
        print(str(contrasts.shape))
        #Find difference between columns in design matrix and number of trials
        dif=tbt_dm[i].shape[1]-contrasts.shape[1]
        #print(dif)
        #Pad with zeros
        contrasts=np.pad(contrasts, ((0,0),(0,dif)),'constant')
        #print(contrasts.shape)
        for ii in range(N):
            print('Computing contrast for trial ' + str(ii+1))
            #Add a z-contrast image from each trial
            z_maps.append(model.compute_contrast(contrasts[ii,], output_type='z_score'))
            # Make a variable with condition labels for use in later classification
            conditions_label.append(str(events[i]['choice'][ii]))
    #       session_label.append(session)

    #Save models, design matrix, and zmaps 
    f = open(os.path.join('/users/line/dct_fMRI_analysis/classification/behav_responses/output/flm_objects/'+f'{sub}_flm_zmaps.pkl'), 'wb')
    pickle.dump([model, tbt_dm, conditions_label, z_maps], f)
    f.close() 
    end = datetime.now()

    print('Saved flm objects, design matrix, conditions labels, and zmaps')
    print('Finsihed subject: started at', start.strftime("%H:%M:%S"),'- ended at', end.strftime("%H:%M:%S"))
    