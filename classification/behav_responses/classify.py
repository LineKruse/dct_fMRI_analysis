import pickle 
import pandas as pd 
import numpy as np 
import os 


###########################################################################################################
#                                          Define subject list                                            #
###########################################################################################################

#Define subject list 
dir = '/users/line/dct_fMRI_analysis/classification/behav_responses/output/flm_objects/'
dirlist = os.listdir(dir)
sublist = [sub[0:8] for sub in dirlist if 'zmaps' in sub]

#For testing 
#sublist = ['sub-0025']

###########################################################################################################
#                           Loop through subjects, load zmaps, run classification                         #
###########################################################################################################

for sub in sublist: 

    print('------------------------- Running searchlight on',sub,'------------------------------')

#----------------------- Load models, design matrix, conditions_labels and zmaps---------------------------

    f = open(os.path.join('/users/line/dct_fMRI_analysis/classification/behav_responses/output/flm_objects/'+f'{sub}_flm_zmaps.pkl'), 'rb')
    model, tbt_dm, conditions_label, z_maps = pickle.load(f)
    f.close() 

#---------------------------------- Reshape data for classification ---------------------------------------

    from nilearn.image import new_img_like, load_img, index_img, clean_img
    from sklearn.model_selection import train_test_split, GroupKFold
    from nilearn.image import index_img, concat_imgs

    #Reshaping data
    idx_this=[int(i) for i in range(len(conditions_label)) if conditions_label[i]=='this']
    idx_that=[int(i) for i in range(len(conditions_label)) if conditions_label[i]=='that']

    #Concatenate trial list 
    idx=np.concatenate((idx_this, idx_that))

    #Concatenate zmaps and order according to trial list 
    conditions=np.array(conditions_label)[idx]
    z_maps_conc=concat_imgs(z_maps)
    z_maps_img = index_img(z_maps_conc, idx)
    #shape: 109, 129, 109, 303 

#---------------------------------- Create train-test splits ---------------------------------------
    idx2=np.arange(conditions.shape[0])
    
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(idx2, conditions,
                                                              stratify = conditions,
                                                              test_size=0.3,
                                                              random_state=42)
    
    #Assign zmaps to X_train and X_test by split indices 
    X_train = index_img(z_maps_img, X_train_idx)
    X_test = index_img(z_maps_img, X_test_idx)

#--------------------------------------- Run searchlight  ------------------------------------------
    from glob import glob
    from nilearn import masking
    from nilearn.plotting import plot_img
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    #Load T1 image 
    space = 'MNI152NLin2009cAsym'
    anat_dir = os.path.join('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/derivatives', sub, 'anat')
    anat_filename = sorted(glob(os.path.join(anat_dir, f'*space-{space}*-preproc*.nii.gz')))
    
    #Load whole brain mask (intersecting the three masks)
    mask_dir = os.path.join('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/derivatives',sub,'anat')
    mask_filename = sorted(glob(os.path.join(mask_dir, f'*space-{space}_desc-brain_mask*.nii.gz')))
    mask_img = load_img(mask_filename)

    #Get copy of mask 
    process_mask = mask_img.get_fdata().astype(int)
    process_mask = process_mask[:,:,:,0]
    #Set slices below x in the z-dimension to zero (in voxel space)
    #process_mask[..., :40] = 0
    #Set slices above x in the z-dimension to zero (in voxel space)
    #process_mask[..., 150:] = 0
    process_mask_img = new_img_like(mask_img, process_mask)

    #Plot mask - not working (saying input is 4D, but it is 3D)
    #plot_img(process_mask_img, bg_img=anat_filename,#bg_img=mean_fmri,
    #        title="Mask", display_mode="z",cut_coords=[-30,-20,-10,0,10,20,30,40,50],
    #        vmin=.40, cmap='jet', threshold=0.9, black_bg=True)
    #fig.savefig('/users/line/dct_fMRI_analysis/classification/behav_responses/output/example_SL_mask')

    #Run searchlight with SVM estimator (sphere radius=5)
    from nilearn.decoding import SearchLight
    from sklearn.svm import LinearSVC

    searchlight = SearchLight(
        mask_img,
        estimator=LinearSVC(penalty='l2'),
        process_mask_img=process_mask_img,
        radius=5, n_jobs=-1, #Radius in mm 
        verbose=10, cv=3) #Setting K=3 (instead of 10), extremely computationally costly 
    
    start = datetime.now()
    searchlight.fit(X_train, y_train)
    end=datetime.now() 
    print('Searchlight started:',start.strftime("%Y-%m-%d %H:%M:%S"))
    print('Searchlight ended:', end.strftime("%Y-%m-%d %H:%M:%S"))

    # Saving the objects:
    import pickle
    f = open(os.path.join('/users/line/dct_fMRI_analysis/classification/behav_responses/output/searhlight_objects/'+f'{sub}_searchlight.pkl'),'wb')
    pickle.dump([searchlight], f)
    f.close()

