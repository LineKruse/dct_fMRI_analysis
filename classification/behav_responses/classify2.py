import os
import numpy as np 
import pandas as pd 
from nilearn.plotting import plot_glass_brain, plot_stat_map, plot_img
from nilearn.image import new_img_like, load_img, index_img, clean_img, concat_imgs
import matplotlib.pyplot as plt 
import pickle
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score, permutation_test_score
from nilearn.maskers import NiftiMasker
from sklearn.svm import LinearSVC

# ------------------- Define subject list --------------------# 
dir = os.getcwd()
dir2 = os.path.join(dir,'classification/behav_responses/output/flm_objects/')
dirlist = os.listdir(dir2)
sublist = [sub[0:8] for sub in dirlist if 'zmaps' in sub]

########For testing#########
sublist = ['sub-0072', 'sub-0061', 'sub-0031']
###########################

#------------------ Define outputs -----------------------#
mean_cv_score = []
accuracy_score = []
pval_score = []
perm_score = [] 

for sub in sublist: 

# ----------------- Load searchlight objects -----------------# 
    path = os.path.join(dir, f'classification/behav_responses/output/searhlight_objects/{sub}_searchlight.pkl')
    searchlight = pd.read_pickle(path)[0]

# ------------- Make plots of searchlight results -------------# 
    mask_wb_filename=f'/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/derivatives/{sub}/anat/{sub}_acq-t1mpragetrap2iso_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    anat_filename=f'/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/derivatives/{sub}/anat/{sub}_acq-t1mpragetrap2iso_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

    fig_dir = os.path.join(dir, f'classification/behav_responses/output/searchlight_plots/{sub}')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    #Create an image of the searchlight scores
    searchlight_img = new_img_like(anat_filename, searchlight.scores_)

    plot_glass_brain(searchlight_img, cmap='jet',colorbar=True, threshold=0.01,
                            title='Proximal vs Distal Demonstrative Choice (unthresholded)',
                            plot_abs=False)
    plt.savefig(fig_dir+'/prox_vs_dist_searchlight_uncor.png')

    plot_glass_brain(searchlight_img,threshold=0.6,title='Proximal vs Distal Demonstrative Choice (Acc>0.6')
    plt.savefig(fig_dir+'/prox_vs_dist_searchlight_acc>06.png')

    plot_stat_map(searchlight_img, cmap='jet',threshold=0.6, cut_coords=[-30,-20,-10,0,10,20,30],
                display_mode='z',  black_bg=False,
                title='Proximal vs Distal Demonstrative Choic (Acc>0.6)')
    plt.savefig(fig_dir+'/prox_vs_dist_searchlight_acc>06_statmap.png')

# ------------- Create mask with 500 most predictive voxels -------------# 
    #print(searchlight.scores_.size)
    #Find the percentile that makes the cutoff for the 500 best voxels
    perc=100*(1-500.0/searchlight.scores_.size)
    #Find the cutoff
    cut=np.percentile(searchlight.scores_,perc)

    #Make a mask using cutoff
    mask_img2 = load_img(mask_wb_filename) #Load the whole brain mask

    process_mask2 = mask_img2.get_fdata().astype(int) #.astype() makes a copy.
    process_mask2[searchlight.scores_<=cut] = 0
    process_mask2_img = new_img_like(mask_img2, process_mask2)

    #Plot mask 
    searchlight_img = new_img_like(anat_filename, searchlight.scores_)
    #Plot the searchlight scores on an anatomical background
    plot_img(searchlight_img, bg_img=anat_filename,#bg_img=mean_fmri,
            title="Searchlight", display_mode="z",cut_coords=[-25,-20,-15,-10,-5,0,5],
            vmin=.40, cmap='jet', threshold=cut, black_bg=True)
    plt.savefig(fig_dir+'/prox_vs_dist_searchlight_on_anat.png')
    
    #plotting.plot_glass_brain effects
    plot_glass_brain(searchlight_img,threshold=cut)
    plt.savefig(fig_dir+'/prox_vs_dist_searchlight_best500.png')

#----------------------- Load models, design matrix, conditions_labels and zmaps---------------------------

    f = open(os.path.join('/users/line/dct_fMRI_analysis/classification/behav_responses/output/flm_objects/'+f'{sub}_flm_zmaps.pkl'), 'rb')
    model, tbt_dm, conditions_label, z_maps = pickle.load(f)
    f.close() 

#---------------------------------- Reshape data for classification ---------------------------------------
    #-----Data is reshaped and split exactly identical to before training searchlight(was just in another script)

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

#---------------------------------- Create train-test splits --------------------------------------#
    idx2=np.arange(conditions.shape[0])
    
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(idx2, conditions,
                                                              stratify = conditions,
                                                              test_size=0.3,
                                                              random_state=42)
    
    #Assign zmaps to X_train and X_test by split indices 
    X_train = index_img(z_maps_img, X_train_idx)
    X_test = index_img(z_maps_img, X_test_idx)

#---------------------------------- Classification on test set --------------------------------------#
    
    masker = NiftiMasker(mask_img=process_mask2_img, standardize=False)

    # We use masker to retrieve a 2D array ready for machine learning with scikit-learn
    fmri_masked = masker.fit_transform(X_test)
    #Print size of matrix (images x voxels)
    #print(fmri_masked.shape)

    #Cross validation 
    cv_score = cross_val_score(LinearSVC(penalty='l2'), fmri_masked, y_test, cv=10)
    mean_cv_score.append(np.mean(cv_score))

    #Permutation test 
    score, permutation_scores, pvalue= permutation_test_score(
        LinearSVC(penalty='l2'), fmri_masked, y_test, cv=10, n_permutations=1000, 
        n_jobs=-1, random_state=0, verbose=0, scoring=None)
    accuracy_score.append(score)
    pval_score.append(pvalue)
    perm_score.append(permutation_scores)

perm_df = pd.DataFrame(perm_score).T
perm_df.columns = sublist
perm_df.to_csv(os.path.join(dir, f'classification/behav_responses/output/permutation_scores.csv'))

acc_df = pd.DataFrame({'sub':sublist,'cv_score':mean_cv_score,'perm_acc':accuracy_score, 'perm_pval':pval_score})
acc_df.to_csv(os.path.join(dir, f'classification/behav_responses/output/accuracy_scores.csv'))