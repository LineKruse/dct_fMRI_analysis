#------------------ Import libraries -------------------- #

import pickle
import glob
import nilearn
from nilearn import plotting
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt 
import os 
import nibabel as nib
from scipy.stats import norm
from nilearn.plotting import plot_stat_map
from matplotlib.cm import get_cmap

#-------------------- Load objects ----------------------- # 

#Load first-level model objects 
# flms_folder = 'fitted_flms'
# flms_dir = glob.glob(r'/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/analysis/{}/*.plk'.format(flms_folder))
# flms = [] 
# for model in flms_dir:
#     pickled_model = pickle.load(open(model,'rb')) # load the model 
#     flms.append(pickled_model)
# print(len(flms))

input_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/line_dct_fMRI/output'
output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/line_dct_fMRI/output_new'
alpha = 0.001

#Load this vs. baseline first level contrast images 
contrast_dir = 'flcon_zmap_this_vs_baseline'
files = os.listdir(os.path.join(input_dir, contrast_dir))
this_baseline_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    this_baseline_con_imgs.append(img)

#Load that vs. baseline first level contrast images 
contrast_dir = 'flcon_zmap_that_vs_baseline'
files = os.listdir(os.path.join(input_dir, contrast_dir))
that_baseline_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    that_baseline_con_imgs.append(img)

#Load this + that first level contrast images 
contrast_dir = 'fl_con_zmap_this_plus_that'
files = os.listdir(os.path.join(input_dir, contrast_dir))
this_plus_that_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    this_plus_that_con_imgs.append(img)

#Load this - that first-level contrast images 
contrast_dir = 'fl_con_zmap_this-that'
files = os.listdir(os.path.join(input_dir, contrast_dir))
this_that_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    this_that_con_imgs.append(img)

#Load that - this first-level contrast images 
contrast_dir = 'fl_con_zmap_that-this'
files = os.listdir(os.path.join(input_dir, contrast_dir))
that_this_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    that_this_con_imgs.append(img)


#-------------------- Function - Create thresholded map and plot on glass brain ------------- # 
def threshold_map_and_plot_func(contrast_map, alpha, height_control, contrast, output_path):

    """
    Operations: 
    - Compute thresholded map given method of correction, and alpha 
    - Plot glass brain and stat map of the thresholded map and save 
    - Compute cluster information using atlasreader and save in cluster_dir 

    contrast_map: a second level contrast map
    alpha: threshold for "height-control"
    height_control: FWER method (bonferroni, fdr, fpr)
    contrast: contrast considered, only used to name files, str
    output_path: main directory of second level model results 
    """

    from nilearn.glm import threshold_stats_img
    from nilearn import plotting
    from matplotlib.cm import get_cmap

    # Step 1: Create thresholded map
    thresholded_map, threshold = threshold_stats_img(
        contrast_map, 
        alpha=alpha, 
        height_control=height_control) 


    print('The {}={} threshold is {}'.format(height_control, alpha, threshold))


    # Step 2: Plot on glass brain 
    plotting.plot_glass_brain(thresholded_map, colorbar=True,black_bg=True,display_mode='lyrz', plot_abs=False, cmap='jet', title='{}-controlled map (alpha={}) for contrast {}'.format(height_control, alpha, contrast))
    plt.savefig(output_path+'_glassZmap.png',facecolor='k', edgecolor='k')
    plt.show()

    # Step 3: Plot stats map 
    plot_stat_map(thresholded_map, cmap='jet', threshold=threshold,cut_coords=[-30,-20,-10,0,10,20,30],display_mode='z',  black_bg=True, title='{}-controlled map (alpha={}) for contrast {}'.format(height_control, alpha, contrast))
    plt.savefig(output_path+'_statMap.png',facecolor='k', edgecolor='k')
    plt.show()

    # Step 4: Compute cluster information 
    from atlasreader import create_output
    cluster_dir = os.path.join(output_dir, 'cluster_results',contrast,height_control)
    if not os.path.exists(cluster_dir):
        os.mkdir(cluster_dir)

    # Step 3: Create output 
    create_output(
        filename = thresholded_map,
        atlas='default',
        voxel_thresh=1.96,
        direction='both',
        cluster_extent=5,
        glass_plot_kws = {'black_bg':True,'vmax':20,'colorbar':True, 'cmap': 'jet'},
        stat_plot_kws = {'black_bg':True,'cmap': 'jet','title':False},
        outdir=cluster_dir
        )
    
# ------------- Function: compute parametric and non-parametric tests and plot ------------- # 
#This function computes parametric correction and permutation test (non-parametric) and create and saves plot with each
def permutation_test_and_figure(slm, second_level_input, design_matrix, contrast, file_name):
    import numpy as np 
    from nilearn.image import get_data, math_img 
    #------ Obtain null distribution by estimating the effect N times and randomly flipping the sign of each datapoint -----#
    #Get an image of p-values 
    p_val = slm.compute_contrast(output_type='p_value', second_level_contrast='intercept')
    #Get the number of voxels in the image 
    n_voxels = np.sum(get_data(slm.masker_.mask_img_))
    #Making an image with significant values, correcting the p-values for multiple testing and taking negative log 
    neg_log_pval = math_img(
        '-np.log10(np.minimum(1, img * {}))'.format(str(n_voxels)),
        img=p_val)

    #--------- Perform permutations ---------- # 
    from nilearn.glm.second_level import non_parametric_inference 
    n_perm = 1000 
    #Calculate p-values using permutations 
    out_dict = non_parametric_inference(
        second_level_input,
        design_matrix=design_matrix,
        second_level_contrast='intercept',
        model_intercept=True,
        n_perm=n_perm,  
        two_sided_test=False,
        smoothing_fwhm=8.0,
        n_jobs=1  
        )
    
    #------- Plot maps for parametric and non-parametric corrections ------- # 
    threshold_log = round(-np.log10(1 / 20))  # p < 0.05 - take into account that effects are now negative log of the p-value 
    vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
    cut_coords = [0]

    #Prepare images to plot 
    IMAGES = [
        neg_log_pval,
        out_dict,
    ]
    TITLES = [
        'Parametric Test',
        'Permutation Test\n(Voxel-Level Error Control)',
    ]

    #Make loop with plots
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    img_counter = 0
    i_row=0
    for j_row in range(2):
            ax = axes[j_row]     
            plotting.plot_glass_brain(
                IMAGES[img_counter],
                colorbar=True,
                vmax=vmax,
                plot_abs=False,
                black_bg=True, 
                display_mode='lyrz',
                cmap='jet',
                cut_coords=cut_coords,
                threshold=threshold_log,
                figure=fig,
                axes=ax,
            )

            font_dict = {'color': 'white'}
            ax.set_title(TITLES[img_counter], fontdict=font_dict)
            img_counter += 1
    fig.suptitle('Group effect {} contrast\n(negative log10 p-values)'.format(contrast), color='white')
    plt.savefig(file_name,facecolor='k', edgecolor='k')
    plt.show()


#---------------------------------Intercept models ------------------------------------ # 
from nilearn.glm.second_level import SecondLevelModel

#Define design matrix (intercept model only)
design_matrix = pd.DataFrame(
    [1] * len(this_plus_that_con_imgs),
    columns=['intercept'],
)

#Set alpha for all models 
#alpha = 0.001 #run - results in "output/sl_intercept_models_alpha-001"
alpha=0.05 #run - results in "output/sl_intercept_models_alpha-05"


#-------------------- Intercept models with Gender + Age as confounds ------------- # 


#Create list of subjects (in the order they are presented in the "files" object)
bids_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS'
sublist = []
gender = []
age = []
for file in files: 
    #Extract subject name 
    sub = file.split('_')[0]
    sublist.append(sub)

    #Load event file  
    path = os.path.join(bids_path, sub, 'func')
    search = path + '/*run-1_events.tsv'
    eventFile = glob.glob(search)
    df = pd.read_csv(eventFile[0], sep=",")

    #Extract Gender 
    gen = df['Gender'][0]
    if gen=='Female':
        gender.append(0)
    if gen=='Male':
        gender.append(1)

    #Extract Age
    a = df['Age'][0]
    age.append(a) 

#Define design matrix (intercept model only)
intercept = np.ones_like(age)
gender = np.array(gender)
age = np.array(age)

design_matrix = pd.DataFrame({
    'intercept': intercept,
    'gender': gender,
    'age': age}
)

#=========================================================================================#
#--------------------------This vs. baseline contrast ----------------# 
#=========================================================================================#
from nilearn.glm.second_level import SecondLevelModel

print('-------- Running second level models for contrast: "this vs basline"----------')

print('- Fitting second level model')

slm_int_this = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_this = slm_int_this.fit(
    this_baseline_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: intercept')

zmap_int_this = slm_int_this.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score',
)
data = zmap_int_this.get_fdata()
print(data.max()) # 9.22
print(data.min()) #-8.72
# fdr threshold (0.001) = 3.65 
# fdr threshold (0.05) = 2.29
# bonferroni (0.001) = 5.96
# bonferroni (0.05) = 5.29 

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_this,black_bg=True, cmap='jet',display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group effect - "this" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_int_this_uncor_glassZmap.png',facecolor='k', edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_this, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',black_bg=True,
              title='Group effect - "this" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_int_this_uncor_statMap.png',facecolor='k', edgecolor='k')

#Correcting for multiple comparisons - FDR 
#- this function both computes thresholded maps, figures, and cluster information 

print('- Running FDR correction')

out_path = output_dir+'/zmap_int_this_fdr'
threshold_map_and_plot_func(zmap_int_this, alpha=alpha, height_control='fdr',contrast='this',output_path=out_path)

#Correcting for multiple comparisons - FWER 
#- this function both computes thresholded maps, figures, and cluster information 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap1_int_this_fwer')
threshold_map_and_plot_func(zmap_int_this, alpha=alpha, height_control='bonferroni',contrast='this',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_int_this_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_this, this_baseline_con_imgs, design_matrix, contrast='this',file_name=file_name)



#=========================================================================================#
#--------------------------That vs. baseline contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models for contrast: "that vs basline"----------')

print('- Fitting second level model')

slm_int_that = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_that = slm_int_that.fit(
    that_baseline_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: intercept')

zmap_int_that = slm_int_that.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score',
)
data = zmap_int_that.get_fdata()
print(data.max()) # 8.99
print(data.min()) # -8.68
# fdr threshold = 3.65 
# fdr threshold = 2.29 
# bonferroni threshold = 5.96
# bonferroni threshold = 5.28 

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_that, cmap='jet', black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group effect - "that" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_int_that_uncor_glassZmap.png', facecolor='k', edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_that, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group effect - "that" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_int_that_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_int_that_fdr'
threshold_map_and_plot_func(zmap_int_that, alpha=alpha, height_control='fdr',contrast='that',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap1_int_that_fwer')
threshold_map_and_plot_func(zmap_int_that, alpha=alpha, height_control='bonferroni',contrast='that',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_int_that_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_that, that_baseline_con_imgs, design_matrix, contrast='that',file_name=file_name)


#=========================================================================================#
#-------------------------- This + that contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models for contrast: "this + that"----------')

print('- Fitting second level model')

slm_int_this_plus_that = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_this_plus_that= slm_int_this_plus_that.fit(
    this_plus_that_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: intercept')

zmap_int_this_plus_that = slm_int_this_plus_that.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score',
)
data = zmap_int_this_plus_that.get_fdata()
print(data.max()) # 9.16 
print(data.min()) # -8.75 
# fdr threshold (0.001) = 3.64
# fdr threshold (0.05) = 2.28
# bonferroni threshold (0.001) = 5.96 
# bonferroni threshold (0.05) = 5.28

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_this_plus_that, cmap='jet',black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group effect - "this + that" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_int_this_plus_that_uncor_glassZmap.png', facecolor='k',edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_this_plus_that, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group effect - "this + that" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_int_this_plus_that_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_int_this_plus_that_fdr'
threshold_map_and_plot_func(zmap_int_this_plus_that, alpha=alpha, height_control='fdr',contrast='this+that',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap1_int_this_plus_that_fwer')
threshold_map_and_plot_func(zmap_int_this_plus_that, alpha=alpha, height_control='bonferroni',contrast='this+that',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_int_this_plus_that_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_this_plus_that, this_plus_that_con_imgs, design_matrix, contrast='this+that',file_name=file_name)

#------ Manual non-parametric test and plot ----------------- # 
import numpy as np 
from nilearn.image import get_data, math_img 
#------ Obtain null distribution by estimating the effect N times and randomly flipping the sign of each datapoint -----#
#Get an image of p-values 
p_val = slm_int_this_plus_that.compute_contrast(output_type='p_value', second_level_contrast='intercept')
#Get the number of voxels in the image 
n_voxels = np.sum(get_data(slm_int_this_plus_that.masker_.mask_img_))
#Making an image with significant values, correcting the p-values for multiple testing and taking negative log 
neg_log_pval = math_img(
    '-np.log10(np.minimum(1, img * {}))'.format(str(n_voxels)),
    img=p_val)

#Perform permutations# 
from nilearn.glm.second_level import non_parametric_inference 
n_perm = 1000 
#Calculate p-values using permutations 
out_dict = non_parametric_inference(
    this_plus_that_con_imgs,
    design_matrix=design_matrix,
    second_level_contrast='intercept',
    model_intercept=True,
    n_perm=n_perm,  
    two_sided_test=False,
    smoothing_fwhm=8.0,
    n_jobs=1  
    )
nib.save(out_dict, output_dir+'/out_dict_this_plus_that.nii')

#------- Plot maps for parametric and non-parametric corrections ------- # 
import nibabel as nib
out_dict = nib.load(output_dir+'/out_dict_this_plus_that.nii')

threshold_log = round(-np.log10(1 / 20))  # p < 0.05 - take into account that effects are now negative log of the p-value 
vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
cut_coords = [0]

contrast='this plus that'
#Make loop with plots
plotting.plot_glass_brain(
            out_dict,
            colorbar=True,
            vmax=vmax,
            plot_abs=False,
            black_bg=True, 
            display_mode='lyrz',
            cmap="jet",
            cut_coords=cut_coords,
            threshold=threshold_log,
            title='A) "This"+"That" > Baseline'
        )
plt.savefig(output_dir+'/thisplusthat_perm05_glassZmap.png')
plt.show()



#=========================================================================================#
#-------------------------- This-that contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models for contrast: "this - that"----------')

print('- Fitting second level model')

slm_int_this_that = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_this_that = slm_int_this_that.fit(
    this_that_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: intercept')

zmap_int_this_that = slm_int_this_that.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score',
)
data = zmap_int_this_that.get_fdata()
data.shape
print(data.max()) # 4.90
print(data.min()) # -3.40
# fdr threshold (0.001) = inf 
# fdr threshold (0.05) = inf 
# bonferroni threshold (0.001) = 5.96 
# bonferroni threshold (0.05) = 5.28 


#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_this_that, cmap='jet',black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group effect - "this-that" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_int_this-that_uncor_glassZmap.png', facecolor='k',edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_this_that, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group effect - "this-that" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_int_this-that_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_int_this-that_fdr'
threshold_map_and_plot_func(zmap_int_this_that, alpha=alpha, height_control='fdr',contrast='this-that',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap1_int_this-that_fwer')
threshold_map_and_plot_func(zmap_int_this_that, alpha=alpha, height_control='bonferroni',contrast='this-that',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_int_this-that_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_this_that, this_that_con_imgs, design_matrix, contrast='this-that',file_name=file_name)


#------ Manual non-parametric test and plot ----------------- # 
import numpy as np 
from nilearn.image import get_data, math_img 
#------ Obtain null distribution by estimating the effect N times and randomly flipping the sign of each datapoint -----#
#Get an image of p-values 
p_val = slm_int_this_that.compute_contrast(output_type='p_value', second_level_contrast='intercept')
#Get the number of voxels in the image 
n_voxels = np.sum(get_data(slm_int_this_that.masker_.mask_img_))
#Making an image with significant values, correcting the p-values for multiple testing and taking negative log 
neg_log_pval = math_img(
    '-np.log10(np.minimum(1, img * {}))'.format(str(n_voxels)),
    img=p_val)

#Perform permutations# 
from nilearn.glm.second_level import non_parametric_inference 
n_perm = 1000 
#Calculate p-values using permutations 
out_dict = non_parametric_inference(
    this_that_con_imgs,
    design_matrix=design_matrix,
    second_level_contrast='intercept',
    model_intercept=True,
    n_perm=n_perm,  
    two_sided_test=False,
    smoothing_fwhm=8.0,
    n_jobs=1  
    )
nib.save(out_dict, output_dir+'/out_dict_this-that.nii')

#------- Plot maps for parametric and non-parametric corrections ------- # 
import nibabel as nib
out_dict = nib.load(output_dir+'/out_dict_this-that.nii')

threshold_log = round(-np.log10(1 / 20))  # p < 0.05 - take into account that effects are now negative log of the p-value 
vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
cut_coords = [0]

contrast='this-that'
#Make loop with plots
plotting.plot_glass_brain(
            out_dict,
            colorbar=True,
            vmax=vmax,
            plot_abs=False,
            black_bg=True, 
            display_mode='lyrz',
            cmap='jet',
            cut_coords=cut_coords,
            threshold=threshold_log,
            title='B) "This" > "That"'
        )
plt.savefig(output_dir+'/this-that_perm05_glassZmap.png')
plt.show()

#Test plot filled contours 
levels=[threshold_log] #threshold argument for add_contours()
display = plotting.plot_glass_brain(None, plot_abs=False, display_mode='lyr', black_bg=False, threshold=threshold_log, vmax=vmax, colorbar=True)
display.add_contours(out_dict, filled=True, levels=levels)
plt.savefig(output_dir+'/test5.png')
plt.show()


#Plot surface map (nothing to see... )
from nilearn import surface, datasets
fsaverage = datasets.fetch_surf_fsaverage()

curv_right = surface.load_surf_data(fsaverage.curv_right)
curv_right_sign = np.sign(curv_right)

texture = surface.vol_to_surf(out_dict, fsaverage.pial_right)

fig = plotting.plot_surf_stat_map(
    fsaverage.infl_right, texture, hemi='right',
    title='Surface right hemisphere', colorbar=True,
    threshold=1., bg_map=curv_right_sign,
)
plt.savefig(output_dir+'/test2.png',facecolor='k', edgecolor='k')
fig.show()



#=========================================================================================#
#-------------------------- That - this contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models for contrast: "that-this"----------')

print('- Fitting second level model')

slm_int_that_this = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_that_this = slm_int_that_this.fit(
    that_this_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: intercept')

zmap_int_that_this = slm_int_that_this.compute_contrast(
    second_level_contrast='intercept',
    output_type='z_score',
)
data = zmap_int_that_this.get_fdata()
print(data.max()) # 3.40
print(data.min()) # -4.90
# fdr threshold (0.001) = inf 
# fdr threshold (0.05) = inf 
# bonferroni threshold (0.001) = 5.96 
# bonferroni threshold (0.05) = 5.28

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_that_this, cmap='jet',black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group effect - "that-this" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_int_that-this_uncor_glassZmap.png', facecolor='k',edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_that_this, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group effect - "that-this" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_int_that-this_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_int_that-this_fdr'
threshold_map_and_plot_func(zmap_int_that_this, alpha=alpha, height_control='fdr',contrast='that-this',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap1_int_that-this_fwer')
threshold_map_and_plot_func(zmap_int_that_this, alpha=alpha, height_control='bonferroni',contrast='that-this',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_int_that-this_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_that_this, that_this_con_imgs, design_matrix, contrast='that-this',file_name=file_name)


#------ Manual non-parametric test and plot ----------------- # 
import numpy as np 
from nilearn.image import get_data, math_img 
#------ Obtain null distribution by estimating the effect N times and randomly flipping the sign of each datapoint -----#
#Get an image of p-values 
p_val = slm_int_that_this.compute_contrast(output_type='p_value', second_level_contrast='intercept')
#Get the number of voxels in the image 
n_voxels = np.sum(get_data(slm_int_that_this.masker_.mask_img_))
#Making an image with significant values, correcting the p-values for multiple testing and taking negative log 
neg_log_pval = math_img(
    '-np.log10(np.minimum(1, img * {}))'.format(str(n_voxels)),
    img=p_val)

#Perform permutations# 
from nilearn.glm.second_level import non_parametric_inference 
n_perm = 1000 
#Calculate p-values using permutations 
out_dict = non_parametric_inference(
    that_this_con_imgs,
    design_matrix=design_matrix,
    second_level_contrast='intercept',
    model_intercept=True,
    n_perm=n_perm,  
    two_sided_test=False,
    smoothing_fwhm=8.0,
    n_jobs=1  
    )
nib.save(out_dict, output_dir+'/out_dict_that-this.nii')

#------- Plot maps for parametric and non-parametric corrections ------- # 
import nibabel as nib
out_dict = nib.load(output_dir+'/out_dict_that-this.nii')

threshold_log = round(-np.log10(1 / 20))  # p < 0.05 - take into account that effects are now negative log of the p-value 
vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
cut_coords = [0]

contrast='that-this'
#Make loop with plots
plotting.plot_glass_brain(
            out_dict,
            colorbar=True,
            vmax=vmax,
            plot_abs=False,
            black_bg=True, 
            display_mode='lyrz',
            cmap='jet',
            cut_coords=cut_coords,
            threshold=threshold_log,
            title='C) "That" > "This"'
        )
plt.savefig(output_dir+'/that-this_perm05_glassZmap.png')
plt.show()

# -------------- Run cluster detection on permutation map (this-that contrast) -------------------#

""" #First - Run permutation test to get thresholded map 
from nilearn.glm.second_level import non_parametric_inference 
n_perm = 1000 
#Calculate p-values using permutations 
second_level_input = this_plus_that_con_imgs
out_dict = non_parametric_inference(
    second_level_input,
    design_matrix=design_matrix,
    model_intercept=True,
    n_perm=n_perm,  
    two_sided_test=False,
    smoothing_fwhm=8.0,
    n_jobs=1, 
    )

second_level_input = this_that_con_imgs
out_dict1 = non_parametric_inference(
    second_level_input,
    design_matrix=design_matrix,
    model_intercept=True,
    n_perm=n_perm,  
    two_sided_test=False,
    smoothing_fwhm=8.0,
    n_jobs=1, 
    )

threshold_log = round(-np.log10(1 / 20))  # p < 0.05 - take into account that effects are now negative log of the p-value 
vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
cut_coords = [0]

#Check how many voxels survive permutation threshold (p<0.05)
out_dict.shape
vox = out_dict.get_fdata().flatten()
nVoxCluster = sum(i > threshold_log for i in vox)
# 43 voxels in significant cluster 


output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/'

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.figure(figsize=(1000*px, 600*px)) #save with width=1000 pixels (required by OHBM)
plotting.plot_glass_brain(out_dict, colorbar=True, vmax=vmax, plot_abs=False, black_bg=True, display_mode='lyrz',cmap='jet', cut_coords=cut_coords,threshold=threshold_log)
plt.savefig(output_dir+'this_plus_that_perm_glassZmap.png',facecolor='k', edgecolor='k')
plt.show()

#Second - use create_output funtionc of atlasreader to get cluster results 
os.system('pip install atlasreader')
from atlasreader import create_output

# Step 3: Create output 
create_output(
    filename = out_dict,
    atlas='default',
    voxel_thresh=1.96,
    direction='both',
    cluster_extent=5,
    glass_plot_kws = {'black_bg':True,'vmax':20,'colorbar':True, 'cmap': 'jet'},
    stat_plot_kws = {'black_bg':True,'cmap': 'jet','title':False},
    outdir=output_dir
    )


#Plot two glassbrains together (used for submission to OHBM only )
IMAGES = [
    out_dict1,
    out_dict,
]
TITLES = [
    'A',
    'B',
]
file_name = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/permutation_two_images.png'

#Make loop with plots
fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
img_counter = 0
i_row=0
for j_row in range(2):
        ax = axes[j_row]     
        plotting.plot_glass_brain(
            IMAGES[img_counter],
            colorbar=True,
            vmax=vmax,
            plot_abs=False,
            black_bg=True, 
            display_mode='lyrz',
            cmap='jet',
            cut_coords=cut_coords,
            threshold=threshold_log,
            figure=fig,
            axes=ax,
        )

        font_dict = {'color': 'white', 'horizontalalignment': 'left'}
        ax.set_title(TITLES[img_counter], fontdict=font_dict, loc='left')
        img_counter += 1
#fig.suptitle('Group effect {} contrast\n(negative log10 p-values)'.format(contrast), color='white')
plt.savefig(file_name,facecolor='k', edgecolor='k')
plt.show() """