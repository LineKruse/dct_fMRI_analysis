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

input_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output'
output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_groupGender_models'
#Set alpha for all models 
alpha = 0.001 #run - results in "output/sl_group_models_alpha-001"
#alpha=0.05 # run - results  in "output/sl_group_models_alpha-05"

#Load first-level model objects 
# flms_folder = 'fitted_flms'
# flms_dir = glob.glob(r'/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/analysis/{}/*.plk'.format(flms_folder))
# flms = [] 
# for model in flms_dir:
#     pickled_model = pickle.load(open(model,'rb')) # load the model 
#     flms.append(pickled_model)
# print(len(flms))

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
        stat_img=contrast_map, 
        alpha=alpha, 
        height_control=height_control,
        two_sided=False) 

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
    p_val = slm.compute_contrast(second_level_contrast = 'group',output_type='p_value')
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
        second_level_contrast = 'group',
        design_matrix=design_matrix,
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


#--------------------------------- Group difference models  ------------------------------------ # 
from nilearn.glm.second_level import SecondLevelModel

#Create list of subjects (in the order they are presented in the "files" object)
bids_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS'
sublist = []
#group = []
gender = []
for file in files: 
    #Extract subject name 
    sub = file.split('_')[0]
    sublist.append(sub)

    #Extract group index from event file 
    path = os.path.join(bids_path, sub, 'func')
    search = path + '/*run-1_events.tsv'
    eventFile = glob.glob(search)
    df = pd.read_csv(eventFile[0], sep=",")
    #Control subjects = -1
    # if df['group'][0] == 0:
    #     group.append(-1)
    # #Depression subjects = 1
    # if df['group'][0] == 1: 
    #     group.append(1)
    if df['Gender'][0]=='Female':
        gender.append(0)
    if df['Gender'][0]=='Male':
        gender.append(1)
    

#Define design matrix (intercept model only)
intercept = np.ones_like(gender)
group = np.array(gender)

design_matrix = pd.DataFrame({
    'group': group, 
    'intercept': intercept}
)



#=========================================================================================#
#--------------------------This vs. baseline contrast ----------------# 
#=========================================================================================#
print('-------- Running second level models of group difference for contrast: "this vs basline"----------')

print('- Fitting second level model')

slm_int_this = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_this = slm_int_this.fit(
    this_baseline_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: group')

zmap_int_this = slm_int_this.compute_contrast(
    second_level_contrast='group',
    output_type='z_score',
)
data = zmap_int_this.get_fdata()
print(data.max) # 4.16 
print(data.min) # - 3.94 
# fdr threshold (0.05) = inf
# fdr threshold (0.001) = inf 
# bonferroni (0.05) threshold = 5.2
# bonferroni (0.001) treshold = 5.9 

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_this,black_bg=True, cmap='jet',display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group difference - "this" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_group_this_uncor_glassZmap.png',facecolor='k', edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_this, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',black_bg=True,
              title='Group difference - "this" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_group_this_uncor_statMap.png',facecolor='k', edgecolor='k')

#Correcting for multiple comparisons - FDR 
#- this function both computes thresholded maps, figures, and cluster information 

print('- Running FDR correction')

out_path = output_dir+'/zmap_group_this_fdr'
threshold_map_and_plot_func(zmap_int_this, alpha=alpha, height_control='fdr',contrast='this',output_path=out_path)

#Correcting for multiple comparisons - FWER 
#- this function both computes thresholded maps, figures, and cluster information 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap_group_this_fwer')
threshold_map_and_plot_func(zmap_int_this, alpha=alpha, height_control='bonferroni',contrast='this',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_group_this_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_this, this_baseline_con_imgs, design_matrix, contrast='this',file_name=file_name)



#=========================================================================================#
#--------------------------That vs. baseline contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models of group difference for contrast: "that vs basline"----------')

print('- Fitting second level model')

slm_int_that = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_that = slm_int_that.fit(
    that_baseline_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: group')

zmap_int_that = slm_int_that.compute_contrast(
    second_level_contrast='group',
    output_type='z_score',
)
data = zmap_int_that.get_fdata()
print(data.max()) # 4.47
print(data.min()) # -4.25
# fdr threshold (0.001) = inf 
# fdr threshold (0.05) = inf 
# bonferroni threshold (0.001) = 5.84
# bonferroni threshold (0.05) = 5.15

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_that, cmap='jet', black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group difference - "that" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_group_that_uncor_glassZmap.png', facecolor='k', edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_that, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group difference - "that" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_group_that_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_group_that_fdr'
threshold_map_and_plot_func(zmap_int_that, alpha=alpha, height_control='fdr',contrast='that',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap_group_that_fwer')
threshold_map_and_plot_func(zmap_int_that, alpha=alpha, height_control='bonferroni',contrast='that',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_group_that_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_that, that_baseline_con_imgs, design_matrix, contrast='that',file_name=file_name)


#=========================================================================================#
#-------------------------- This + that contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models of group difference for contrast: "this + that"----------')

print('- Fitting second level model')

slm_int_this_plus_that = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_this_plus_that= slm_int_this_plus_that.fit(
    this_plus_that_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: group')

zmap_int_this_plus_that = slm_int_this_plus_that.compute_contrast(
    second_level_contrast='group',
    output_type='z_score',
)
data = zmap_int_this_plus_that.get_fdata()
print(data.max()) # 4.36 
print(data.min()) # -4.15
# fdr threshold (0.001) = inf 
# fdr threshold (0.05) = inf
# bonferroni threshold (0.001) = 5.84
# bonferroni threshold (0.05) = 5.15

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_this_plus_that, cmap='jet',black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group difference - "this + that" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_group_this_plus_that_uncor_glassZmap.png', facecolor='k',edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_this_plus_that, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group difference - "this + that" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_group_this_plus_that_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_group_this_plus_that_fdr'
threshold_map_and_plot_func(zmap_int_this_plus_that, alpha=alpha, height_control='fdr',contrast='this+that',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap_group_this_plus_that_fwer')
threshold_map_and_plot_func(zmap_int_this_plus_that, alpha=alpha, height_control='bonferroni',contrast='this+that',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_group_this_plus_that_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_this_plus_that, this_plus_that_con_imgs, design_matrix, contrast='this+that',file_name=file_name)



#=========================================================================================#
#--------------------------This-that contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models of group difference for contrast: "this - that"----------')

print('- Fitting second level model')

slm_int_this_that = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_this_that = slm_int_this_that.fit(
    this_that_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: group')

zmap_int_this_that = slm_int_this_that.compute_contrast(
    second_level_contrast='group',
    output_type='z_score',
)
data = zmap_int_this_that.get_fdata()
print(data.max()) # 4.29 
print(data.min()) # -3.81
# fdr threshold (0.001) = inf 
# fdr threshold (0.05) = inf 
# bonferroni threshold (0.001) = 5.84 
# bonferroni threshold (0.05) = 5.15

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_this_that, cmap='jet',black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group difference - "this-that" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_group_this-that_uncor_glassZmap.png', facecolor='k',edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_this_that, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group difference - "this-that" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_group_this-that_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_group_this-that_fdr'
threshold_map_and_plot_func(zmap_int_this_that, alpha=alpha, height_control='fdr',contrast='this-that',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap_group_this-that_fwer')
threshold_map_and_plot_func(zmap_int_this_that, alpha=alpha, height_control='bonferroni',contrast='this-that',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_group_this-that_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_this_that, this_that_con_imgs, design_matrix, contrast='this-that',file_name=file_name)





#=========================================================================================#
#--------------------------That vs. baseline contrast ----------------# 
#=========================================================================================#


print('-------- Running second level models of group difference for contrast: "that-this"----------')

print('- Fitting second level model')

slm_int_that_this = SecondLevelModel(smoothing_fwhm=8.0)
slm_int_that_this = slm_int_that_this.fit(
    that_this_con_imgs,
    design_matrix=design_matrix)

print('- Estimating contrast: group')

zmap_int_that_this = slm_int_that_this.compute_contrast(
    second_level_contrast='group',
    output_type='z_score',
)
data = zmap_int_that_this.get_fdata()
print(data.max()) # 3.81
print(data.min()) # -4.29
# fdr threshold (0.001) = inf 
# fdr threshold (0.05) = inf 
# bonferroni threshold (0.001) = 5.84
# bonferroni threshold (0.05) = 5.15 

#Plot zmap from each estimation method next to each other 
p001_unc = norm.isf(0.001)

print('- Plotting uncorrected zmaps on glass brain')

plotting.plot_glass_brain(zmap_int_that_this, cmap='jet',black_bg=True, display_mode='lyrz',colorbar=True, threshold=p001_unc,
                          title='Group difference - "that-this" contrast (unc p<0.001)',
                          plot_abs=False)
plt.show()
plt.savefig(output_dir+'/slm_group_that-this_uncor_glassZmap.png', facecolor='k',edgecolor='k')

print('- Plotting uncorrected stats map')

plot_stat_map(zmap_int_that_this, cmap='jet',threshold=p001_unc, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=True,
              title='Group difference - "that-this" contrast (unc p<0.001)')
plt.show()
plt.savefig(output_dir+'/slm_group_that-this_uncor_statMap.png', facecolor='k',edgecolor='k')

#Correcting for multiple comparisons - FDR 

print('- Running FDR correction')

out_path = output_dir+'/zmap_group_that-this_fdr'
threshold_map_and_plot_func(zmap_int_that_this, alpha=alpha, height_control='fdr',contrast='that-this',output_path=out_path)

#Correcting for multiple comparisons - FWER 

print('- Running Bonferroni correction')

out_path = os.path.join(output_dir, 'zmap_group_that-this_fwer')
threshold_map_and_plot_func(zmap_int_that_this, alpha=alpha, height_control='bonferroni',contrast='that-this',output_path=out_path)


#------ Parametric and non-parametric tests and plots -------- # 

print('- Running parametric and non-parametric tests')

file_name = output_dir+'/slm_group_that-this_parametricVSnonparametric_tests.png'
permutation_test_and_figure(slm_int_that_this, that_this_con_imgs, design_matrix, contrast='that-this',file_name=file_name)


