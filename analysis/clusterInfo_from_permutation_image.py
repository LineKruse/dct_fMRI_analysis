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

input_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output'

#Load this - that first-level contrast images 
contrast_dir = 'fl_con_zmap_this-that'
#contrast_dir = 'fl_con_zmap_this_plus_that'
#contrast_dir = 'flcon_zmap_that_vs_baseline'
#contrast_dir = 'flcon_zmap_this_vs_baseline'
files = os.listdir(os.path.join(input_dir, contrast_dir))
this_that_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    this_that_con_imgs.append(img)


#Define design_matrix
from nilearn.glm.second_level import SecondLevelModel

#Define design matrix (intercept model only)
design_matrix = pd.DataFrame(
    [1] * len(this_that_con_imgs),
    columns=['intercept'],
)

# #Fit second level model 
# slm_this_plus_that = SecondLevelModel(smoothing_fwhm=8.0)
# slm_this_plus_that = slm_this_plus_that.fit(
#     this_that_con_imgs,
#     design_matrix=design_matrix)

# print('- Estimating contrast: intercept')

# zmap_this_plus_that = slm_this_plus_that.compute_contrast(
#     second_level_contrast='intercept',
#     output_type='z_score',
# )
# data = zmap_this_plus_that.get_fdata()

#Plot zmap with coordinates on the cluster of interest (from this-that contrast)
# cut_coords = [-3.628, -73.562, 36.7]
# p001_unc = norm.isf(0.001)

# #output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/cluster_results/this/permutation/'
# #output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/cluster_results/that/permutation/'
# output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/cluster_results/this-that/permutation/'


# plotting.plot_stat_map(zmap_this_plus_that, colorbar=True, black_bg=True, cmap='jet', cut_coords=cut_coords,threshold=p001_unc)
# plt.savefig(output_dir+'this-that_uncor_peakLPrecun_statZmap.png',facecolor='k', edgecolor='k')
# plt.show()

# #Plot value at this coordinate for each subject in bar-diagram 
# from nilearn.maskers import NiftiSpheresMasker
# from itertools import chain
# os.system('pip install seaborn')
# import seaborn as sns
# coords = [(-3.628, -73.562, 36.7)]
# masker = NiftiSpheresMasker(coords, radius=200) #sphere with single voxel (on peak coordinates)
# masker = NiftiSpheresMasker(coords, radius=200) #sphere with radius of 200 mm around peak coordinates
# mask_all_subs = masker.fit_transform(this_that_con_imgs)
# mask_all_subs = np.array(list(chain.from_iterable(mask_all_subs)))


# An "interface" to matplotlib.axes.Axes.hist() method
# plt.figure()
# n, bins, patches = plt.hist(x=mask_all_subs, bins=20, color='#0504aa', alpha=0.7)
# #plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Mean Voxel Signal Per Subject within sphere')
# plt.axvline(x = mask_all_subs.mean(), color = 'orange', linestyle = '--', label='Mean')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# #plt.savefig(output_dir+'this_plus_that_uncor_spherePeakVox_hist.png')
# plt.savefig(output_dir+'this_plus_that_uncor_sphere200mm_hist.png')
# plt.show()

# display = plotting.plot_stat_map(zmap_this_plus_that, threshold=p001_unc, title='Precuneus Sphere Mask',
#                                  cut_coords=cut_coords)
# display.add_markers(marker_coords=[cut_coords], marker_color='g',
#                     marker_size=200, alpha=0.3)
# display.savefig(output_dir+'precuneus_cluster_mask.png')

# -------------- Run cluster detection on permutation map -------------------#

#First - Run permutation test to get thresholded map 
from nilearn.glm.second_level import non_parametric_inference 
n_perm = 1000 

output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output_new/'
#output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/cluster_results/this+that/permutation/'
#output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/cluster_results/that/permutation/'
#output_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/output/sl_intercept_models_alpha-001/cluster_results/this/permutation/'


#Calculate p-values using permutations 
second_level_input = this_that_con_imgs
out_dict_thisplusthat = non_parametric_inference(
    second_level_input,
    design_matrix=design_matrix,
    model_intercept=True,
    n_perm=n_perm,  
    two_sided_test=False,
    smoothing_fwhm=8.0,
    n_jobs=1, 
    )
nib.save(out_dict_thisplusthat, output_dir+'perm_Nifti_thisplusthat.nii')

#--------Plot with threshold p < 0.05 
threshold_log = round(-np.log10(1 / 20))  # p < 0.05 - take into account that effects are now negative log of the p-value 
vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
cut_coords = [0]

plotting.plot_glass_brain(out_dict_thisplusthat,colorbar=True,vmax=vmax,plot_abs=False, black_bg=True, display_mode='lyrz',cmap='jet',cut_coords=cut_coords,threshold=threshold_log)
plt.savefig(output_dir+'thisplusthat_perm05_glassZmap.png',facecolor='k', edgecolor='k')
plt.show()

plotting.plot_stat_map(out_dict_thisplusthat, colorbar=True, vmax=vmax, black_bg=True, cmap='jet',cut_coords=[-30,-20,-10,0,10,20,30],display_mode='z', threshold=threshold_log)
plt.savefig(output_dir+'thisplusthat_perm05_statZmap.png',facecolor='k', edgecolor='k')
plt.show()


#-------- Plot with threshold p < 0.01 
threshold_log = round(-np.log10(1 / 100))
vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
cut_coords = [0]

plotting.plot_glass_brain(out_dict_thisplusthat,colorbar=True,vmax=vmax,plot_abs=False, black_bg=True, display_mode='lyrz',cmap='jet',cut_coords=cut_coords,threshold=threshold_log)
plt.savefig(output_dir+'thisplusthat_perm01_glassZmap.png',facecolor='k', edgecolor='k')
plt.show()

plotting.plot_stat_map(out_dict_thisplusthat, colorbar=True, vmax=vmax, black_bg=True, cmap='jet',cut_coords=[-30,-20,-10,0,10,20,30],display_mode='z', threshold=threshold_log)
plt.savefig(output_dir+'thisplusthat_perm01_statZmap.png',facecolor='k', edgecolor='k')
plt.show()


#-------- Plot with threshold p < 0.001 
threshold_log = round(-np.log10(1 / 1000))
vmax = round(-np.log10(1 / n_perm)) # minimal p-value possible with the number of permuations
cut_coords = [0]

plotting.plot_glass_brain(out_dict_thisplusthat,colorbar=True,vmax=vmax,plot_abs=False, black_bg=True, display_mode='lyrz',cmap='jet',cut_coords=cut_coords,threshold=threshold_log)
plt.savefig(output_dir+'thisplusthat_perm001_glassZmap.png',facecolor='k', edgecolor='k')
plt.show()

plotting.plot_stat_map(out_dict_thisplusthat, colorbar=True, vmax=vmax, black_bg=True, cmap='jet',cut_coords=[-30,-20,-10,0,10,20,30],display_mode='z', threshold=threshold_log)
plt.savefig(output_dir+'thisplusthat_perm001_statZmap.png',facecolor='k', edgecolor='k')
plt.show()


#---------- Run cluster detection with threshold p<0.001 (this + that contrast)
from atlasreader import create_output
threshold_log = round(-np.log10(1 / 1000))
create_output(
    filename = out_dict_thisplusthat,
    atlas='default',
    #voxel_thresh=1.96,
    voxel_thresh=threshold_log,
    direction='both',
    cluster_extent=20,
    glass_plot_kws = {'black_bg':True,'colorbar':True, vmax:5,'cmap': 'jet'},
    stat_plot_kws = {'black_bg':True,'cmap': 'jet','title':False},
    outdir=output_dir+"cluster_thisplusthat_p001"
    )

#Define sphere mask 
coords = [(-3, -73, 36)]
masker1v = NiftiSpheresMasker(coords) #sphere with single voxel (on peak coordinates)
masker200 = NiftiSpheresMasker(coords, radius=200) #sphere with radius of 200 mm around peak coordinates
mask_imgs_this_1v = masker200.fit_transform(out_dict_thisplusthat)


mask_imgs_this_1v = np.array(list(chain.from_iterable(mask_imgs_this_1v))) 


#------ Extract precuneus cluster values per subject for this>baseline, that>baseline and this-that, and barplot ----- # 
from nilearn.maskers import NiftiSpheresMasker
from itertools import chain
os.system('pip install seaborn')
import seaborn as sns

#Load files for all three contrasts 
contrast_dir = 'fl_con_zmap_this-that'
files = os.listdir(os.path.join(input_dir, contrast_dir))
this_that_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    this_that_con_imgs.append(img)

contrast_dir = 'flcon_zmap_that_vs_baseline'
files = os.listdir(os.path.join(input_dir, contrast_dir))
that_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    that_con_imgs.append(img)

contrast_dir = 'flcon_zmap_this_vs_baseline'
files = os.listdir(os.path.join(input_dir, contrast_dir))
this_con_imgs = []
for file in files: 
    img = nib.load(os.path.join(input_dir, contrast_dir, file))
    this_con_imgs.append(img)

#Define sphere mask 
coords = [(-3.628, -73.562, 36.7)]
masker1v = NiftiSpheresMasker(coords) #sphere with single voxel (on peak coordinates)
masker200 = NiftiSpheresMasker(coords, radius=200) #sphere with radius of 200 mm around peak coordinates

#Extract values - this contrast 
mask_imgs_this_1v = masker1v.fit_transform(this_con_imgs)
mask_imgs_this_1v = np.array(list(chain.from_iterable(mask_imgs_this_1v)))

mask_imgs_this_200 = masker200.fit_transform(this_con_imgs)
mask_imgs_this_200 = np.array(list(chain.from_iterable(mask_imgs_this_200)))

#Extract values - that contrast 
mask_imgs_that_1v = masker1v.fit_transform(that_con_imgs)
mask_imgs_that_1v = np.array(list(chain.from_iterable(mask_imgs_that_1v)))

mask_imgs_that_200 = masker200.fit_transform(that_con_imgs)
mask_imgs_that_200 = np.array(list(chain.from_iterable(mask_imgs_that_200)))

#Extract values - this-that contrast 
mask_imgs_thisthat_1v = masker1v.fit_transform(this_that_con_imgs)
mask_imgs_thisthat_1v = np.array(list(chain.from_iterable(mask_imgs_thisthat_1v)))

mask_imgs_thisthat_200 = masker200.fit_transform(this_that_con_imgs)
mask_imgs_thisthat_200 = np.array(list(chain.from_iterable(mask_imgs_thisthat_200)))

#Bind in one dataframe 
precVals1v = pd.DataFrame()
precVals1v['proximal>baseline'] = mask_imgs_this_1v
precVals1v['distal>baseline'] = mask_imgs_that_1v
precVals1v['proximal-distal'] = mask_imgs_thisthat_1v
precVals1v['test'] = precVals1v['proximal>baseline']-precVals1v['distal>baseline']

precVals200 = pd.DataFrame()
precVals200['proximal>baseline'] = mask_imgs_this_200
precVals200['distal>baseline'] = mask_imgs_that_200
precVals200['proximal-distal'] = mask_imgs_thisthat_200
precVals200['test'] = precVals200['proximal>baseline']-precVals200['distal>baseline']

plt.figure()
precVals1v.boxplot(column = ['proximal>baseline', 'distal>baseline', 'proximal-distal']) 
plt.savefig(output_dir+'prec_mask_boxplot_1v.png')

plt.figure()
precVals200.boxplot(column = ['proximal>baseline', 'distal>baseline', 'test']) 
plt.savefig(output_dir+'prec_mask_boxplot_200.png')

# #Plot with peak coordinates for precuneus cluster in this-that contrast 
# cut_coords = [-3.628, -73.562, 36.7]
# plotting.plot_stat_map(out_dict, colorbar=True, vmax=vmax, black_bg=True, cmap='jet', cut_coords=cut_coords,threshold=threshold_log)
# plt.savefig(output_dir+'that_perm_LPrecun_statZmap.png',facecolor='k', edgecolor='k')
# plt.show()


# #Second - use create_output funtionc of atlasreader to get cluster results 
# os.system('pip install atlasreader')
# from atlasreader import create_output


# # Step 3: Create output 
# create_output(
#     filename = out_dict,
#     atlas='default',
#     #voxel_thresh=1.96,
#     voxel_thresh=threshold_log,
#     direction='both',
#     cluster_extent=20,
#     glass_plot_kws = {'black_bg':True,'vmax':vmax,'colorbar':True, 'cmap': 'jet'},
#     stat_plot_kws = {'black_bg':True,'cmap': 'jet','title':False},
#     outdir=output_dir+"that_clustExtent-5"
#     )

# #Investigate coordinate from this-that result cluster in zmap of this+that contrast
