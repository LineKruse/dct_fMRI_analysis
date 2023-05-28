import os 
import numpy as np 
import pandas as pd 
import pickle 
from nilearn.image import new_img_like, load_img, index_img, clean_img
from sklearn.model_selection import train_test_split, GroupKFold
from nilearn.image import index_img, concat_imgs
from numpy import array
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, RepeatVector, TimeDistributed, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers



###########################################################################################################
#                                          Define subject list                                            #
###########################################################################################################

#Define subject list 
dir = os.getcwd()
zmaps_dir = os.path.join(dir, 'classification/behav_responses/output/flm_objects/')
dirlist = os.listdir(zmaps_dir)
sublist = [sub[0:8] for sub in dirlist if 'zmaps' in sub]

###########################################################################################################
#                           Loop through subjects, load zmaps, run 3D CNN                                 #
###########################################################################################################

for sub in sublist: 

    print('------------------------- Running ',sub,'------------------------------')

#----------------------- Load models, design matrix, conditions_labels and zmaps---------------------------

    f = open(os.path.join('/users/line/dct_fMRI_analysis/classification/behav_responses/output/flm_objects/'+f'{sub}_flm_zmaps.pkl'), 'rb')
    model, tbt_dm, conditions_label, z_maps = pickle.load(f)
    f.close() 

#---------------------------------- Reshape data for classification ---------------------------------------

    #Reshaping data
    idx_this=[int(i) for i in range(len(conditions_label)) if conditions_label[i]=='this']
    idx_that=[int(i) for i in range(len(conditions_label)) if conditions_label[i]=='that']

    #Concatenate trial list 
    idx=np.concatenate((idx_this, idx_that))

    #Concatenate zmaps and order according to trial list 
    conditions=np.array(conditions_label)[idx]
    z_maps_conc=concat_imgs(z_maps)
    z_maps_img = index_img(z_maps_conc, idx)
    #shape: 109, 129, 108, 303 

#---------------------------------- Create train-test splits ---------------------------------------
    idx2=np.arange(conditions.shape[0])
    
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(idx2, conditions,
                                                              stratify = conditions,
                                                              test_size=0.3,
                                                              random_state=42)
    
    #Assign zmaps to X_train and X_test by split indices 
    X_train = index_img(z_maps_img, X_train_idx) #109, 129, 108, 212
    X_test = index_img(z_maps_img, X_test_idx) #109, 129, 108, 91 

#----------------------------- Build 3D CNN model  ---------------------------------------
def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()