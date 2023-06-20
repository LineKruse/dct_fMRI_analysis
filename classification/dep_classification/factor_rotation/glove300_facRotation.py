# Factor rotation using library from: https://github.com/SungjoonPark/FactorRotation (Park et al., 2017)

import pandas as pd 
import os 
import numpy as np 

#-------------------Load glove vectors of word in DCT ---------------------#
dir = os.getcwd()

#Load glove semantic space 
embed_dict = {}
with open(os.path.join(dir, 'classification/dep_classification/features_dfs/glove.6B.300d.txt','r')) as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:],'float32')
    embed_dict[word]=vector

#Extract "this" and "that" aggregate vector for each subject 
raw_resp = pd.read_csv('/users/line/dct_fMRI_analysis/classification/dep_classification/features_dfs/raw_responses.csv', index_col=False)

vec_df = pd.DataFrame(columns=np.arange(0,600))
columns = vec_df.columns

for i in range(0, raw_resp.shape[0]):
    t_this = raw_resp[raw_resp.columns[raw_resp.iloc[i] == 'this']].columns
    t_this = [w for w in t_this if 'rep' not in w]
    vecs_this = pd.DataFrame([embed_dict.get(w) for w in t_this])
    avg_vec_this = np.array(vecs_this.mean(axis=0))

    t_that = raw_resp[raw_resp.columns[raw_resp.iloc[i] == 'that']].columns 
    t_that = [w for w in t_that if 'rep' not in w]
    vecs_that = pd.DataFrame([embed_dict.get(w) for w in t_that])
    avg_vec_that = np.array(vecs_that.mean(axis=0)) 

    avg_vec = np.concatenate((avg_vec_this, avg_vec_that))

    zipped = zip(columns, avg_vec)
    output_dict = dict(zipped)
    vec_df = vec_df.append(output_dict, ignore_index=True)

vec_df['ID'] = raw_resp.ID
vec_df['gender'] = raw_resp.gender
vec_df['age'] = raw_resp.age
vec_df['dep_group'] = raw_resp.dep_group
vec_df['PHQ9_sum'] = raw_resp.PHQ9_sum

vec_df.to_csv('/users/line/dct_fMRI_analysis/classification/dep_classification/features_dfs/embed_responses.csv')
