# Factor rotation using library from: https://github.com/SungjoonPark/FactorRotation (Park et al., 2017)

import pandas as pd 
import os 
import numpy as np 

#-------------------Load glove vectors of word in DCT ---------------------#
dir = os.getcwd()

#Load glove semantic space 
embed_dict = {}
with open('/Users/au553087/Dropbox/DeixisSurvey2/glove/glove.6B.300d.txt') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:],'float32')
    embed_dict[word]=vector

#Extract glove vector for each word and store in df
raw_resp = pd.read_csv(os.path.join(dir, 'classification/dep_classification/features_dfs/raw_responses.csv'), index_col=False)
words = raw_resp.loc[:,"day":"loan"].columns

word_vecs = pd.DataFrame([embed_dict.get(w) for w in raw_resp.columns[0:293]])
word_vecs['word'] = words

word_vecs.to_csv(os.path.join(dir, 'classification/dep_classification/features_dfs/words_glove300_vecs.csv'), index=False)
word_vecs = pd.read_csv(os.path.join(dir, 'classification/dep_classification/features_dfs/words_glove300_vecs.csv'), index_col=False)
#--------------- Compute rotated word vectors -------------------#
### NB: does not converge - probably too few words compared to features 

from pathlib import Path
import sys
path_root = os.path.join(dir, 'classification/dep_classification/compute_factor_rotation/')
sys.path.append(str(path_root))

import factor_rotation as fr 
import numpy as np
import torch


#Turn word vectors df into matrix 
word_vecs_mat = word_vecs.iloc[:,0:300].to_numpy()

#Normalize/rescale vectors to improve rotation performance
word_vecs_mat /= np.max(np.abs(word_vecs_mat)) 

#Varimax rotation 
L, T = fr.rotate_factors(word_vecs_mat, 'varimax', dtype=torch.float64, device=torch.device("cpu"))

#Parsimony rotation 

#Parsimax rotation 

#Quartimax rotation 

print(L)
print(T)


#--------------- Compute rotated word vectors on all glove words -------------------#
#Park et al., 2017 show that for glove vectors, the best performances (intruder test - DR_overall) are obtained with varimax orthogonal and quartimax oblique. 
#For semantic analogy tasks the best performances were from varimax oblique and quartimax oblique 


#Convert glove dict to matrix
orderedNames = embed_dict.keys()
glove_mat = np.array([embed_dict[i] for i in orderedNames])

#Rescale 
glove_mat /= np.max(np.abs(glove_mat)) 

#Rotation (varimax orthogonal)
L, T = fr.rotate_factors(glove_mat, 'varimax', dtype=torch.float64, device=torch.device("cpu"))

df = pd.DataFrame(L)
df['word'] = orderedNames
df.to_csv(os.path.join(dir, 'classification/dep_classification/compute_factor_rotation/rotated_glove_vecs_varimax_orthogonal.csv'))
df = pd.read_csv(os.path.join(dir, 'classification/dep_classification/compute_factor_rotation/rotated_glove_vecs_varimax_orthogonal.csv'), index_col=False)

#----------- Compute pairwise correlation between DCT words from rotated vectors ----------# 
dct_words = [w for w in word_vecs.word]

rotated_dct_words = df.loc[df['word'].isin(dct_words)].iloc[:,1:]

cor_mat = rotated_dct_words.iloc[:,0:300].T.corr()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 25))
sns.heatmap(cor_mat, xticklabels=dct_words, yticklabels=dct_words)
#plt.legend(title='Pairwise correlation (varimax orthogonal rotation)')
plt.savefig(os.path.join(dir, 'classification/dep_classification/compute_factor_rotation/rotated_glove_vecs_varimax_orthogonal_corMatrix.png'))


#--------------- Calculate pairwise cosine distance of dct words in rotated space -------------#
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

cosine_sim = 1-pairwise_distances(rotated_dct_words.iloc[:,0:300], metric="cosine")
plt.figure(figsize=(25, 25))
sns.heatmap(cosine_sim, xticklabels=dct_words, yticklabels=dct_words)
#plt.legend(title='Pairwise correlation (varimax orthogonal rotation)')
plt.savefig(os.path.join(dir, 'classification/dep_classification/compute_factor_rotation/rotated_glove_vecs_varimax_orthogonal_cosineDistMatrix.png'))


#------------- Cluster words with k-means from rotated vector space --------------#
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


model = KMeans(n_clusters=10)
model.fit(rotated_dct_words.iloc[:,0:300])
yhat = model.predict(rotated_dct_words.iloc[:,0:300])
rotated_dct_words['cluster'] = yhat

#Cluster 1
c1_words = rotated_dct_words.loc[rotated_dct_words.cluster==1].word
#['elm', 'sweetness', 'brightness', 'rake', 'bonfire', 'eggplant', 'lemonade', 'shit', 'loner', 'crap', 'delirium', 'coolness', 'firework', 'whiteness', 'homesickness', 'squeal', 'joviality']

c2_words = rotated_dct_words.loc[rotated_dct_words.cluster==2].word
#['damage', 'storm', 'highway', 'kill', 'accident', 'plot', 'tourist', 'riot', 'subway', 'cloud', 'scare', 'cyclone', 'volcano', 'tornado', 'stampede', 'avoidance']

c3_words = rotated_dct_words.loc[rotated_dct_words.cluster==3].word
#['fish', 'animal', 'dog', 'chicken', 'cooking', 'zoo', 'egg', 'ham', 'jungle', 'duck', 'bee', 'pie', 'snake', 'honey', 'monkey', 'turtle', 'pumpkin', 'carrot', 'pineapple']

c4_words = rotated_dct_words.loc[rotated_dct_words.cluster==4].word
#['man', 'car', 'band', 'woman', 'room', 'movie', 'hall', 'hotel', 'lake', 'someone', 'couple', 'friend', 'van', 'beach', 'boy', 'mountain', 'store', 'giant', 'table', 'door', 'actor', 'bar', 'eye', 'artist', 'foot', 'stone', 'boat', 'soldier', 'color', 'screen', 'truck', 'garden', 'arm', 'hair', 'dinner', 'coffee', 'window', 'camera', 'chair', 'shoulder', 'beer', 'drink', 'dress', 'cathedral', 'sight', 'kitchen', 'tribute', 'sand', 'nose', 'rent', 'girlfriend', 'finger', 'flower', 'kiss', 'choir', 'boyfriend', 'shoe', 'shelf', 'cab', 'carriage', 'lamp', 'cafeteria']

c5_words = rotated_dct_words.loc[rotated_dct_words.cluster==5].word
#['love', 'thing', 'sense', 'feeling', 'fun', 'wonder', 'mystery', 'luck', 'hell', 'humor', 'joke', 'darkness', 'excuse', 'shame', 'grief', 'happiness', 'irony', 'clue', 'scream', 'distraction', 'satire', 'awe', 'paradox', 'envy', 'greatness', 'dime', 'goodness', 'analogy', 'jealousy', 'thirst', 'dread', 'abyss', 'deceit', 'stupidity', 'torment', 'emptiness', 'woe']

c6_words = rotated_dct_words.loc[rotated_dct_words.cluster==6].word
#['health', 'hospital', 'care', 'doctor', 'pain', 'patient', 'depression', 'sleep', 'hunger', 'hygiene', 'sickness', 'relaxation']

c7_words = rotated_dct_words.loc[rotated_dct_words.cluster==7].word
#['world', 'team', 'game', 'top', 'play', 'win', 'cup', 'college', 'football', 'field', 'goal', 'ball', 'basketball', 'sport', 'tennis']

c8_words = rotated_dct_words.loc[rotated_dct_words.cluster==8].word
#['day', 'company', 'group', 'minister', 'high', 'part', 'home', 'party', 'friday', 'court', 'family', 'life', 'number', 'month', 'use', 'area', 'place', 'right', 'law', 'system', 'office', 'meeting', 'money', 'army', 'line', 'death', 'election', 'need', 'human', 'rose', 'change', 'interest', 'island', 'radio', 'summer', 'act', 'problem', 'cost', 'drug', 'situation', 'thought', 'study', 'battle', 'class', 'society', 'computer', 'question', 'step', 'addition', 'plant', 'speech', 'paper', 'comment', 'cabinet', 'lawyer', 'guard', 'debate', 'type', 'era', 'strategy', 'advantage', 'package', 'reading', 'loan', 'poverty', 'pilot', 'victim', 'journalist', 'worker', 'diplomat', 'ease', 'fate', 'engineer', 'conversation', 'businessman', 'complaint', 'scientist', 'routine', 'vacation', 'voter', 'burden', 'noise', 'semester']

c9_words = rotated_dct_words.loc[rotated_dct_words.cluster==9].word
#['criminal', 'freedom', 'sex', 'trust', 'knowledge', 'faith', 'truth', 'religion', 'testimony', 'wealth', 'belief', 'hate', 'refusal', 'plea', 'optimism', 'sin', 'guilt', 'motive', 'denial', 'hierarchy', 'bribe', 'legality', 'perjury']

c10_words = rotated_dct_words.loc[rotated_dct_words.cluster==10].word
# None 

#---------------- Find top dct words on each feature ------------------#
features = []
max_20_all = []
for i in range(0,300):
  feat = rotated_dct_words.columns[i]
  features.append(feat)
  max_20 = rotated_dct_words.nlargest(20, columns=feat).word
  max_20 = [w for w in max_20]
  max_20_all.append(max_20)

max_df = pd.DataFrame(max_20_all)
max_df['feat'] = features

#--------------- Find top features for each dct word ------------------#
for w in dct_words: 
  new_df = rotated_dct_words.iloc[:,0:300].T 
  new_df.columns = rotated_dct_words.word
  max_10 = new_df.nlargest(10, columns=w).sort_values(w, ascending=False)
  index.to_numpy()



