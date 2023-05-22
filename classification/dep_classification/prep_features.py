import pandas as pd 
import numpy as np 
import os 

###########################################################################################################
#                                          Raw responses                                                  #
###########################################################################################################


#---------------------Load and prep behavioral files -------------------------#

df_behav = pd.read_csv('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/cleaned_behav_and_survey_data/behav_all.csv')

#Remove subject 039 - only have 201 trials in logfile 
#Remove subject 009 - only have 203 trials in logfile  
exclude = [39, 9]
df_behav = df_behav[~df_behav['ID'].isin(exclude)] #n=73

#Transpose to one-vector subject representation 
col_list = [w for w in (df_behav['word'].unique())]+['ID','gender','age','dep_group']
df = pd.DataFrame(index=range(len(df_behav['ID'].unique())),columns=range(len(col_list)))
df.columns = col_list
for i in range(0,len(df_behav['ID'].unique())):
    sub = df_behav['ID'].unique()[i]
    print("Running sub", str(sub))
    df_sub = df_behav.loc[df_behav['ID']==sub][['word','choice']]
    df_sub = df_sub.set_index('word').T
    df_sub['ID'] = sub
    df_sub['gender'] = df_behav['Gender'].loc[df_behav['ID']==sub].unique()[0]
    df_sub['age'] = df_behav['Age'].loc[df_behav['ID']==sub].unique()[0]
    df_sub['dep_group'] = [1 if len(str(sub))>2 else 0]
    vec = [el for el in df_sub.iloc[0,:]]
    df.iloc[i,:] = vec


#Load survey data (PHQ9 og PID5)
df_phq = pd.read_csv('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/cleaned_behav_and_survey_data/survey_all.csv')

#Select only subjects also present in behav data 
id_list = [id for id in df.ID]
df_phq = df_phq[df_phq['ID'].isin(df.ID)]
#Add empty row for subject 111 - we don't have survey data from this sub 
random_row = df_phq.iloc[0,:]
random_row.iloc[:] = np.nan
random_row.ID = 111
df_phq = df_phq.append(random_row)

#Add the PHQ9_sum to behav data 
df = pd.merge(df, df_phq[['ID','PHQ9_sum']], on="ID")
df.to_csv('/users/line/dct_fMRI_analysis/classification/dep_classification/features_dfs/raw_responses.csv', index=False)

###########################################################################################################
#                                          PCA behav responses                                            #
###########################################################################################################
from sklearn.decomposition import PCA
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

raw_resp = pd.read_csv('/users/line/dct_fMRI_analysis/classification/dep_classification/features_dfs/raw_responses.csv', index_col=False)
pca_input = raw_resp.loc[:,:'loan']

encoder = ce.OrdinalEncoder(handle_missing='value')
pca_input = encoder.fit_transform(pca_input)

pca = PCA(n_components=70)
pca_resp = pca.fit_transform(pca_input)
pca_resp = pd.DataFrame(pca_resp)

pca_resp['ID'] = raw_resp.ID
pca_resp['gender'] = raw_resp.gender
pca_resp['age'] = raw_resp.age
pca_resp['dep_group'] = raw_resp.dep_group
pca_resp['PHQ9_sum'] = raw_resp.PHQ9_sum

pca_resp.to_csv(os.path.join(dir, 'classification/dep_classification/features_dfs/pca_responses.csv'), index=False)

###########################################################################################################
#                                       Word embedding representations                                    #
###########################################################################################################
#For each subject, generate an averaged vector representation for words where subject chose “this” 
# and words where subject chose “that” → concatenate into one 600-dimensional vector representation 
# for that subject 

#Load glove semantic space 
embed_dict = {}
with open('/users/line/dct_fMRI_analysis/classification/dep_classification/features_dfs/glove.6B.300d.txt','r') as f:
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

    
###########################################################################################################
#                                            Text features from LSTM                                      #
###########################################################################################################
#Treat all trials (stimulus and response) as one long sentence ("this cat that house that party this day...")
#Apply LSTM to extract features from the text - use as input for classification
#Treated as natural language processing problem  

#Create text representations for each subject 
dir = os.getcwd()
path = os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv')
raw_resp = pd.read_csv(path, index_col=False)

def generateList(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
                      for sub in [lst1, lst2]]

text_df = pd.DataFrame(columns=np.arange(0,293))
columns = text_df.columns
for i in range(0, raw_resp.shape[0]):
    print(i)
    l1 = [str(col) for col in raw_resp.loc[:,:'loan'].columns]
    l2 = [str(resp) for resp in raw_resp.loc[:,:'loan'].iloc[i,:]]
    text = generateList(l2, l1)
    text = [' '.join(text[ii:ii+2]) for ii in np.arange(0,586,2)]
    #text = [" ".join(text)]
    zipped = zip(columns, text)
    output_dict = dict(zipped)
    text_df = text_df.append(output_dict, ignore_index=True)

text_df.to_csv(os.path.join(dir,'classification/dep_classification/features_dfs/text_seq_responses.csv'), index=False)

#Trian LSTM autoencoder 
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow import keras

seq_list = []
for i in range(0, text_df.shape[0]):
   text = [str(w) for w in text_df.iloc[i,:]]
   #text = ' '.join(text)
   seq_list.append(text)

#Integer encode text sequences 
t = Tokenizer()
t.fit_on_texts(seq_list)
seq_encoded = t.texts_to_sequences(seq_list)
seq_encoded_df = pd.DataFrame(seq_encoded)
seq_encoded_df.to_csv(os.path.join(dir,'classification/dep_classification/features_dfs/text_seq_encoded_responses.csv'), index=False)

word_to_index_dict = t.index_word

#Ordinal encoder
import category_encoders as ce 
#ord_encoder = ce.OrdinalEncoder(handle_missing='value')
#seq_encoded_df = ord_encoder.fit_transform(seq_list)
#bin_encoder = ce.BinaryEncoder(handle_missing='value')
#seq_encoded_df = bin_encoder.fit_transform(seq_list)

#Scale input 
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = StandardScaler() # Gives range (-0.9, 7.9)
#scaler = RobustScaler() # Gives range (-1, 596)
input_scaled = scaler.fit_transform(seq_encoded_df)

#3D input shape: samples (n sequences), timesteps (n words), features (n feature types - we have one: words)
input = input_scaled.reshape(73, 293, 1)

n_timesteps = input.shape[1]
n_features = input.shape[2]

#Define model and fit 
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
#model.add(Dropout(0.1))
model.add(RepeatVector(n_timesteps))
#model.add(Dropout(0.1))
model.add(LSTM(100, activation='relu', return_sequences=True))
#model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(n_features)))
opt = keras.optimizers.Adam(clipnorm=1.)
model.compile(optimizer=opt, loss='mse')

#Train to reconstruct self 
model.fit(input, input, epochs=300, verbose=1)

#Connect the encoder LSTM as the output layer
model_enc = Model(inputs=model.inputs, outputs=model.layers[0].output)

#Get the feature vectors for the input sequences
yhat = model_enc.predict(input)
print(yhat.shape)
print(yhat)      



###########################################################################################################
#                              Transformer (BERT) based feature representations                           #
###########################################################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#Load text df and combine all words into one text sequence in one column 
dir = os.getcwd()
path = os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv')
raw_resp = pd.read_csv(path, index_col=False)

def generateList(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
                      for sub in [lst1, lst2]]

text_df = pd.DataFrame(columns=np.arange(0,1))
columns = text_df.columns
for i in range(0, raw_resp.shape[0]):
    print(i)
    l1 = [str(col) for col in raw_resp.loc[:,:'loan'].columns]
    l2 = [str(resp) for resp in raw_resp.loc[:,:'loan'].iloc[i,:]]
    text = generateList(l2, l1)
    #text = [' '.join(text[ii:ii+2]) for ii in np.arange(0,586,2)]
    text = [" ".join(text)]
    zipped = zip(columns, text)
    output_dict = dict(zipped)
    text_df = text_df.append(output_dict, ignore_index=True)

text_df = text_df.rename(columns={0: "text"})

#Add group labels
raw_resp = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv'), index_col=False)
text_df['dep_group'] = raw_resp.dep_group

#Label encoding 
LE = LabelEncoder()
text_df['label'] = LE.fit_transform(text_df['dep_group'])
text_df.head()

#Split into train-test (random seed to fit with split in classification pipeline)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_df.text, text_df.label,
                                                    stratify = text_df.label,
                                                    test_size=0.3,
                                                    random_state=42)

#Generate text embeddings 
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

#Tokenize 
tokenized_train = tokenizer(X_train.values.tolist(), padding = True, truncation = True, return_tensors="pt")
tokenized_test = tokenizer(X_test.values.tolist(), padding = True, truncation = True, return_tensors="pt")

print(tokenized_train.keys())

#Move on device (GPU)
tokenized_train = {k:torch.tensor(v).to(device) for k,v in tokenized_train.items()}
tokenized_test = {k:torch.tensor(v).to(device) for k,v in tokenized_test.items()}

#Get the text ([CLS]) hiddden states - run model 
with torch.no_grad():
  hidden_train = model(**tokenized_train) #dim : [batch_size(nr_sentences), tokens, emb_dim]
  hidden_test = model(**tokenized_test)

#get only the [CLS] hidden states
cls_train = hidden_train.last_hidden_state[:,0,:]
cls_test = hidden_test.last_hidden_state[:,0,:]

cls_train_df = pd.DataFrame(cls_train.numpy())
cls_test_df = pd.DataFrame(cls_test.numpy())

cls_train_df['ID'] = raw_resp.ID
cls_train_df['gender'] = raw_resp.gender
cls_train_df['age'] = raw_resp.age
cls_train_df['dep_group'] = raw_resp.dep_group
cls_train_df['PHQ9_sum'] = raw_resp.PHQ9_sum

cls_test_df['ID'] = raw_resp.ID
cls_test_df['gender'] = raw_resp.gender
cls_test_df['age'] = raw_resp.age
cls_test_df['dep_group'] = raw_resp.dep_group
cls_test_df['PHQ9_sum'] = raw_resp.PHQ9_sum

cls_train_df.to_csv(os.path.join(dir,'classification/dep_classification/features_dfs/BERT_features_responses_train.csv'), index=False)
cls_test_df.to_csv(os.path.join(dir,'classification/dep_classification/features_dfs/BERT_features_responses_test.csv'), index=False)

###########################################################################################################
#                                    Distribution of PHQ9 sum scores by dep group                         #
###########################################################################################################

import matplotlib.pyplot as plt
raw_resp = df_behav
raw_resp.columns.values[306] = 'dep_group'

sum0 = raw_resp.loc[df['dep_group'] == 0, 'PHQ9_sum']
sum1 = raw_resp.loc[df['dep_group'] == 1, 'PHQ9_sum']

plt.clf()
f1 = plt.figure()
plt.hist(sum0, alpha=0.5, label='control', color='blue')
plt.hist(sum1, alpha=0.5, label='patient', color='orange')
plt.title('PHQ9 Sum Score Distribution by Group')
plt.xlabel('PHQ9 Sum Score')
plt.ylabel('Count')
plt.legend(title='Group')
plt.savefig('/users/line/dct_fMRI_analysis/classification/dep_classification/output/phq9_dist.png')


#N in dep group with PHQ9 sum above threshold (10)
x1 = raw_resp.loc[(raw_resp['dep_group']==1) & (raw_resp['PHQ9_sum']>9)] #n=27 
# N in dep group with PHQ9 sum below threshold (10)
x2 = raw_resp.loc[(raw_resp['dep_group']==1) & (raw_resp['PHQ9_sum']<10)] #n=3
# N in cont group wtih PHQ9 sum above threshold (10)
x3 = raw_resp.loc[(raw_resp['dep_group']==0) & (raw_resp['PHQ9_sum']>9)] #n=7
# N in cont gropu with PHQ9 sum below thresohld 
x4 = raw_resp.loc[(raw_resp['dep_group']==0) & (raw_resp['PHQ9_sum']<10)] #35 

#Plot confusion mat 
import seaborn as sn
array = [[x1.shape[0], x2.shape[0]],[x3.shape[0],x4.shape[0]]]
df_cm = pd.DataFrame(array, index = ['patient','control'],
                  columns = ['phq>10','phq<10'])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('/users/line/dct_fMRI_analysis/classification/dep_classification/output/group_confMat.png')
