###########################################################################################################
#                                            Text features from LSTM                                      #
###########################################################################################################
#Treat all trials (stimulus and response) as one long sentence ("this cat that house that party this day...")
#Apply LSTM to extract features from the text - use as input for classification
#Treated as natural language processing problem  

import os 
import numpy as np 
import pandas as pd 

#-------------------------- Prep data ------------------------------#
#Create text representations for each subject 
dir = os.getcwd()
path = os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv')
raw_resp = pd.read_csv(path, index_col=False)

def generateList(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
                      for sub in [lst1, lst2]]

text_df = pd.DataFrame(columns=np.arange(0,586))
columns = text_df.columns
for i in range(0, raw_resp.shape[0]):
    print(i)
    l1 = [str(col) for col in raw_resp.loc[:,:'loan'].columns]
    l2 = [str(resp) for resp in raw_resp.loc[:,:'loan'].iloc[i,:]]
    text = generateList(l2, l1)
    #text = [' '.join(text[ii:ii+2]) for ii in np.arange(0,586,2)]
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
from tensorflow.keras.layers import LSTM, Dropout, Dense, RepeatVector, TimeDistributed, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow import keras

#Define each timestep to include stim + response 
text_df = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/text_seq_responses.csv'), index_col=False)
text_df.isna().sum().sum() #39 nan 

#Fill nan with random 
import random
while(text_df.isna().sum().sum()!=0):
    text_df.fillna(random.choice(['this','that']),inplace=True,limit=1)
text_df.isna().sum().sum() #0 nan 

seq_list = []
for i in range(0, text_df.shape[0]):
   text = [str(w) for w in text_df.iloc[i,:]]
   #text = ' '.join(text)
   text = [' '.join(text[ii:ii+2]) for ii in np.arange(0,586,2)]
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
#from sklearn.preprocessing import StandardScaler, RobustScaler
#scaler = StandardScaler() # Gives range (-0.9, 7.9)
#scaler = RobustScaler() # Gives range (-1, 596)
#input_scaled = scaler.fit_transform(seq_encoded_df)

#3D input shape: samples (n sequences), timesteps (n words), features (n feature types - we have two: stim + response)
#One seq = all trials per sub, one trial = stim + response 
input = seq_encoded_df.to_numpy().reshape(73, 293, 1)

#Get labels and encode 
from sklearn.preprocessing import LabelEncoder
raw_resp = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv'), index_col=False)
labels = raw_resp.dep_group

LE = LabelEncoder()
labels = LE.fit_transform(labels)

#-------------------------- Create train-test splits ------------------------------#

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)

#Compute class weights to deal with imbalance 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=labels)
class_weights = dict(zip([0,1], class_weights))

#-------------------------- Define model ------------------------------#

n_timesteps = X_train.shape[1] #Should consider each trial to include word + response 
n_features = X_train.shape[2] #Word and response
vocab_size = X_train.max()+1 #Vocab indices are in range(1,586), +1 because the embedding look-up is zero-indexed

regularizers_list = [regularizers.L1L2(0.1,0.0), regularizers.L1L2(0.0,0.1), regularizers.L1L2(0.1,0.1)]

train_loss = []
train_acc = []
train_f1 = []
test_loss = []
test_acc = []
test_f1 = []

l1_reg = np.arange(0,0.11,0.01)
l2_reg = np.arange(0,0.11,0.01)

#Hyperparameter tuning (l1l2 reg)
def build_model(hp):
    #Define model and fit 
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=n_timesteps))
    model.add(LSTM(100, input_shape=(n_timesteps, n_features), bias_regularizer=regularizers.L1L2(hp.Float("l1", min_value=1e-2, max_value=1.5, sampling="log"),hp.Float("l1", min_value=1e-2, max_value=1.5, sampling="log"))))
    #model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dense(n_features, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train to reconstruct self 
    #model.fit(X_train, y_train, epochs=300, verbose=0, class_weight=class_weights)
    #model.fit(X_train, y_train, epochs=300, verbose=0)
    return model 
#Check model build works 
build_model(keras_tuner.HyperParameters())

#Use random search tuner 
import keras_tuner
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory=os.path.join(dir, "classification"),
    project_name="dep_classification",
)



#------------------------Evaluate on train and test set-------------------------#
train_eval = model.evaluate(X_train, y_train)
test_eval = model.evaluate(X_test, y_test)

train_pred = np.transpose(model.predict(X_train))[0]
test_pred = np.transpose(model.predict(X_test))[0]

yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
yhat_test = list(map(lambda x: 1 if x>0.5 else 0, test_pred))

f1_train = f1_score(y_train, yhat_train, average='macro') #Macro needed with imbalanced classes 
f1_test = f1_score(y_test, yhat_test, average='macro')

embed_weights = pd.DataFrame(model.layers[0].get_weights()[0]) #shape (920,100) becuase our vocab size=100 
embed_weights = embed_weights.iloc[296:,:] #Get the weights corresponding to our vocab, shape(624, 100)

#train_loss.append(train_eval[0])
#train_acc.append(train_eval[1])
#train_f1.append(f1_train)
#test_loss.append(test_eval[0])
#test_acc.append(test_eval[1])
#test_f1.append(f1_test)




#---------------------- Confusion matrix ----------------- #
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt 
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          verbose=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose:
            print("Normalized confusion matrix")
    else:
        if verbose:
            print('Confusion matrix, without normalization')
    if verbose:
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.rcParams["figure.figsize"] = [4, 4]
    plt.savefig('CM_MobileNet_Best.pdf', bbox_inches = 'tight')
    plt.show() 

yhat = test_pred.argmax(axis=-1)  
true = y_test
cm=confusion_matrix(yhat, true)
classes=["Control","Patient"]
plot_confusion_matrix(cm_test, classes, normalize=True, cmap=plt.cm.Blues)

