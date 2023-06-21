import os 
import numpy as np 
import pandas as pd 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, RepeatVector, TimeDistributed, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import optimizers
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

###########################################################################################################
#                                                 Model 1                                                 #
# LSTM, paired (stim-resp) encodings, L1L2 reg, clipnorm=1, hyperparam tuned, fitted with eval data       # 
###########################################################################################################

#Load and split data (identically to when training the model)
#-------------------------- Prep data ------------------------------#
#Add labels and reshape 
dir = os.getcwd()
seq_encoded_df = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/text_seq_encoded_responses.csv'), index_col=False)
input = seq_encoded_df.to_numpy().reshape(73, 293, 1)

#Get labels and encode 
raw_resp = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv'), index_col=False)
labels = raw_resp.dep_group

LE = LabelEncoder()
labels = LE.fit_transform(labels)

train_len = np.arange(0,73)
X_train_idx, X_test_idx, y_train, y_test = train_test_split(train_len, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
X_train = input[X_train_idx]
X_test = input[X_test_idx]

#------------------- Load trained model ---------------------#
rec_model1 = load_model(os.path.join(dir, 'classification/dep_classification/trained_models/models/model1.h5'))
#Error in loading saved optimizer - initialized optimizer, not sure if that's a problem for prediction (don't think so)

rec_model1_random = load_model(os.path.join(dir, 'classification/dep_classification/trained_models/models/model1_random.h5'))

#---------------- Evaluate on test set --------------#
train_eval = rec_model1.evaluate(X_train, y_train)
test_eval = rec_model1.evaluate(X_test, y_test)
test_loss = test_eval[0]

train_pred = np.transpose(rec_model1.predict(X_train))[0]
test_pred = np.transpose(rec_model1.predict(X_test))[0]

yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
yhat_test = list(map(lambda x: 1 if x>0.5 else 0, test_pred))

f1_train = f1_score(y_train, yhat_train, average='macro') #Macro needed with imbalanced classes 
f1_test = f1_score(y_test, yhat_test, average='macro') #0.78

test_loss_random = rec_model1_random.evaluate(X_test, y_test)[0]
test_pred_random = np.transpose(rec_model1_random.predict(X_test))[0]
yhat_test_random = list(map(lambda x: 1 if x>0.5 else 0, test_pred_random))
f1_test_random = f1_score(y_test, yhat_test_random, average='macro') #0.47

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
    #plt.savefig('CM_MobileNet_Best.pdf', bbox_inches = 'tight')
    #plt.show() 


cm=confusion_matrix(yhat_test, y_test)
classes=["Control","Patient"]
plot_confusion_matrix(cm, classes, normalize=True, cmap=plt.cm.Blues)
plt.savefig(os.path.join(dir,'classification/dep_classification/models_performance/model1_confMat.png'))

cm=confusion_matrix(yhat_test_random, y_test)
classes=["Control","Patient"]
plt.figure()
plot_confusion_matrix(cm, classes, normalize=True, cmap=plt.cm.Blues)
plt.savefig(os.path.join(dir,'classification/dep_classification/models_performance/model1_confMat_random.png'))


#-------------- Plot classificaiton probabilities --------------#
n_classifiers = 1
plt.figure(figsize=(3 * 2, n_classifiers * 2))
plt.subplots_adjust(bottom=0.2, top=0.95)

xx = np.linspace(3, 9, 100)
yy = np.linspace(1, 5, 100).T
xx, yy = np.meshgrid(xx, yy)
Xfull = np.c_[xx.ravel(), yy.ravel()]

classifiers = ({'model':rec_model1, 'random_model':rec_model1_random})
preds = [test_pred, test_pred_random]


#--------------- Plot CV train/validation accuracy and loss ----------------# 
#Load performance df 
mod1_perf = pd.read_csv(os.path.join(dir, 'classification/dep_classification/models_performance/model1_performance.csv'))

from matplotlib.pylab import plt
from numpy import arange
 
# Load the training and validation loss dictionaries
 
# Retrieve each dictionary's values
train_loss = mod1_perf.train_loss
val_loss = mod1_perf.val_loss
train_acc = mod1_perf.train_acc
val_acc = mod1_perf.val_acc
 
# Generate a sequence of integers to represent the epoch numbers
epochs = range(0, len(train_loss))
 
#---- Acc 
# Plot and label the training and validation loss values
plt.figure()
plt.plot(epochs, train_acc, label='Training Accuracy', color='blue')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='green')
plt.axhline(y=np.mean(train_acc), color='blue', linestyle='--', label='train mean')
plt.axhline(y=np.mean(val_acc), color='green', linestyle='--', label='validation mean')
plt.axhline(y=f1_test, color='orange',linestyle='--', label='test f1 accuracy')
plt.axhline(y=f1_test_random, color='brown',linestyle='--', label='shuffled f1 accuracy')
# Add in a title and axes labels
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
 
# Set the tick locations
plt.xticks(arange(0, len(train_loss), 2))
 
# Display the plot
plt.legend(loc='best')
#plt.show()
plt.savefig(os.path.join(dir, 'classification/dep_classification/models_performance/model1_cv_acc.png'))

#----Loss
plt.figure()
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', color='green')
plt.axhline(y=np.mean(train_loss), color='blue', linestyle='--', label='train loss')
plt.axhline(y=np.mean(val_loss), color='green', linestyle='--', label='validation loss')
 
# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
# Set the tick locations
plt.xticks(arange(0, len(train_loss), 2))
 
# Display the plot
plt.legend(loc='best')
#plt.show()
plt.savefig(os.path.join(dir, 'classification/dep_classification/models_performance/model1_cv_loss.png'))


#------------------ Plot predicted class probability against choice on each word ------------# 
df_plot = raw_resp.iloc[X_test_idx,:]
df_plot['pred_group'] = yhat_test

words = []
prop_this_dep = []
prop_this_cont = []
for i in range(0, 293):
    print(i)
    word = df_plot.columns[i]
    dep_sub = df_plot[df_plot['pred_group']==1]
    dep_prop = dep_sub.iloc[:,i].value_counts()
    if 'this' in dep_prop:
        dep_prop_this = (dep_prop['this'])/len(dep_sub)
    else: 
         dep_prop_this = 0
    cont_sub = df_plot[df_plot['pred_group']==0]
    cont_prop = cont_sub.iloc[:,i].value_counts()
    if 'this' in cont_prop:
        cont_prop_this = (cont_prop['this'])/len(cont_sub)
    else: 
        cont_prop_this = 0
    words.append(word)
    prop_this_dep.append(dep_prop_this)
    prop_this_cont.append(cont_prop_this)


df = pd.DataFrame({'words':words,'prop_dep':prop_this_dep,'prop_cont':prop_this_cont})
df['diff'] = df.prop_cont-df.prop_dep
df['abs_diff'] = abs(df['diff'])

df_max_diff = df.nlargest(100,'abs_diff')
df_max_diff = df_max_diff.sort_values(by=['diff'], ascending=True)

from colour import Color
red = Color("darkorange")
colors = list(red.range_to(Color("green"),100))
colors = [color.rgb for color in colors]

import matplotlib.pyplot as plt
plt.figure(figsize=(10,15))
plt.barh(df_max_diff.words,df_max_diff['diff'],color=colors)
#plt.yticks(np.arange(len(mean_df)+1),mean_df.feature_name)
plt.title('Response proportions by predicted labels',size=16)
plt.ylim((-1,len(df_max_diff)+1))
#plt.plot([baseline_mae,baseline_mae],[-1,len(mean_df)+1], '--', color='orange',
#            label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
plt.xlabel('Difference in proportion proximal choices between groups',size=14)
plt.ylabel('Feature',size=14)
plt.legend()
#plt.show()
plt.savefig(os.path.join(dir,f'classification/dep_classification/models_performance/responses_by_model_pred.png'))


#------------------ Plot demonstrative choices against actual class ------------# 
df_plot = raw_resp

words = []
prop_this_dep = []
prop_this_cont = []
for i in range(0, 293):
    print(i)
    word = df_plot.columns[i]
    dep_sub = df_plot[df_plot['dep_group']==1]
    dep_prop = dep_sub.iloc[:,i].value_counts()
    if 'this' in dep_prop:
        dep_prop_this = (dep_prop['this'])/len(dep_sub)
    else: 
         dep_prop_this = 0
    cont_sub = df_plot[df_plot['dep_group']==0]
    cont_prop = cont_sub.iloc[:,i].value_counts()
    if 'this' in cont_prop:
        cont_prop_this = (cont_prop['this'])/len(cont_sub)
    else: 
        cont_prop_this = 0
    words.append(word)
    prop_this_dep.append(dep_prop_this)
    prop_this_cont.append(cont_prop_this)


df = pd.DataFrame({'words':words,'prop_dep':prop_this_dep,'prop_cont':prop_this_cont})
df['diff'] = df.prop_cont-df.prop_dep
df['abs_diff'] = abs(df['diff'])

df_max_diff = df.nlargest(100,'abs_diff')
df_max_diff = df_max_diff.sort_values(by=['diff'], ascending=True)

from colour import Color
red = Color("darkorange")
colors = list(red.range_to(Color("green"),100))
colors = [color.rgb for color in colors]

import matplotlib.pyplot as plt
plt.figure(figsize=(10,15))
plt.barh(df_max_diff.words,df_max_diff['diff'],color=colors)
#plt.yticks(np.arange(len(mean_df)+1),mean_df.feature_name)
plt.title('Response proportions by predicted labels',size=16)
plt.ylim((-1,len(df_max_diff)+1))
#plt.plot([baseline_mae,baseline_mae],[-1,len(mean_df)+1], '--', color='orange',
#            label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
plt.xlabel('Difference in proportion proximal choices between groups',size=14)
plt.ylabel('Feature',size=14)
plt.legend()
#plt.show()
plt.savefig(os.path.join(dir,f'classification/dep_classification/models_performance/responses_by_true_group.png'))




#---------------------- Load feature importances and plot --------------------#
mod1_feat_imp = pd.read_csv(os.path.join(dir,'classification/dep_classification/models_performance/model1_feature_importance_trainTrue.csv'), index_col=False)

features = mod1_feat_imp['feature'].unique()
mean_imp = mod1_feat_imp.groupby('feature').mean()

mean_df = pd.DataFrame({'feature':features, 'mean_mae':mean_imp.mae})

#Get trial names 
text_df = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/text_seq_responses.csv'), index_col=False)
index = np.arange(1,586,2)
feature_names = text_df.iloc[1,:].to_numpy()[index]
feature_names = np.append(feature_names, 'BASELINE')

mean_df['feature_name'] = feature_names
baseline_mae = mean_df['mean_mae'].loc[mean_df['feature']=='BASELINE'].item()
mean_df.drop('BASELINE')
mean_df = mean_df.sort_values(by=['mean_mae'], ascending=True)

#Plot all features 
plt.figure(figsize=(10,20))
plt.barh(mean_df.feature_name,mean_df.mean_mae)
#plt.yticks(np.arange(len(mean_df)+1),mean_df.feature_name)
plt.title('LSTM Feature Importance',size=16)
plt.ylim((-1,len(mean_df)+1))
plt.plot([baseline_mae,baseline_mae],[-1,len(mean_df)+1], '--', color='orange',
            label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
plt.xlabel('Mean OOF MAE with feature permuted',size=14)
plt.ylabel('Feature',size=14)
plt.legend()
#plt.show()
plt.savefig(os.path.join(dir,f'classification/dep_classification/models_performance/model1_feature_importance_trainTrue_names.png'))

#Plot max 50 features (largest loss in model performance)              
df_max = mean_df.nlargest(50, 'mean_mae')
df_max = df_max.sort_values(by=['mean_mae'], ascending=True)
plt.figure(figsize=(10,20))
plt.barh(df_max.feature_name,df_max.mean_mae)
#plt.yticks(np.arange(len(mean_df)+1),mean_df.feature_name)
plt.title('LSTM Feature Importance',size=16)
plt.ylim((-1,len(df_max)+1))
plt.plot([baseline_mae,baseline_mae],[-1,len(df_max)+1], '--', color='orange',
            label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
plt.xlabel('Mean OOF MAE with feature permuted',size=14)
plt.ylabel('Feature',size=14)
plt.legend()
#plt.show()
plt.savefig(os.path.join(dir,f'classification/dep_classification/models_performance/model1_feature_importance_trainTrue_names_max50.png'))


#----------- Get feature importance with SHAP -----------# 
import shap

#explainer = shap.DeepExplainer(rec_model1, X_train)
#shap_values = explainer.shap_values(X_test)

explainer = shap.Explainer(rec_model1) 
shap_values = explainer(X_test)

# visualize the first prediction's explanation for the POSITIVE output class
shap.plots.text(shap_values[0, :, "POSITIVE"])