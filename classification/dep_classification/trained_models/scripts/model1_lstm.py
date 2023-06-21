###########################################################################################################
#                                            Text features from LSTM                                      #
###########################################################################################################
#Treat all trials (stimulus and response) as one long sentence ("this cat that house that party this day...")
#Apply LSTM to extract features from the text - use as input for classification
#Treated as natural language processing problem  

import os 
import numpy as np 
import pandas as pd 
import tensorflow
from tensorflow.keras.models import Model
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

#-------------------------- Prep data ------------------------------#
#Ordinal encoder

#------------ Add labels and reshape ---------------# 
seq_encoded_df = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/text_seq_encoded_responses.csv'), index_col=False)

#3D input shape: samples (n sequences), timesteps (n words), features (n feature types - we have two: stim + response)
#One seq = all trials per sub, one trial = stim + response 
input = seq_encoded_df.to_numpy().reshape(73, 293, 1)

#Get labels and encode 
raw_resp = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv'), index_col=False)
labels = raw_resp.dep_group

LE = LabelEncoder()
labels = LE.fit_transform(labels)

#-------------------------- Create train-test splits ------------------------------#

X_train, X_test, y_train, y_test = train_test_split(input, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)

#Compute class weights to deal with imbalance 
class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=labels)
class_weights = dict(zip([0,1], class_weights))

#-------------------------- Define model and tune hyperparam ------------------------------#

n_timesteps = X_train.shape[1] #Should consider each trial to include word + response 
n_features = X_train.shape[2] #Word and response
vocab_size = X_train.max()+1 #Vocab indices are in range(1,586), +1 because the embedding look-up is zero-indexed

#Hyperparameter tuning (l1l2 reg)
import keras_tuner
def build_model(hp):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=n_timesteps))
    model.add(LSTM(100, input_shape=(n_timesteps, n_features), bias_regularizer=regularizers.L1L2(hp.Float("l1", min_value=1e-2, max_value=1.5, sampling="log"), hp.Float("l2", min_value=1e-2, max_value=1.5, sampling="log"))))
    model.add(Dense(n_features, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.legacy.Adam(clipvalue=1.), metrics=['accuracy'])
    return model 

#Check model build works 
build_model(keras_tuner.HyperParameters())

X_train_1 = X_train[:36,:,:] #Using 36 of 51 train datapoints to train (for tuning)
X_train_2 = X_train[36:,:,:] #Using 15 of 51 train datapoints to validate (for tuning)
y_train_1 = y_train[:36]
y_train_2 = y_train[36:]

#Use random search tuner 
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    seed=42, 
    executions_per_trial=2,
    overwrite=False,
    directory=os.path.join(dir), 
    project_name='lstm_hyperparam_tuning'
)
tuner.search_space_summary()

#Start search (does not respect class_weight arguments)
tuner.search(X_train_1, y_train_1, epochs=300, validation_data=(X_train_2, y_train_2))

#Extract and build best model 
models = tuner.get_best_models(num_models=1)
best_model = models[0]

# Build the model
best_model.build()
best_model.summary() 

#Get hyperparams of best model 
best_hp = tuner.get_best_hyperparameters()
best_l1 = best_hp[0].values['l1'] #0.86
best_l2 = best_hp[0].values['l2'] #0.06

#-------------------- Cross validation on best hyperparams -----------------#
#                           (select best model)                             #
#Commented this out, because I read somewhere that you should not do cross-validation on deep learning models, 
#as they already need to be run several times for training. Rather, input a validation_data= argument to the 
#.fit() function that trains the model. 

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []
TRAIN_ACCURACY = []
TRAIN_LOSS = []

fold_var = 1
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
epochs = 100
Y=y_train

for train_index, val_index in cv.split(np.zeros(len(Y)),Y):
    print(f'----------------Running fold {fold_var}---------------')
    training_data = X_train[train_index]
    validation_data = X_train[val_index]

    training_y = y_train[train_index]
    validation_y = y_train[val_index]

    # CREATE NEW MODEL
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=False,
        directory=os.path.join(dir), 
        project_name='lstm_hyperparam_tuning'
    )

    #Extract and build best model 
    models = tuner.get_best_models(num_models=1)
    model = models[0]
    model.build() #builds and compiles 

    # CREATE CALLBACKS
    #checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(fold_var), 
    #                        monitor='val_accuracy', verbose=1, 
    #                        save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]
    # There can be other callbacks, but just showing one because it involves the model name
    # This saves the best model
    # FIT THE MODEL
    history = model.fit(training_data, training_y,
                epochs=epochs,
                #callbacks=callbacks_list,
                validation_data=(validation_data, validation_y))

    # LOAD BEST MODEL to evaluate the performance of the model
    #model.load_weights(os.path.join("saved_models/model_"+str(fold_var)+".h5"))
    results_train = model.evaluate(training_data, training_y)
    results_train = dict(zip(model.metrics_names, results_train))
    results_val = model.evaluate(validation_data, validation_y)
    results_val = dict(zip(model.metrics_names,results_val))

    TRAIN_ACCURACY.append(results_train['accuracy'])
    TRAIN_LOSS.append(results_train['loss'])
    VALIDATION_ACCURACY.append(results_val['accuracy'])
    VALIDATION_LOSS.append(results_val['loss'])

    #tensorflow.keras.backend.clear_session()
    fold_var += 1

model1_performance = pd.DataFrame({'fold':range(0,25),'train_acc':TRAIN_ACCURACY,'train_loss':TRAIN_LOSS,'val_acc':VALIDATION_ACCURACY, 'val_loss':VALIDATION_LOSS})
model1_performance.to_csv(os.path.join(dir, 'classification/dep_classification/models_performance/model1_performance.csv'),index=False)

#------------------------ Fit and evaluate best model -------------------------#
#Fit
best_model.fit(X_train_1, y_train_1, epochs=300, verbose=1, class_weight=class_weights, validation_data=(X_train_2, y_train_2))

train_eval = best_model.evaluate(X_train_1, y_train_1)
valid_eval = best_model.evaluate(X_train_2, y_train_2)

train_pred = np.transpose(best_model.predict(X_train_1))[0]
valid_pred = np.transpose(best_model.predict(X_train_2))[0]

yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
yhat_valid = list(map(lambda x: 1 if x>0.5 else 0, valid_pred))

f1_train = f1_score(y_train_1, yhat_train, average='macro') #Macro needed with imbalanced classes 
f1_test = f1_score(y_train_2, yhat_valid, average='macro')
#f1_train = 1.0 
#f1_test = 0.78

#------------------- Save model ----------------------#
best_model.save(os.path.join(dir, 'classification/dep_classification/trained_models/models/model1.h5'))


#--------------- Compute feature importance --------------# 
import tqdm

# detect and init the TPU
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# GET GPU STRATEGY
from tensorflow import keras
import matplotlib as plt 
gpu_strategy = tensorflow.distribute.get_strategy()

X = X_train
y = y_train

EPOCH = 300
NUM_FOLDS = 5
COLS = X.shape[1]
TRAIN_MODEL=True 
INFER_TEST=False 
COMPUTE_LSTM_IMPORTANCE=True
ONE_FOLD_ONLY=False

file_name = 'model1_feature_importance_trainTrue'

with gpu_strategy.scope():
    kf = cv
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        #keras.clear_session()
        
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = X[train_idx], X[test_idx]
        y_train, y_valid = y[train_idx], y[test_idx]
        

        checkpoint_filepath = f"folds{fold}.hdf5"
        if TRAIN_MODEL:
            tuner = keras_tuner.RandomSearch(
                hypermodel=build_model,
                objective="val_accuracy",
                max_trials=3,
                executions_per_trial=2,
                overwrite=False,
                directory=os.path.join(dir), 
                project_name='lstm_hyperparam_tuning'
            )

            #Extract and build best model 
            models = tuner.get_best_models(num_models=1)
            model = models[0]
            model.build() #builds and compiles 

            model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCH)
            
        else:
            model = keras.models.load_model(os.path.join(dir, 'classification/dep_classification/trained_models/models/model1.h5'))

        if INFER_TEST:
            print(' Predicting test data...')
            test_preds.append(model.predict(X_test,verbose=0).squeeze().reshape(-1, 1).squeeze())
                    
        if COMPUTE_LSTM_IMPORTANCE:
            results = []
            print(' Computing LSTM feature importance...')
            
            # COMPUTE BASELINE (NO SHUFFLE)
            oof_preds = model.predict(X_valid, verbose=0).squeeze() 
            baseline_mae = np.mean(np.abs( oof_preds-y_valid ))
            results.append({'feature':'BASELINE','mae':baseline_mae})           

            for k in tqdm.tqdm(range(0,COLS)):
                
                # SHUFFLE FEATURE K
                save_col = X_valid[:,k,:].copy()
                np.random.shuffle(X_valid[:,k,:])
                        
                # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
                oof_preds = model.predict(X_valid, verbose=0).squeeze() 
                mae = np.mean(np.abs( oof_preds-y_valid ))
                results.append({'feature':k,'mae':mae})
                X_valid[:,k,:] = save_col
         
            # DISPLAY LSTM FEATURE IMPORTANCE
            df = pd.DataFrame(results)
            df = df.sort_values('mae')
            plt.figure(figsize=(10,20))
            plt.barh(np.arange(COLS+1),df.mae)
            plt.yticks(np.arange(COLS+1),df.feature.values)
            plt.title('LSTM Feature Importance',size=16)
            plt.ylim((-1,COLS+1))
            plt.plot([baseline_mae,baseline_mae],[-1,COLS+1], '--', color='orange',
                     label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
            plt.xlabel(f'Fold {fold+1} OOF MAE with feature permuted',size=14)
            plt.ylabel('Feature',size=14)
            plt.legend()
            #plt.show()
            plt.savefig(os.path.join(dir,f'classification/dep_classification/models_performance/{file_name}.png'))
                               
            # SAVE LSTM FEATURE IMPORTANCE
            df = df.sort_values('mae',ascending=False)
            df['fold'] = fold
            if fold==0: 
                df_out = df
            else: 
                df_out = pd.concat((df_out, df))
                               
        # ONLY DO ONE FOLD
        if ONE_FOLD_ONLY: break
        df_out.to_csv(os.path.join(dir,f'classification/dep_classification/models_performance/{file_name}.csv'), index=False)


