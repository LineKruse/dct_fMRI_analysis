import os 
import pandas as pd 
import numpy as np 
import nilearn 
import sklearn 
import matplotlib.pyplot as plt

from sklearn.linear_model import(
 SGDRegressor,
 SGDClassifier,
 LogisticRegression,
 LogisticRegressionCV,
 LinearRegression,
 LinearDiscriminantAnalaysis, 
)

from sklearn.ensemble import(
    AdaBoostRegressor,
    BaggingRegressor,
    AdaBoostClassifier,
    BaggingClassifier,
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
    permutation_test_score,
    RepeatedKFold
)
from sklearn.inspection import permutation_importance

from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    roc_auc_score,
    f1_score
)
from scipy.stats import sem
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

###########################################################################################################
#                                          Prepare data                                                   #
###########################################################################################################

raw_resp = pd.read_csv('/users/line/dct_fMRI_analysis/classification/dep_classification/features_dfs/raw_responses.csv', index_col=False)





###########################################################################################################
#                                              Classification setup                                    #
###########################################################################################################
#--------------------------- Select input features ----------------------------#
features_list = ['raw_resp','PCA_resp']

input_feat = features_list[0]

if input_feat=='raw_resp':
    df = raw_resp
elif input_feat=='PCA_resp':
    df = PCA_resp


#--------------------------- Prepare output dataframe ----------------------------#
dataframe = pd.DataFrame(columns = ['target',
                                    'features',
                                    'encoder',
                                    'model',
                                    'base_cv_score_train',
                                    'base_cv_SD_train',
                                    'base_cv_SE_train',
                                    'base_score_train', 
                                    'base_auc_train', 
                                    'base_score_test', 
                                    'base_auc_test', 
                                    'base_perm_acc_train', 
                                    'base_perm_score_train', 
                                    'base_perm_pval_train', 
                                    'base_perm_acc_test', 
                                    'base_perm_score_test', 
                                    'base_perm_pval_test', 
                                    'feature_names', 
                                    'fs_score',
                                    'fs_SD', 
                                    'fs_SE', 
                                    'grid_best_params', 
                                    'grid_best_score', 
                                    'grid_best_SD', 
                                    ])
columns = list(dataframe)


#--------------------------------- Define basic params/funcs -------------------------------#
#Define category encoders, imputing nulls through the encoding for categorical columns
ord_encoder = ce.OrdinalEncoder(handle_missing='value') #when no ordinal mapping is given, assigns values to cats at random
bin_encoder = ce.BinaryEncoder(handle_missing='value') #binary version of one-hot encoding 
targ_encoder = ce.TargetEncoder(handle_missing='value') #inlcude information about relationship to target 
encoder_list = [ord_encoder, bin_encoder, targ_encoder]
encoder_list_names = ['ordinal','binary','target']

target_type = 'predefined_group'

cv1 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

n_permutations=100

models = []
models.append(('LR', LogisticRegression(class_weight = 'balanced', solver = 'saga', random_state=42, penalty='none')))
models.append(('LDA', LinearDiscriminantAnalysis(solver='svd', random_state=42)))
models.append(('KNN', KNeighborsClassifier(random_state=42)))
models.append(('RF', RandomForestClassifier(class_weight='balanced',random_state=42)))
models.append(('NB', GaussianNB())) #Doesn't do anything with random initiation, takes no random seed input 
models.append(('SVM', SVC(class_weight='balanced', random_state=42)))
models.append(('XGBoost',xgb.XGBRegressor(objective="binary:logistic",seed=42)))

#Function: Permutation test on accuracy 
def permutation_test(features, target, model): 
    score, permutation_scores, pvalue = permutation_test_score(
                                        model,
                                        features,
                                        target,
                                        scoring = 'accuracy',
                                        cv = cv1,
                                        n_permutations = n_permutations,
                                        n_jobs = -1,
                                        random_state=42,                                                                     
                                        verbose = 1)
    return(score, permutation_scores, pvalue)

#Function: permutation test on feature importances
def permutation_importance_test(features, target, model):
    r = permutation_importance(model,
                               features,
                               target,
                               n_repeats=100,
                               scoring = 'accuracy',
                               n_jobs=-1,
                               random_state=42)
    return(r.importances_mean, r.importances_std, r.importances)

############# 
# How to make permutation for xgboost models?
#############

#--------------------------- Create train-test splits ----------------------------#
if target_type == 'predefined_group':
    y = df['dep_group'] 
if target_type == 'phq9_group':
    df['phq9_group'] = [1 if df.loc[df['PHQ9_sum']>9] else 0]
y=y.astype('int')
X = df.loc[:,'day':]
X = df.loc[:,:'loan']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify = y,
                                                    test_size=0.3,
                                                    random_state=42)

###########################################################################################################
#                                              Classification pipeline                                    #
###########################################################################################################
#Loop through encoders, fit and evaluate models 

for i in range(0, len(encoder_list)):
    enc = encoder_list[i]
    encoder = encoder_list_names[i] 
    print('------- Running encoder:',encoder,'-------------')

    for model_name, model in models:
        print('----Fitting model:',model_name,'------')

        #--------------------------------- Basline models -------------------------------#
        #---------------------------(no PCA - raw word vectors)-------------------------#
        
        #Transform categorical features by encoder
        if input_feat == 'raw_resp':
            if encoder=='target':
                X_train_tf = enc.fit_transform(X_train, y_train)
                X_test_tf = enc.fit_transform(X_test, y_test)
            else:
                X_train_tf = enc.fit_transform(X_train)
                X_test_tf = enc.fit_transform(X_test)
        
        #Evaluate model by cv
        from xgboost import cv
        if model_name == 'XGBoost':
            params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,'max_depth': 5, 'alpha': 10}
            data_dmatrix = xgb.DMatrix(data=X_train_tf,label=y_train)
            xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=5,num_boost_round=50, early_stopping_rounds=10, metrics="error", as_pandas=True, seed=42)

            #Accuracy is 1-error 
            base_cv_score_train = 1-xgb_cv['test-error-mean'][0]
            base_cv_SD_train = 1-xgb_cv['test-error-std'][0]
            base_cv_SE_train = np.nan #not reported 
        
        else:
            base_cv = cross_val_score(model,
                                        X_train_tf,
                                        y_train,
                                        scoring= 'accuracy',
                                        cv=cv1, 
                                        n_jobs=-1,
                                        error_score='raise')

            base_cv_score_train = base_cv.mean()
            base_cv_SD_train = base_cv.std()
            base_cv_SE_train = sem(base_cv)



        #----------------------- Evaluate baseline models on test set -------------------------------#
        #---------------------------(no PCA - raw word vectors)-------------------------#
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        print('--Evaluating performance--')

        if model_name=='XGBoost':
            model = model.fit(X_train_tf, y_train)
            y_pred = model.predict(X_train_tf)
            preds = [round(value) for value in y_pred]
            
            base_score_train = accuracy_score(y_train, preds)
            base_auc_train = roc_auc_score(y_train, preds)

            y_pred_test = model.predict(X_test_tf)
            preds_test = [round(value) for value in y_pred_test]

            base_score_test = accuracy_score(y_test, preds_test)
            base_auc_test = roc_auc_score(y_test, preds_test)

            base_perm_acc_train = np.nan
            base_perm_score_train = np.nan
            base_perm_pval_train = np.nan

            base_perm_acc_test = np.nan
            base_perm_score_test = np.nan
            base_perm_pval_test = np.nan
        
        else:
            model = model.fit(X_train_tf, y_train)
            base_score_train = model.score(X_train_tf, y_train)
            base_auc_train = roc_auc_score(y_train, model.predict(X_train_tf))

            base_score_test = model.score(X_test_tf, y_test)
            base_auc_test = roc_auc_score(y_test, model.predict(X_test_tf))

            #base_perm_acc_train, base_perm_score_train, base_perm_pval_train = permutation_test(X_train_tf, y_train, model)
            #base_perm_acc_test, base_perm_score_test, base_perm_pval_test = permutation_test(X_test_tf, y_test, model)

        
        #-------------------------Feature Selection-----------------------
        #Unfitted base estimator 
        name, base = [m for m in models if model_name in m][0]

        #making the forward selection with cv and fit 
        forward = SequentialFeatureSelector(
            base,
            k_features = 'best',
            cv = cv1,
            forward = True, 
            floating = False,
            scoring = 'accuracy', 
            verbose = 2, 
            n_jobs = -1 
            ).fit(X_train_tf, y_train)

        #saving forward-output to a dataframe
        le = pd.DataFrame.from_dict(forward.get_metric_dict()).T

        #finding the index of the best combination of features
        max_index = np.argmax(le.avg_score)

        #saving metrics from the best combination
        feature_names = le.iloc[max_index,3]
        fs_score = le.iloc[max_index,2]
        fs_SD = le.iloc[max_index, 5]
        fs_SE = le.iloc[max_index, 6]

        #Choosing best features in X_train and use that in the rest of the loop
        feature_names_list = list(feature_names)
        X_train_tf_fs = X_train_tf[[c for c in X_train_tf.columns if c in feature_names_list]]
        #defining holdout sets after feature selection
        X_test_tf_fs = X_test_tf[[c for c in X_test_tf.columns if c in feature_names_list]]

        #------------------ Gridsearch parameter optimization-----------------------#

        #Params for LR
        LR_params = [{
            "C":np.logspace(-3,3,7), #Controls strength of regularization, smaller is stronger
            "penalty": ['l1','l2','elasticnet','none'], #Type of regularization, L1=lasso, L2=ridge
            "l1_ratio": np.linspace(0,1,11),
        }]

        LDA_params = [{
            "shrinkage": np.arange(0, 1, 0.01)
        }]

        KNN_params = [{
            "leaf_size": list(range(1,50)),
            "n_neighbors": list(range(1,30)),
            "p":[1,2]
        }]

        RF_params = [{
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)].append(None),
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
            }]

        GaussianNB_params = [{
            'var_smoothing': np.logspace(0,-9, num=100)
        }]

        SVM_params = [{
            'kernel':['linear','rbf','sigmoid'] 
            'C': [0.1, 1, 10, 100, 1000], #L2 regulariation strength
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale','auto'], #Kernel coef for rbf and sigmoid
        }]

        from hyperopt import hp 
        XGB_params = [{
            'eta':np.linspace(0,1,10), #learning rate 
            'gamma': hp.uniform ('gamma', 1,9), #minimum loss reduction for further leaf partition, larger=more conservative
            'max_depth': hp.quniform("max_depth", 3, 18, 1), #max depth of tree, larger vals=more complex model and more prone to overfit
            'lambda': hp.uniform('reg_lambda', 0,1), #L2 regularization on weights 
            'alpha': hp.quniform('reg_alpha', 40,180,1) #L1 regularization on weights 
        }]

        if model_name == 'LR':
            parameters = LR_params
        elif model_name == 'LDA':
            parameters = LDA_params
        elif model_name == 'KNN':
            parameters = KNN_params, 
        elif model_name == 'RF':
            parameters = RF_params
        elif model_name == 'GaussianNB': 
            parameters = GaussianNB_params
        elif model_name == 'SVM': 
            parameters = SVM_params
        elif model_name == 'XGBoost': 
            parameters = XGB_params 

        #making the gridsearch and fitting features and target
        grid = GridSearchCV(
            estimator = model,
            param_grid = parameters,
            scoring = 'accuracy',
            n_jobs = -1,
            cv=cv1, 
            refit = True).fit(X_train_tf_fs, y_train)

        #logging penalty, regurilization strength and best score
        grid_best_params = grid.best_estimator_.get_params() 
        grid_best_score = grid.best_score_ 
        grid_best_SD = grid.cv_results['std_test_score'][grid.best_index_]
        grid_best_model = grid.best_estimator_


        #----------------------Ensemble classification using best gridsearch model ---------------------#
        #------------------------------------ Boosting and bagging -------------------------------------#
        #-----------------(XGBoost already implement this, so not included in this step)----------------#

        #AdaBoost
        boost = AdaBoostClassifier(base_estimator = grid_best_model,
                                        random_state=42)

        #Bagging
        bag = BaggingClassifier(base_estimator = grid_best_model,
                                        random_state=42,
                                        n_jobs = -1)

        #Evaluate Adaboost by cv
        boost_cv = cross_val_score(boost,
                                X_train_tf_fs,
                                y_train,
                                scoring= 'accuracy',
                                cv=cv, 
                                n_jobs=-1,
                                error_score='raise')

        #Evaluate Bagging by CV
        bag_cv = cross_val_score(bag,
                                X_train_tf_fs,
                                y_train,
                                scoring= 'accuracy',
                                cv=cv, 
                                n_jobs=-1,
                                error_score='raise')

        #logging scores
        boost_cv_score = boost_cv.mean()
        boost_cv_SD = boost_cv.std()
        boost_cv_SE = sem(boost_cv)

        bag_cv_score = bag_cv.mean()
        bag_cv_SD = bag_cv.std()
        bag_cv_SE = sem(bag_cv)

        #-----------------------Evaluate tuned models on test set ---------------------------#
        grid_best_model = grid_best_model.fit(X_train_tf_fs, y_train)
        grid_train_score = grid_best_model.score(X_train_tf_fs, y_train)
        grid_train_auc = roc_auc_score(y_train, grid_best_model.predict(X_train_tf_fs))
        grid_test_score = grid_best_model.score(X_test_tf_fs, y_test)
        grid_test_auc = roc_auc_score(y_test, grid_best_model.predict(X_test_tf_fs))
        
        boost = boost.fit(X_train_tf_fs, y_train)
        boost_train_score = boost.score(X_train_tf_fs, y_train)
        boost_train_auc = roc_auc_score(y_train, boost.predict(X_train_tf_fs))
        boost_test_score = boost.score(X_test_tf_fs, y_test)
        boost_test_auc = roc_auc_score(y_test, boost.predict(X_test_tf_fs))

        bag = bag.fit(X_train_tf_fs, y_train)
        bag_train_score = bag.score(X_train_tf_fs, y_train)
        bag_train_auc = roc_auc_score(y_train, bag.predict(X_train_tf_fs))
        bag_test_score = bag.score(X_test_tf_fs, y_test)
        bag_test_auc = roc_auc_score(y_test, bag.predict(X_test_tf_fs))

        #Select best model 
        test_scores = (base_score_test, grid_test_score, boost_test_score, bag_test_score)
        model_names = ('baseline_model', 'tuned_fs_model', 'tuned_fs_boost_model','tuned_fs_bag_model')
        index_max = test_scores.index(max(test_scores)) 
        best_model_name = model_names[index_max]
        best_model_test_score = test_scores[index_max]

        if best_model_name == 'baseline_model': 
            name, best_model = [m for m in models if model_name in m][0]
        elif best_model_name == 'tuned_fs_model':



        best_model_coef = best_model.coef_ 
        best_model_intercept = best_model.intercept_ 

        #Permutation on best model performance 
        if best_model_name=='baseline_model':
            best_model_perm_acc_train, best_model_perm_scores_train, best_model_perm_pval_train = permutation_test(X_train_tf, y_train) 
            best_model_perm_acc_test, best_model_perm_scores_test, best_model_perm_pval_test = permutation_test(X_train_tf, y_train)

            #Permutation on feature importances 
            FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train_tf, y_train)
            FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test_tf, y_test)

        else: 
            best_model_perm_acc_train, best_model_perm_scores_train, best_model_perm_pval_train = permutation_test(X_train_tf_fs, y_train) 
            best_model_perm_acc_test, best_model_perm_scores_test, best_model_perm_pval_test = permutation_test(X_train_tf_fs, y_train)
            
            #Permutation test on best model feature importance 
            FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train_tf_fs, y_train)
            FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test_tf_fs, y_test)

            
            

        #------------------------------ Save outputs ------------------------------#
        output = [target_type,
                input_feat,
                encoder,
                model_name, 
                base_cv_score_train,
                base_cv_SD_train,
                base_cv_SE_train,
                base_score_train, 
                base_auc_train, 
                base_score_test, 
                base_auc_test, 
                base_perm_acc_train, 
                base_perm_score_train, 
                base_perm_pval_train, 
                base_perm_acc_test, 
                base_perm_score_test, 
                base_perm_pval_test, 
                feature_names, 
                fs_score,
                fs_SD, 
                fs_SE,
                grid_best_params, 
                grid_best_score, 
                grid_best_SD, 
                boost_cv_score, 
                boost_cv_SD, 
                boost_cv_SE, 
                bag_cv_score, 
                bag_cv_SD, 
                bag_cv_SE, 
                boost_train_score, 
                boost_train_auc, 
                boost_test_score, 
                boost_test_auc, 
                bag_train_score, 
                bag_train_auc, 
                bag_test_score, 
                bag_test_auc 
                ]

        #Turning outputs into zipped
        zipped = zip(columns, output)
        #Turning zipped into dictionary
        output_dict = dict(zipped)
        #Appending to our data
        dataframe = dataframe.append(output_dict, ignore_index=True)


            

