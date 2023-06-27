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
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

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
from sklearn.naive_bayes import BernoulliNB
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
#                                    Define params and functions                                          #
###########################################################################################################

#Function: permutation test on performance (accuracy)
def permutation_test(features, target, model): 
    score, permutation_scores, pvalue = permutation_test_score(
                                        model,
                                        features,
                                        target,
                                        scoring = 'accuracy',
                                        cv = cv,
                                        n_permutations = n_permutations,
                                        n_jobs = -1,
                                        random_state=42,                                                                     
                                        verbose = 0)
    return(score, permutation_scores, pvalue)

#Function: permutation test on feature importances
def permutation_importance_test(features, target, model):
    r = permutation_importance(model,
                               features,
                               target,
                               n_repeats=1000,
                               scoring = 'accuracy',
                               n_jobs=-1,
                               random_state=42)
    return(r.importances_mean, r.importances_std, r.importances)

###########################################################################################################
#                                          Define classifiers                                             #
#(including gridsearch, CV, evaluation on test set and permutation test on accuracy and feature importances)#
###########################################################################################################


#Function: classification pipeline 
def run_logReg_classifier(X, y, encoder=['label','onehot'], solver='saga', model_name='no_name'):

    #Define cv 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    #Set n permutations 
    n_permutations=1000

    print('------------------------------ Running logReg model -----------------------------')
    print(f'Encoder: {encoder}')
    print(f'Solver: {solver}')

    #Apply encoding if encoder is specified 
    if encoder=='label':
        LE = LabelEncoder()
        X_enc = LE.fit_transform(X)
    if encoder=='onehot':
        OH = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
        X_enc = OH.fit_transform(X)
    
    #Encode outcome labels 
    LE = LabelEncoder()
    labels = LE.fit_transform(y)

    #Create train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_enc, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
    
    print(f'Train set dim: {X_train.shape}')
    print(f'Test set dim: {X_test.shape}')
    
    #Compute class_weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = dict(zip([0,1], class_weights))

    print(f'Class weights: {class_weights}')

    #Define model 
    model = LogisticRegression(class_weight = class_weights, solver = solver, random_state=42, penalty='none')
    
    #Gridsearch parameter optimization
    grid_params = [{
        "C":np.logspace(-3,3,7), #Controls strength of regularization, smaller is stronger
        "penalty": ['l1','l2','elasticnet','none'], #Type of regularization, L1=lasso, L2=ridge
        "l1_ratio": np.linspace(0,1,11),
    }]

    #Making the gridsearch and fitting features and target
    print('----- Running gridsearch ------')

    grid = GridSearchCV(
        estimator = model,
        param_grid = grid_params,
        scoring = 'accuracy',
        n_jobs = -1,
        cv=cv, 
        verbose=1,
        refit = True).fit(X_train, y_train)

    #Logging penalty, regurilization strength and CV acc and sd 
    grid_best_params = grid.best_estimator_.get_params() 
    cv_acc = grid.best_score_ 
    cv_SD = grid.cv_results_['std_test_score'][grid.best_index_]
    grid_best_model = grid.best_estimator_

    """     #Compute cross-validation score of best model from gridsearch
    (identical results to gridsearch scores)
    base_cv = cross_val_score(grid_best_model,
                              X_train,
                              y_train,
                              scoring= 'accuracy',
                              cv=cv, 
                              n_jobs=-1,
                              error_score='raise')

    base_cv_score_train = base_cv.mean()
    base_cv_SD_train = base_cv.std()
    base_cv_SE_train = sem(base_cv) """

    #Train model (using validation set)
    grid_best_model.fit(X_train, y_train)

    #Save best model trained 
    grid_best_model.save(os.path.join(dir, f'classification/dep_classification/trained_models/models/{model_name}_logReg.h5'))

    #Evaluate model on train set 
    train_pred = grid_best_model.predict(X_train)
    #yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
    f1_train = f1_score(y_train, train_pred, average='macro') #Macro needed with imbalanced classes 
    auc_train = roc_auc_score(y_train, grid_best_model.predict(X_train))

    #Evaluate on test set 
    test_pred = grid_best_model.predict(X_test)
    f1_test = f1_score(y_test, test_pred, average='macro')
    auc_test = roc_auc_score(y_test, grid_best_model.predict(X_test))

    #Permutation tests on performance 
    print('----- Running permutation performance test on train set ------')
    perm_acc_train, perm_scores_train, perm_pval_train = permutation_test(X_train, y_train, grid_best_model) 
    perm_sd_train = np.std(perm_scores_train)
    print('----- Running permutation performance test on test set ------')
    perm_acc_test, perm_scores_test, perm_pval_test = permutation_test(X_test, y_test, grid_best_model)
    perm_sd_test = np.std(perm_scores_test)

    #Permutation tests on feature importances 
    print('----- Running permutation importance test on train set ------')
    FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train, y_train, grid_best_model)
    print('----- Running permutation importance test on test set ------')
    FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test, y_test, grid_best_model)

    FI_df = pd.DataFrame({'features': X_train.columns, 'FI_mean_train':FI_mean_train, 'FI_sd_train':FI_SD_train, 'FI_mean_test':FI_mean_test, 'FI_sd_test':FI_SD_test})

    #Define performance output df 
    modelname='LogReg'
    output = [modelname, 
              encoder, 
              solver, 
              grid_best_params,
              cv_acc,
              cv_SD, 
              f1_train, 
              auc_train, 
              f1_test, 
              auc_test, 
              perm_acc_train, 
              perm_sd_train,
              perm_pval_train, 
              perm_acc_test, 
              perm_sd_test,
              perm_pval_test]

    return output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df

def run_LDA_classifier(X, y, encoder=['label','onehot'], solver='svd', model_name='no_name'):

    #Define cv 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    #Set n permutations 
    n_permutations=1000

    print('------------------------------ Running LDA model -----------------------------')
    print(f'Encoder: {encoder}')
    print(f'Solver: {solver}')

    #Apply encoding if encoder is specified 
    if encoder=='label':
        LE = LabelEncoder()
        X_enc = LE.fit_transform(X)
    if encoder=='onehot':
        OH = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
        X_enc = OH.fit_transform(X)
    
    #Encode outcome labels 
    LE = LabelEncoder()
    labels = LE.fit_transform(y)

    #Create train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_enc, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
    
    print(f'Train set dim: {X_train.shape}')
    print(f'Test set dim: {X_test.shape}')
    
    #Compute class_weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = dict(zip([0,1], class_weights))

    print(f'Class weights: {class_weights}')

    #Define model
    np.random.seed(42)
    model = LinearDiscriminantAnalysis(solver=solver) #LDA does not take a random_state param and class_weight (class priors are automatically inferred from data)
    
    #Gridsearch parameter optimization
    grid_params = [
        {"solver": ['svd', 'lsqr', 'eigen']
        },
        {"solver": ['lsqr', 'eigen'],
         "shrinkage": np.arange(0, 1, 0.01)
        }]

    #Making the gridsearch and fitting features and target
    print('----- Running gridsearch ------')

    grid = GridSearchCV(
        estimator = model,
        param_grid = grid_params,
        scoring = 'accuracy',
        n_jobs = -1,
        cv=cv, 
        verbose=1,
        refit = True).fit(X_train, y_train)

    #Logging penalty, regurilization strength and CV acc and sd 
    grid_best_params = grid.best_estimator_.get_params() 
    cv_acc = grid.best_score_ 
    cv_SD = grid.cv_results_['std_test_score'][grid.best_index_]
    grid_best_model = grid.best_estimator_

    """     #Compute cross-validation score of best model from gridsearch
    (identical results to gridsearch scores)
    base_cv = cross_val_score(grid_best_model,
                              X_train,
                              y_train,
                              scoring= 'accuracy',
                              cv=cv, 
                              n_jobs=-1,
                              error_score='raise')

    base_cv_score_train = base_cv.mean()
    base_cv_SD_train = base_cv.std()
    base_cv_SE_train = sem(base_cv) """

    #Train model (using validation set)
    grid_best_model.fit(X_train, y_train)
    
    #Save best model trained 
    grid_best_model.save(os.path.join(dir, f'classification/dep_classification/trained_models/models/{model_name}_LDA.h5'))

    #Evaluate model on train set 
    train_pred = grid_best_model.predict(X_train)
    #yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
    f1_train = f1_score(y_train, train_pred, average='macro') #Macro needed with imbalanced classes 
    auc_train = roc_auc_score(y_train, grid_best_model.predict(X_train))

    #Evaluate on test set 
    test_pred = grid_best_model.predict(X_test)
    f1_test = f1_score(y_test, test_pred, average='macro')
    auc_test = roc_auc_score(y_test, grid_best_model.predict(X_test))

    #Permutation tests on performance 
    print('----- Running permutation performance test on train set ------')
    perm_acc_train, perm_scores_train, perm_pval_train = permutation_test(X_train, y_train, grid_best_model) 
    perm_sd_train = np.std(perm_scores_train)
    print('----- Running permutation performance test on test set ------')
    perm_acc_test, perm_scores_test, perm_pval_test = permutation_test(X_test, y_test, grid_best_model)
    perm_sd_test = np.std(perm_scores_test)

    #Permutation tests on feature importances 
    print('----- Running permutation importance test on train set ------')
    FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train, y_train, grid_best_model)
    print('----- Running permutation importance test on test set ------')
    FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test, y_test, grid_best_model)

    FI_df = pd.DataFrame({'features': X_train.columns, 'FI_mean_train':FI_mean_train, 'FI_sd_train':FI_SD_train, 'FI_mean_test':FI_mean_test, 'FI_sd_test':FI_SD_test})

    #Define performance output df 
    modelname='LDA'
    output = [modelname, 
              encoder, 
              solver, 
              grid_best_params,
              cv_acc,
              cv_SD, 
              f1_train, 
              auc_train, 
              f1_test, 
              auc_test, 
              perm_acc_train, 
              perm_sd_train,
              perm_pval_train, 
              perm_acc_test, 
              perm_sd_test,
              perm_pval_test]

    return output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df


def run_kNN_classifier(X, y, encoder=['label','onehot'], solver='auto', model_name='no_name'):

    #Define cv 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    #Set n permutations 
    n_permutations=1000
    print('------------------------------ Running kNN model -----------------------------')
    print(f'Encoder: {encoder}')
    print(f'Solver: {solver}')

    #Apply encoding if encoder is specified 
    if encoder=='label':
        LE = LabelEncoder()
        X_enc = LE.fit_transform(X)
    if encoder=='onehot':
        OH = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
        X_enc = OH.fit_transform(X)
    
    #Encode outcome labels 
    LE = LabelEncoder()
    labels = LE.fit_transform(y)

    #Create train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_enc, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
    
    print(f'Train set dim: {X_train.shape}')
    print(f'Test set dim: {X_test.shape}')
    
    #Compute class_weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = dict(zip([0,1], class_weights))

    print(f'Class weights: {class_weights}')

    #Define model
    np.random.seed(42)
    model = KNeighborsClassifier() #Does not take class weight input 
    
    #Gridsearch parameter optimization
    grid_params = [{
            "leaf_size": list(range(1,50)),
            "n_neighbors": list(range(1,30)),
            "p":[1,2] 
        }]

    #Making the gridsearch and fitting features and target
    print('----- Running gridsearch ------')

    grid = GridSearchCV(
        estimator = model,
        param_grid = grid_params,
        scoring = 'accuracy',
        n_jobs = -1,
        cv=cv, 
        verbose=1,
        refit = True).fit(X_train, y_train)

    #Logging penalty, regurilization strength and CV acc and sd 
    grid_best_params = grid.best_estimator_.get_params() 
    cv_acc = grid.best_score_ 
    cv_SD = grid.cv_results_['std_test_score'][grid.best_index_]
    grid_best_model = grid.best_estimator_

    """     #Compute cross-validation score of best model from gridsearch
    (identical results to gridsearch scores)
    base_cv = cross_val_score(grid_best_model,
                              X_train,
                              y_train,
                              scoring= 'accuracy',
                              cv=cv, 
                              n_jobs=-1,
                              error_score='raise')

    base_cv_score_train = base_cv.mean()
    base_cv_SD_train = base_cv.std()
    base_cv_SE_train = sem(base_cv) """

    #Train model (using validation set)
    grid_best_model.fit(X_train, y_train)

    #Save best model trained 
    grid_best_model.save(os.path.join(dir, f'classification/dep_classification/trained_models/models/{model_name}_kNN.h5'))

    #Evaluate model on train set 
    train_pred = grid_best_model.predict(X_train)
    #yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
    f1_train = f1_score(y_train, train_pred, average='macro') #Macro needed with imbalanced classes 
    auc_train = roc_auc_score(y_train, grid_best_model.predict(X_train))

    #Evaluate on test set 
    test_pred = grid_best_model.predict(X_test)
    f1_test = f1_score(y_test, test_pred, average='macro')
    auc_test = roc_auc_score(y_test, grid_best_model.predict(X_test))

    #Permutation tests on performance 
    print('----- Running permutation performance test on train set ------')
    perm_acc_train, perm_scores_train, perm_pval_train = permutation_test(X_train, y_train, grid_best_model) 
    perm_sd_train = np.std(perm_scores_train)
    print('----- Running permutation performance test on test set ------')
    perm_acc_test, perm_scores_test, perm_pval_test = permutation_test(X_test, y_test, grid_best_model)
    perm_sd_test = np.std(perm_scores_test)

    #Permutation tests on feature importances 
    print('----- Running permutation importance test on train set ------')
    FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train, y_train, grid_best_model)
    print('----- Running permutation importance test on test set ------')
    FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test, y_test, grid_best_model)

    FI_df = pd.DataFrame({'features': X_train.columns, 'FI_mean_train':FI_mean_train, 'FI_sd_train':FI_SD_train, 'FI_mean_test':FI_mean_test, 'FI_sd_test':FI_SD_test})

    #Define performance output df 
    modelname='kNN'
    output = [modelname, 
              encoder, 
              solver, 
              grid_best_params,
              cv_acc,
              cv_SD, 
              f1_train, 
              auc_train, 
              f1_test, 
              auc_test, 
              perm_acc_train, 
              perm_sd_train,
              perm_pval_train, 
              perm_acc_test, 
              perm_sd_test,
              perm_pval_test]

    return output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df



def run_RF_classifier(X, y, encoder=['label','onehot'], solver=None, model_name='no_name'):

    #Define cv 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    #Set n permutations 
    n_permutations=1000
    print('------------------------------ Running RF model -----------------------------')
    print(f'Encoder: {encoder}')
    print(f'Solver: {solver}')

    #Apply encoding if encoder is specified 
    if encoder=='label':
        LE = LabelEncoder()
        X_enc = LE.fit_transform(X)
    if encoder=='onehot':
        OH = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
        X_enc = OH.fit_transform(X)
    
    #Encode outcome labels 
    LE = LabelEncoder()
    labels = LE.fit_transform(y)

    #Create train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_enc, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
    
    print(f'Train set dim: {X_train.shape}')
    print(f'Test set dim: {X_test.shape}')
    
    #Compute class_weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = dict(zip([0,1], class_weights))

    print(f'Class weights: {class_weights}')

    #Define model
    model = RandomForestClassifier(class_weight=class_weights,random_state=42)
    
    #Gridsearch parameter optimization
    grid_params = [{
            'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
            }]

    #Making the gridsearch and fitting features and target
    print('----- Running gridsearch ------')

    grid = GridSearchCV(
        estimator = model,
        param_grid = grid_params,
        scoring = 'accuracy',
        n_jobs = -1,
        cv=cv, 
        verbose=2,
        refit = True).fit(X_train, y_train)

    #Logging penalty, regurilization strength and CV acc and sd 
    grid_best_params = grid.best_estimator_.get_params() 
    cv_acc = grid.best_score_ 
    cv_SD = grid.cv_results_['std_test_score'][grid.best_index_]
    grid_best_model = grid.best_estimator_

    """     #Compute cross-validation score of best model from gridsearch
    (identical results to gridsearch scores)
    base_cv = cross_val_score(grid_best_model,
                              X_train,
                              y_train,
                              scoring= 'accuracy',
                              cv=cv, 
                              n_jobs=-1,
                              error_score='raise')

    base_cv_score_train = base_cv.mean()
    base_cv_SD_train = base_cv.std()
    base_cv_SE_train = sem(base_cv) """

    #Train model (using validation set)
    grid_best_model.fit(X_train, y_train)

    #Save best model trained 
    grid_best_model.save(os.path.join(dir, f'classification/dep_classification/trained_models/models/{model_name}_RF.h5'))

    #Evaluate model on train set 
    train_pred = grid_best_model.predict(X_train)
    #yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
    f1_train = f1_score(y_train, train_pred, average='macro') #Macro needed with imbalanced classes 
    auc_train = roc_auc_score(y_train, grid_best_model.predict(X_train))

    #Evaluate on test set 
    test_pred = grid_best_model.predict(X_test)
    f1_test = f1_score(y_test, test_pred, average='macro')
    auc_test = roc_auc_score(y_test, grid_best_model.predict(X_test))

    #Permutation tests on performance 
    print('----- Running permutation performance test on train set ------')
    perm_acc_train, perm_scores_train, perm_pval_train = permutation_test(X_train, y_train, grid_best_model) 
    perm_sd_train = np.std(perm_scores_train)
    print('----- Running permutation performance test on test set ------')
    perm_acc_test, perm_scores_test, perm_pval_test = permutation_test(X_test, y_test, grid_best_model)
    perm_sd_test = np.std(perm_scores_test)

    #Permutation tests on feature importances 
    print('----- Running permutation importance test on train set ------')
    FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train, y_train, grid_best_model)
    print('----- Running permutation importance test on test set ------')
    FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test, y_test, grid_best_model)

    FI_df = pd.DataFrame({'features': X_train.columns, 'FI_mean_train':FI_mean_train, 'FI_sd_train':FI_SD_train, 'FI_mean_test':FI_mean_test, 'FI_sd_test':FI_SD_test})

    #Define performance output df 
    modelname='RF'
    output = [modelname, 
              encoder, 
              solver, 
              grid_best_params,
              cv_acc,
              cv_SD, 
              f1_train, 
              auc_train, 
              f1_test, 
              auc_test, 
              perm_acc_train, 
              perm_sd_train,
              perm_pval_train, 
              perm_acc_test, 
              perm_sd_test,
              perm_pval_test]

    return output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df


def run_NB_classifier(X, y, encoder=['label','onehot'], solver=None, model_name='no_name'):

    #Define cv 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    #Set n permutations 
    n_permutations=1000
    print('------------------------------ Running NB model -----------------------------')
    print(f'Encoder: {encoder}')
    print(f'Solver: {solver}')

    #Apply encoding if encoder is specified 
    if encoder=='label':
        LE = LabelEncoder()
        X_enc = LE.fit_transform(X)
    if encoder=='onehot':
        OH = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
        X_enc = OH.fit_transform(X)
    
    #Encode outcome labels 
    LE = LabelEncoder()
    labels = LE.fit_transform(y)

    #Create train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_enc, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
    
    print(f'Train set dim: {X_train.shape}')
    print(f'Test set dim: {X_test.shape}')
    
    #Compute class_weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = dict(zip([0,1], class_weights))

    print(f'Class weights: {class_weights}')

    #Define model
    model = BernoulliNB(binarize=None) #Doesn't do anything with random initiation, takes no random seed input. Adjust priors of classes according to the data automatically.
    
    #Gridsearch parameter optimization
    grid_params = [{
            'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]
        }]

    #Making the gridsearch and fitting features and target
    print('----- Running gridsearch ------')

    grid = GridSearchCV(
        estimator = model,
        param_grid = grid_params,
        scoring = 'accuracy',
        n_jobs = -1,
        cv=cv, 
        verbose=1,
        refit = True).fit(X_train, y_train)

    #Logging penalty, regurilization strength and CV acc and sd 
    grid_best_params = grid.best_estimator_.get_params() 
    cv_acc = grid.best_score_ 
    cv_SD = grid.cv_results_['std_test_score'][grid.best_index_]
    grid_best_model = grid.best_estimator_

    """     #Compute cross-validation score of best model from gridsearch
    (identical results to gridsearch scores)
    base_cv = cross_val_score(grid_best_model,
                              X_train,
                              y_train,
                              scoring= 'accuracy',
                              cv=cv, 
                              n_jobs=-1,
                              error_score='raise')

    base_cv_score_train = base_cv.mean()
    base_cv_SD_train = base_cv.std()
    base_cv_SE_train = sem(base_cv) """

    #Train model (using validation set)
    grid_best_model.fit(X_train, y_train)

    #Save best model trained 
    grid_best_model.save(os.path.join(dir, f'classification/dep_classification/trained_models/models/{model_name}_NB.h5'))

    #Evaluate model on train set 
    train_pred = grid_best_model.predict(X_train)
    #yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
    f1_train = f1_score(y_train, train_pred, average='macro') #Macro needed with imbalanced classes 
    auc_train = roc_auc_score(y_train, grid_best_model.predict(X_train))

    #Evaluate on test set 
    test_pred = grid_best_model.predict(X_test)
    f1_test = f1_score(y_test, test_pred, average='macro')
    auc_test = roc_auc_score(y_test, grid_best_model.predict(X_test))

    #Permutation tests on performance 
    print('----- Running permutation performance test on train set ------')
    perm_acc_train, perm_scores_train, perm_pval_train = permutation_test(X_train, y_train, grid_best_model) 
    perm_sd_train = np.std(perm_scores_train)
    print('----- Running permutation performance test on test set ------')
    perm_acc_test, perm_scores_test, perm_pval_test = permutation_test(X_test, y_test, grid_best_model)
    perm_sd_test = np.std(perm_scores_test)

    #Permutation tests on feature importances 
    print('----- Running permutation importance test on train set ------')
    FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train, y_train, grid_best_model)
    print('----- Running permutation importance test on test set ------')
    FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test, y_test, grid_best_model)

    FI_df = pd.DataFrame({'features': X_train.columns, 'FI_mean_train':FI_mean_train, 'FI_sd_train':FI_SD_train, 'FI_mean_test':FI_mean_test, 'FI_sd_test':FI_SD_test})

    #Define performance output df 
    modelname='NB'
    output = [modelname, 
              encoder, 
              solver, 
              grid_best_params,
              cv_acc,
              cv_SD, 
              f1_train, 
              auc_train, 
              f1_test, 
              auc_test, 
              perm_acc_train, 
              perm_sd_train,
              perm_pval_train, 
              perm_acc_test, 
              perm_sd_test,
              perm_pval_test]

    return output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df

def run_SVC_classifier(X, y, encoder=['label','onehot'], kernel=None, model_name='no_name'):

    #Define cv 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    #Set n permutations 
    n_permutations=1000
    print('------------------------------ Running SVC model -----------------------------')
    print(f'Encoder: {encoder}')
    print(f'Solver: {solver}')

    #Apply encoding if encoder is specified 
    if encoder=='label':
        LE = LabelEncoder()
        X_enc = LE.fit_transform(X)
    if encoder=='onehot':
        OH = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
        X_enc = OH.fit_transform(X)
    
    #Encode outcome labels 
    LE = LabelEncoder()
    labels = LE.fit_transform(y)

    #Create train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_enc, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
    
    print(f'Train set dim: {X_train.shape}')
    print(f'Test set dim: {X_test.shape}')
    
    #Compute class_weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = dict(zip([0,1], class_weights))

    print(f'Class weights: {class_weights}')

    #Define model
    if kernel is not None: 
        model = SVC(kernel=kernel, class_weight=class_weights, random_state=42) #Run with chosen kernel and don't include kernel in gridsearch
    else: 
        model = SVC(class_weight=class_weights, random_state=42) #Initate with default kernel, and run different kernels in gridseach
    
    #Gridsearch parameter optimization
    if kernel is None: 
        grid_params = [{
                'kernel':['linear','rbf','sigmoid'],
                'C': [0.1, 1, 10, 100, 1000], #L2 regulariation strength
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale','auto'], #Kernel coef for rbf and sigmoid
            }]
    else: 
        grid_params = [{
                'C': [0.1, 1, 10, 100, 1000], #L2 regulariation strength
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale','auto'], #Kernel coef for rbf and sigmoid
            }] 

    #Making the gridsearch and fitting features and target
    print('----- Running gridsearch ------')

    grid = GridSearchCV(
        estimator = model,
        param_grid = grid_params,
        scoring = 'accuracy',
        n_jobs = -1,
        cv=cv, 
        verbose=1,
        refit = True).fit(X_train, y_train)

    #Logging penalty, regurilization strength and CV acc and sd 
    grid_best_params = grid.best_estimator_.get_params() 
    cv_acc = grid.best_score_ 
    cv_SD = grid.cv_results_['std_test_score'][grid.best_index_]
    grid_best_model = grid.best_estimator_

    """     #Compute cross-validation score of best model from gridsearch
    (identical results to gridsearch scores)
    base_cv = cross_val_score(grid_best_model,
                              X_train,
                              y_train,
                              scoring= 'accuracy',
                              cv=cv, 
                              n_jobs=-1,
                              error_score='raise')

    base_cv_score_train = base_cv.mean()
    base_cv_SD_train = base_cv.std()
    base_cv_SE_train = sem(base_cv) """

    #Train model (using validation set)
    grid_best_model.fit(X_train, y_train)

    #Save best model trained 
    grid_best_model.save(os.path.join(dir, f'classification/dep_classification/trained_models/models/{model_name}_SVC.h5'))

    #Evaluate model on train set 
    train_pred = grid_best_model.predict(X_train)
    #yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
    f1_train = f1_score(y_train, train_pred, average='macro') #Macro needed with imbalanced classes 
    auc_train = roc_auc_score(y_train, grid_best_model.predict(X_train))

    #Evaluate on test set 
    test_pred = grid_best_model.predict(X_test)
    f1_test = f1_score(y_test, test_pred, average='macro')
    auc_test = roc_auc_score(y_test, grid_best_model.predict(X_test))

    #Permutation tests on performance 
    print('----- Running permutation performance test on train set ------')
    perm_acc_train, perm_scores_train, perm_pval_train = permutation_test(X_train, y_train, grid_best_model) 
    perm_sd_train = np.std(perm_scores_train)
    print('----- Running permutation performance test on test set ------')
    perm_acc_test, perm_scores_test, perm_pval_test = permutation_test(X_test, y_test, grid_best_model)
    perm_sd_test = np.std(perm_scores_test)

    #Permutation tests on feature importances 
    print('----- Running permutation importance test on train set ------')
    FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train, y_train, grid_best_model)
    print('----- Running permutation importance test on test set ------')
    FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test, y_test, grid_best_model)

    FI_df = pd.DataFrame({'features': X_train.columns, 'FI_mean_train':FI_mean_train, 'FI_sd_train':FI_SD_train, 'FI_mean_test':FI_mean_test, 'FI_sd_test':FI_SD_test})

    #Define performance output df 
    modelname='SVC'
    solver=kernel
    output = [modelname, 
              encoder, 
              solver, 
              grid_best_params,
              cv_acc,
              cv_SD, 
              f1_train, 
              auc_train, 
              f1_test, 
              auc_test, 
              perm_acc_train, 
              perm_sd_train,
              perm_pval_train, 
              perm_acc_test, 
              perm_sd_test,
              perm_pval_test]

    return output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df

def run_xgboost_classifier(X, y, encoder=['label','onehot'], solver=None, model_name='no_name'):

    #Define cv 
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    #Set n permutations 
    n_permutations=1000

    print('------------------------------ Running xgboost model -----------------------------')
    print(f'Encoder: {encoder}')
    print(f'Solver: {solver}')

    #Apply encoding if encoder is specified 
    if encoder=='label':
        LE = LabelEncoder()
        X_enc = LE.fit_transform(X)
    if encoder=='onehot':
        OH = ce.OneHotEncoder(handle_unknown='return_nan',return_df=True,use_cat_names=True)
        X_enc = OH.fit_transform(X)
    
    #Encode outcome labels 
    LE = LabelEncoder()
    labels = LE.fit_transform(y)

    #Create train-test split 
    X_train, X_test, y_train, y_test = train_test_split(X_enc, labels,
                                                    stratify = labels,
                                                    test_size=0.3,
                                                    random_state=42)
    
    print(f'Train set dim: {X_train.shape}')
    print(f'Test set dim: {X_test.shape}')
    
    #Compute class_weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=y_train)
    class_weights = dict(zip([0,1], class_weights))

    print(f'Class weights: {class_weights}')

    #Define model
    model = xgb.XGBClassifier(objective="binary:logistic",seed=42)
    
    #Gridsearch parameter optimization
    grid_params = [{
        'n_estimators': [50,100,150],
        'max_depth': [5,10,15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.7],
    }]

    #Making the gridsearch and fitting features and target
    print('----- Running gridsearch ------')

    grid = GridSearchCV(
        estimator = model,
        param_grid = grid_params,
        scoring = 'accuracy',
        n_jobs = -1,
        cv=cv, 
        verbose=1,
        refit = True).fit(X_train, y_train)

    #Logging penalty, regurilization strength and CV acc and sd 
    grid_best_params = grid.best_estimator_.get_params() 
    cv_acc = grid.best_score_ 
    cv_SD = grid.cv_results_['std_test_score'][grid.best_index_]
    grid_best_model = grid.best_estimator_

    """     #Compute cross-validation score of best model from gridsearch
    (identical results to gridsearch scores)
    base_cv = cross_val_score(grid_best_model,
                              X_train,
                              y_train,
                              scoring= 'accuracy',
                              cv=cv, 
                              n_jobs=-1,
                              error_score='raise')

    base_cv_score_train = base_cv.mean()
    base_cv_SD_train = base_cv.std()
    base_cv_SE_train = sem(base_cv) """

    #Train model (using validation set)
    grid_best_model.fit(X_train, y_train)

    #Save best model trained 
    grid_best_model.save(os.path.join(dir, f'classification/dep_classification/trained_models/models/{model_name}_xgboost.h5'))

    #Evaluate model on train set 
    train_pred = grid_best_model.predict(X_train)
    #yhat_train = list(map(lambda x: 1 if x>0.5 else 0, train_pred))
    f1_train = f1_score(y_train, train_pred, average='macro') #Macro needed with imbalanced classes 
    auc_train = roc_auc_score(y_train, grid_best_model.predict(X_train))

    #Evaluate on test set 
    test_pred = grid_best_model.predict(X_test)
    f1_test = f1_score(y_test, test_pred, average='macro')
    auc_test = roc_auc_score(y_test, grid_best_model.predict(X_test))

    #Permutation tests on performance 
    print('----- Running permutation performance test on train set ------')
    perm_acc_train, perm_scores_train, perm_pval_train = permutation_test(X_train, y_train, grid_best_model) 
    perm_sd_train = np.std(perm_scores_train)
    print('----- Running permutation performance test on test set ------')
    perm_acc_test, perm_scores_test, perm_pval_test = permutation_test(X_test, y_test, grid_best_model)
    perm_sd_test = np.std(perm_scores_test)

    #Permutation tests on feature importances 
    print('----- Running permutation importance test on train set ------')
    FI_mean_train, FI_SD_train, FI_importances_train = permutation_importance_test(X_train, y_train, grid_best_model)
    print('----- Running permutation importance test on test set ------')
    FI_mean_test, FI_SD_test, FI_importances_test = permutation_importance_test(X_test, y_test, grid_best_model)

    FI_df = pd.DataFrame({'features': X_train.columns, 'FI_mean_train':FI_mean_train, 'FI_sd_train':FI_SD_train, 'FI_mean_test':FI_mean_test, 'FI_sd_test':FI_SD_test})

    #Define performance output df 
    modelname='xgboost'
    output = [modelname, 
              encoder, 
              solver, 
              grid_best_params,
              cv_acc,
              cv_SD, 
              f1_train, 
              auc_train, 
              f1_test, 
              auc_test, 
              perm_acc_train, 
              perm_sd_train,
              perm_pval_train, 
              perm_acc_test, 
              perm_sd_test,
              perm_pval_test]

    return output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df

