import os 
import pandas as pd 
import numpy as np 
import nilearn 
import sklearn 
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/dct_fmri_analysis_repo/dct_fMRI_analysis-2/classification/dep_classification/trained_models/scripts')

import batch1_classifiers
from batch1_classifiers import run_logReg_classifier, run_LDA_classifier, run_kNN_classifier, run_NB_classifier, run_RF_classifier, run_SVC_classifier, run_xgboost_classifier

###########################################################################################################
#                                           Run classifiers                                               #
#                                   INPUT: raw respones (encoded)                                         #
###########################################################################################################

#--------------------------- Prepare output dataframe ----------------------------#
dataframe = pd.DataFrame(columns = ['model',
                                    'encoder',
                                    'solver',
                                    'grid_best_params',
                                    'cv_acc',
                                    'cv_SD',
                                    'f1_train',
                                    'auc_train',
                                    'f1_test',
                                    'auc_test',
                                    'perm_acc_train',
                                    'perm_sd_train',
                                    'perm_pval_train',
                                    'perm_acc_test',
                                    'perm_sd_test',
                                    'perm_pval_test'
                                    ])
columns = list(dataframe)

col1 = np.arange(0,1000)
col2 = ['model','encoder']
outcols = np.concatenate([col1, col2])
df_feat_imp_train_full = pd.DataFrame(columns=outcols)
df_feat_imp_test_full = pd.DataFrame(columns=outcols)
df_perm_train_full = pd.DataFrame(columns=outcols)
df_perm_test_full = pd.DataFrame(columns=outcols)

#--------------------- Load data --------------------#
dir = os.getcwd()
raw_resp = pd.read_csv(os.path.join(dir,'classification/dep_classification/features_dfs/raw_responses.csv'), index_col=False)
X = raw_resp.iloc[:,0:293]
y = raw_resp.dep_group

#--------------------- LogReg (no feature selection) ------------------------#
#Label encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_logReg_classifier(X, y, encoder='label', solver='saga', model_name='model1_logReg_label')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'logReg', 'label'
FI_test_df['model'], FI_test_df['encoder'] = 'logReg', 'label'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'logReg', 'label'
perm_test_df['model'], perm_test_df['encoder'] = 'logReg', 'label'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#OneHot encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_logReg_classifier(X, y, encoder='label', solver='saga', model_name='model1_logReg_onehot')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'logReg', 'onehot'
FI_test_df['model'], FI_test_df['encoder'] = 'logReg', 'onehot'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'logReg', 'onehot'
perm_test_df['model'], perm_test_df['encoder'] = 'logReg', 'onehot'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))

#--------------------- LDA (no feature selection) ------------------------#
#Label encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_LDA_classifier(X, y, encoder='label', solver='saga', model_name='model1_LDA_label')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'LDA', 'label'
FI_test_df['model'], FI_test_df['encoder'] = 'LDA', 'label'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'LDA', 'label'
perm_test_df['model'], perm_test_df['encoder'] = 'LDA', 'label'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#OneHot encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_logReg_classifier(X, y, encoder='label', solver='saga', model_name='model1_LDA_onehot')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'LDA', 'onehot'
FI_test_df['model'], FI_test_df['encoder'] = 'LDA', 'onehot'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'LDA', 'onehot'
perm_test_df['model'], perm_test_df['encoder'] = 'LDA', 'onehot'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))

#--------------------- kNN (no feature selection) ------------------------#
#Label encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_kNN_classifier(X, y, encoder='label', solver='saga', model_name='model1_kNN_label')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'kNN', 'label'
FI_test_df['model'], FI_test_df['encoder'] = 'kNN', 'label'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'kNN', 'label'
perm_test_df['model'], perm_test_df['encoder'] = 'kNN', 'label'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#OneHot encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_kNN_classifier(X, y, encoder='label', solver='saga', model_name='model1_kNN_onehot')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'kNN', 'onehot'
FI_test_df['model'], FI_test_df['encoder'] = 'kNN', 'onehot'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'kNN', 'onehot'
perm_test_df['model'], perm_test_df['encoder'] = 'kNN', 'onehot'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#--------------------- RF (no feature selection) ------------------------#
#Label encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_RF_classifier(X, y, encoder='label', solver='saga', model_name='model1_RF_label')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'RF', 'label'
FI_test_df['model'], FI_test_df['encoder'] = 'RF', 'label'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'RF', 'label'
perm_test_df['model'], perm_test_df['encoder'] = 'RF', 'label'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#OneHot encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_RF_classifier(X, y, encoder='label', solver='saga', model_name='model1_RF_onehot')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'RF', 'onehot'
FI_test_df['model'], FI_test_df['encoder'] = 'RF', 'onehot'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'RF', 'onehot'
perm_test_df['model'], perm_test_df['encoder'] = 'RF', 'onehot'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))

#--------------------- BernoulliNB (no feature selection) ------------------------#
#Label encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_NB_classifier(X, y, encoder='label', solver='saga', model_name='model1_NB_label')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'NB', 'label'
FI_test_df['model'], FI_test_df['encoder'] = 'NB', 'label'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'NB', 'label'
perm_test_df['model'], perm_test_df['encoder'] = 'NB', 'label'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#OneHot encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_NB_classifier(X, y, encoder='label', solver='saga', model_name='model1_NB_onehot')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'NB', 'onehot'
FI_test_df['model'], FI_test_df['encoder'] = 'NB', 'onehot'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'NB', 'onehot'
perm_test_df['model'], perm_test_df['encoder'] = 'NB', 'onehot'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))

#--------------------- SVC (no feature selection) ------------------------#
#Label encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_SVC_classifier(X, y, encoder='label', solver='saga', model_name='model1_SVC_label')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'SVC', 'label'
FI_test_df['model'], FI_test_df['encoder'] = 'SVC', 'label'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'SVC', 'label'
perm_test_df['model'], perm_test_df['encoder'] = 'SVC', 'label'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#OneHot encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_SVC_classifier(X, y, encoder='label', solver='saga', model_name='model1_SVC_onehot')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'SVC', 'onehot'
FI_test_df['model'], FI_test_df['encoder'] = 'SVC', 'onehot'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'SVC', 'onehot'
perm_test_df['model'], perm_test_df['encoder'] = 'SVC', 'onehot'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))

#--------------------- XGBoost (no feature selection) ------------------------#
#Label encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_xgboost_classifier(X, y, encoder='label', solver='saga', model_name='model1_xgboost_label')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'xgboost', 'label'
FI_test_df['model'], FI_test_df['encoder'] = 'xgboost', 'label'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'xgboost', 'label'
perm_test_df['model'], perm_test_df['encoder'] = 'xgboost', 'label'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


#OneHot encoder 
output, grid_best_model, perm_scores_train, perm_scores_test, FI_importances_train, FI_importances_test, FI_df = run_xgboost_classifier(X, y, encoder='label', solver='saga', model_name='model1_xgboost_onehot')
dataframe = pd.concat([dataframe, pd.DataFrame([output],columns=columns)], ignore_index=True)
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_performance_rawResp.csv'))

FI_train_df = pd.DataFrame(FI_importances_train)
FI_test_df = pd.DataFrame(FI_importances_test)
FI_train_df['model'], FI_train_df['encoder'] = 'xgboost', 'onehot'
FI_test_df['model'], FI_test_df['encoder'] = 'xgboost', 'onehot'
FI_train_df.columns = df_feat_imp_train_full.columns 
FI_test_df.columns = df_feat_imp_test_full.columns 
df_feat_imp_train_full = pd.concat([df_feat_imp_train_full, FI_train_df])
df_feat_imp_test_full = pd.concat([df_feat_imp_test_full, FI_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/models_FI_rawResp.csv'))

perm_train_df = pd.DataFrame(perm_scores_train).T
perm_test_df = pd.DataFrame(perm_scores_test).T
perm_train_df['model'], perm_train_df['encoder'] = 'xgboost', 'onehot'
perm_test_df['model'], perm_test_df['encoder'] = 'xgboost', 'onehot'
perm_train_df.columns = df_perm_train_full.columns 
perm_test_df.columns = df_perm_test_full.columns 
df_perm_train_full = pd.concat([df_perm_train_full, perm_train_df])
df_perm_test_full = pd.concat([df_perm_test_full, perm_test_df])
dataframe.to_csv(os.path.join(dir,'classification/dep_classification/output/model_rawResp.csv'))


