import os 
import pandas as pd 
from glob import glob
import numpy as np 

#-------------------------------- Import and clean behav data --------------------------------#
behav_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/logfiles_raw/'
files = glob(os.path.join(behav_dir, '*.csv'))
files = [file for file in files if str.split(file, '/')[-1][0].isdigit()] #n=79
#files = [file for file in files if file[0].isdigit()] #86 subjects 
#exclude = ['015.csv','027.csv','035.csv'] #Fell asleep, too many missing responses 
#sub_files = [file for file in files if file not in exclude]

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

exclude = []
df['n_missing'] = 0

#Check missing responses by subject by block 
for i in range(0,len(df['ID'].unique())): 
    sub = df['ID'].unique()[i]
    total_nan = 0
    for i in range(1,4): 
        na_resp = [str(resp) for resp in df.choice.loc[(df['ID']==sub) & (df['block']==i)]].count('nan')
        total_nan= total_nan + na_resp
        if na_resp>5: 
            exclude.append(sub)
    df['n_missing'].loc[df['ID']==sub] = total_nan
        
exclude = np.unique(np.array(exclude))

#Remove subs in exclude (the ~ reverses true/false, so it becomes "is not in")
df = df[~df['ID'].isin(exclude)] #n=75 

#Save 
df.to_csv('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/cleaned_behav_and_survey_data/behav_all.csv')  


#-------------------------- Load and clean post-fMRI survey data (PHQ + PID5) --------------------------------#
survey_dir = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/fMRI_post_survey_raw/'
files = glob(os.path.join(survey_dir, '*vol1_data.csv'))

df_list = []

for file in files: 
    df2 = pd.read_csv(file)
    questions = df2.stimulus 
    responses = df2.responseClean

    df3 = pd.DataFrame(list(zip(questions, responses)), columns=['item','value'])
    df3 = df3[~df3['item'].isna()]
    remove = ['follow-up','email']
    df3 = df3[~df3['item'].isin(remove)]
    df3.drop(df3.tail(1).index,inplace=True)
    df3 = df3.set_index('item').T

    #Compute PHQ9 sum score 
    phq_cols = ['phq9_1','phq9_2', 'phq9_3', 'phq9_4', 'phq9_5', 'phq9_6', 'phq9_7', 'phq9_8', 'phq9_9']
    df3[phq_cols] = df3[phq_cols].apply(pd.to_numeric)
    df3['PHQ9_sum'] = df3[phq_cols].sum(axis=1)
    df3['ID'] = df2.fMRI_id[0]

    df_list.append(df3)

df_surv_all = pd.concat(df_list)
df_surv_all.to_csv('/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/cleaned_behav_and_survey_data/survey_all.csv')  
