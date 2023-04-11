
import glob
import shutil

#Loop through subjects in BIDS folder
#Find all event.tsv files 
#Copy events.json file into the folder and give it the same name as the event.tsv file (with .json extension)

#bids_path = '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/'
#search_str = bids_path + '/sub*' 
#subject_list = glob.glob(search_str) 
subject_list = ['/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/sub-0085', '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/sub-0086', '/projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS/sub-0087']

for subject in subject_list: 
    #Identify event.tsv files 
    search_eventFiles = subject + '/func/*event*.tsv'
    eventFiles_list = glob.glob(search_eventFiles)

    #Loop through event files 
    for file in eventFiles_list: 
        name =  file.rsplit('/', 1)[1]
        name = name.rsplit('.')[0]
        target_filename = subject + "/func/" + name + ".json"
        shutil.copy("/projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/event_description.json", target_filename)

