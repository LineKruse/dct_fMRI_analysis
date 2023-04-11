# fMRI analysis of DCT study 
December 2022. 

Preprocessing
- First converts raw data to BIDS format (with bidsmapper and bidscoiner) 
- Then runs fmriprep (standard pipeline) 
- Convert event files to BIDS and add to each BOLD file (+ corresponding .json sidecar) 

Analysis
- First-level models (computes and stores all contrasts as .pkl objects) 
- Second-level models (intercept models and two-sample group difference tests) 
- Non-parametric permutation tests 
- Result figures 

All scripts are submitted to cluster and run via "script_submit_to_cluster.py" 

