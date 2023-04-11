import os

print("Running bidsmapper")
os.system('bidsmapper -a /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS -t /projects/MINDLAB2022_MR-semantics-of-depression/scripts/test_line/bidsmap_template_line2.yaml')

print("Running bidscoiner")
os.system('bidscoiner /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/raw /projects/MINDLAB2022_MR-semantics-of-depression/scratch/line/BIDS')
