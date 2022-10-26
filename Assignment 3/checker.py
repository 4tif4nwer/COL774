import os
import numpy as np
import subprocess
# os.chdir("./Q3")
# os.chdir('./Qb')
train = "data/COL774_mammography"
val = "data/COL774_mammography"
test = "data/COL774_mammography"
outfile = "Output/Q1"
for part in ['e']:
    subprocess.run(['python',f'dt_mammography.py',train,val, test,outfile,f'{part}'])
# train = "data/COL774_drug_review"
# val = "data/COL774_drug_review"
# test = "data/COL774_drug_review"
# outfile = "Output/Q2"
# for part in ['c']:
#     subprocess.run(['python',f'dt_drug_review.py',train,val, test,outfile,f'{part}'])
