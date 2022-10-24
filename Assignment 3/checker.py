import os
import numpy as np
import subprocess
# os.chdir("./Q3")
# os.chdir('./Qb')
train = "data/COL774_mammography"
val = "data/COL774_mammography"
test = "data/COL774_mammography"
outfile = "Output/Q1"
subprocess.run(['python',f'dt_mammography.py',train,val, test,outfile,'f'])