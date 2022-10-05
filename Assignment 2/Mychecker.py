import os
import numpy as np
import subprocess
os.chdir("./Q2")
os.chdir('./Qb')
train = "D:/GitHub/COL774/Assignment 2/data/part2_data"
test = "D:/GitHub/COL774/Assignment 2/data/part2_data"
subprocess.run(['python',f'qb.py',train, test])