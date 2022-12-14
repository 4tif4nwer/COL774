import os
import numpy as np
import subprocess

# CNN
data_path = "data"
outfile = "Output/Q3"
for part in ['f']:#['b', 'c', 'd', 'e', 'f', 'g']:
    subprocess.run(['python',f'cnn.py',data_path])