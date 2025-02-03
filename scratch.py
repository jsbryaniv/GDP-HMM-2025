
# Import libraries
import os
import numpy as np



data_path = 'data/HNC_001+A4Ac+MOS_25934.npz'
data_npz = np.load(data_path, allow_pickle=True)
data_dict = dict(data_npz)['arr_0'].item()