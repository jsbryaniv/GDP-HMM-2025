
# Import libraries
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


### Competition Code ###

# Import data_loader
from comptetition_objects import GetLoader

# Get config
cfig = yaml.load(open('submodules/challenge_repo/config_files/config_dummy.yaml'), Loader=yaml.FullLoader)

# Get data loader
loaders = GetLoader(cfig = cfig['loader_params'])
train_loader =loaders.train_dataloader()
val_loader = loaders.val_dataloader()

# Done
print("Done")


### My Code ###

# Get HaN files
path_train_han = "data/han/train"
files = os.listdir(path_train_han)

# Load first file
file = files[0]
data_npz = np.load(os.path.join(path_train_han, file), allow_pickle=True)
data_dict = dict(data_npz)['arr_0'].item()

# Done
print("Done")
