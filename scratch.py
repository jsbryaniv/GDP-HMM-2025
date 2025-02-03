
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt


# Get HaN files
path_train_han = "data/han/train"
files = os.listdir(path_train_han)

# Load first file
file = files[0]
data_npz = np.load(os.path.join(path_train_han, file), allow_pickle=True)
data_dict = dict(data_npz)['arr_0'].item()

# Plot data
plt.imshow(data, cmap="gray")

# Done
print("Done")
