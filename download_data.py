
# Import libraries
import os
import json
import zipfile
from huggingface_hub import login
from huggingface_hub import snapshot_download

# Set base path
path_data = "data"

# Create directories
path_raw = os.path.join(path_data, "raw_data")
path_han = os.path.join(path_data, "han")
path_lung = os.path.join(path_data, "lung")
os.makedirs(path_raw, exist_ok=True)
os.makedirs(path_han, exist_ok=True)
os.makedirs(path_lung, exist_ok=True)


### DOWNLOAD DATA ###
print("Downloading data.")

# Login to the Hugging Face Hub from token in config.json
try:
    # Get token from config
    with open("config.json", "r") as f:
        config = json.load(f)
        token = config["hf_token"]
    # Login with token
    login(token)
    print("Successfully logged into Hugging Face!")
except KeyError:  
    # Exception for when token not in config.json
    print("Please add your Hugging Face token to config.json")
except Exception as e:  
    # Exception for other errors
    print(f"Error logging into Hugging Face: {e}")

# Download the dataset
snapshot_download(
    repo_id="Jungle15/GDP-HMM_Challenge", 
    repo_type="dataset", 
    local_dir=path_raw
)


### UNZIP DATA ###
print("Unzipping data.")

# Unzip files
paths = [
    (f"{path_raw}/train_HaN.zip", path_han),
    (f"{path_raw}/train_Lung.zip", path_lung),
    (f"{path_raw}/valid_HaN_nodose.zip", path_han),
    (f"{path_raw}/valid_Lung_nodose.zip", path_lung)
]
for (in_path, out_path) in paths:
    print(f"-- Unzipping {in_path} to {out_path}")
    with zipfile.ZipFile(in_path, 'r') as zip_ref:
        zip_ref.extractall(out_path)



# Done
print('Done')

