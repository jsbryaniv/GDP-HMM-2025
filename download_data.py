
# Import libraries
import os
import json
import shutil
import zipfile
from huggingface_hub import login
from huggingface_hub import snapshot_download


# Define function to download data
def download_data(path_data):

    # Set paths
    path_raw = os.path.join(path_data, "raw_data")
    path_han = os.path.join(path_data, "han")
    path_lung = os.path.join(path_data, "lung")


    ### DOWNLOAD DATA ###
    print("Downloading data.")

    # Login to the Hugging Face Hub from token in config.json
    try:
        # Get token from config
        with open("config.json", "r") as f:
            config = json.load(f)
            token = config["HUGGINGFACE_TOKEN"]
        # Login with token
        login(token)
        print("Successfully logged into Hugging Face!")
    except KeyError:  
        # Exception for when token not in config.json
        raise KeyError("Please add your Hugging Face token to config.json")
    except Exception as e:  
        # Exception for other errors
        raise Exception(f"Error logging into Hugging Face: {e}")

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
    return


# Download data
if __name__ == '__main__':
    
    # Get data path from config
    with open("config.json", "r") as f:
        config = json.load(f)
        path_data = config["PATH_DATA"]

    # Download data
    download_data(path_data)

    
