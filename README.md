# GDP-HMM-2025
Code for the 2025 GDP-HMM competition.

## Getting Started

To get started set up the environment by running the following command:
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Next clone the challenge module by running the following command:
```bash
git clone https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge submodules/challenge_repo
```

Configure vscode to use the venv by pressing `Ctrl+Shift+P` and selecting `Python: Select Interpreter` and then selecting the python version in the `.env` folder. It may be labeled as "Recommended".

If you install new packages, make sure to update the `requirements.txt` file by running:
```bash
pip3 freeze -> requirements.txt
```

If you clone this repository to a new machine, be sure to create a new config.json file in the root directory. This file should contain the following:
```json
{
    "MACHINE": "name-of-your-machine",
    "PATH_DATA": "path/to/data",
    "HUGGINGFACE_TOKEN": "your-huggingface-token"
}
```
In particular, to get a Huggingface token, you can sign up for an account at [Huggingface](https://huggingface.co/). Then, you can find your token by going to your profile and clicking on the "API token" tab.

## Downloading data

To download the data, run the following command:
```bash
python download_data.py
```
