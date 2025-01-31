# GDP-HMM-2025
Code for the 2025 GDP-HMM competition.

## Getting Started

To get started set up the environment by running the following command:
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
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
    "PATH_DATA": "path/to/data"
}
```
