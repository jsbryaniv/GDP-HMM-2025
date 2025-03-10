
# Import libraries
import os
import sys
import json
import resource

# Open config
with open('config.json', 'r') as f:
    config = json.load(f)

# Extract variables
MACHINE = config['MACHINE']
PATH_DATA = config['PATH_DATA']
PATH_OUTPUT = config['PATH_OUTPUT']
PATH_METADATA = config['PATH_METADATA']

# Apply memory limit only on "carina@mca"
if MACHINE == "carina@mca":
    MAX_MEMORY = 16 * 1024**3  # 16 GB
    print(f"Setting memory limit to {MAX_MEMORY / (1024**3)} GB for {MACHINE}.")
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY, MAX_MEMORY))
