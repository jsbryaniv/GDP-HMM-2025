#!/bin/bash
#SBATCH --job-name=test_training       # Job name
#SBATCH --output=outfiles/out_%j.txt   # Output log file (%j = job ID)
#SBATCH --error=outfiles/err_%j.txt    # Error log file
#SBATCH --time=12:00:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=gpu                # Specify GPU partition (change as needed)
#SBATCH --gres=gpu:1                   # Request 1 GPU (modify if needed)
#SBATCH --mem=24G                      # Memory allocation

# Load the required modules (if using an environment module system)
module load python
module load cuda

# Activate your Python environment
source activate .env

# Print environment details (useful for debugging)
echo "Running on: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"

# Run your Python script
python -u main.py

# Print completion message
echo "Job finished at $(date)"
