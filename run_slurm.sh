#!/bin/bash
#SBATCH --job-name=train_deeplearning         # Job name
#SBATCH --output=outfiles/logs/out_$1_%j.txt  # Output log file (%j = job ID)
#SBATCH --error=outfiles/logs/err_$1_%j.txt   # Error log file
#SBATCH --time=12:00:00                       # Max execution time (HH:MM:SS)
#SBATCH --partition=gpu                       # Specify GPU partition (change as needed)
#SBATCH --gres=gpu:1                          # Request 1 GPU (modify if needed)
#SBATCH --mem=24G                             # Memory allocation

# Set up environment
module load python
module load cuda
source activate .env

# Get the command-line argument
ARGS=$1

# Print environment details (useful for debugging)
echo "Running on: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Arguments: $ARGS"

# Run your Python script
python -u main.py $ARGS

# Print completion message
echo "Job finished at $(date)"
