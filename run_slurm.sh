#!/bin/bash
#SBATCH --job-name=train_deeplearning         # Job name
#SBATCH --output=outfiles/logs/out_%j.txt     # Output log file (%j = job ID)
#SBATCH --error=outfiles/logs/err_%j.txt      # Error log file
#SBATCH --time=12:00:00                       # Max execution time (HH:MM:SS)
#SBATCH --partition=gpu                       # Specify GPU partition (change as needed)
#SBATCH --gres=gpu:1                          # Request 1 GPU (modify if needed)
#SBATCH --mem=24G                             # Memory allocation

# Set up environment
module load python
module load cuda
source activate .env

# Define constants
N_ITER=10

# Get the command-line argument
ARGS=$1
ITER=$2

# If ITER is not provided, set it to 0
if [ -z "$ITER" ]; then
    ITER=0
fi

# Print environment details
echo "Running on: $(hostname)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "CUDA Devices: [$CUDA_VISIBLE_DEVICES]"
echo "Job Arguments: [$ARGS, $ITER]"

# Run your Python script
python -u main.py $ARGS $ITER

# If ITER is less than N_ITER, resubmit the job
if [ $ITER -lt $N_ITER ]; then
    ITER=$((ITER+1))
    echo "Resubmitting job with iteration $ITER/$N_ITER."
    sbatch run_slurm.sh $ARGS $ITER
fi

# Print completion message
echo "Job finished at $(date)"
