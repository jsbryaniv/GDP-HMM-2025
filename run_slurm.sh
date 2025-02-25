#!/bin/bash
#SBATCH --job-name=train_deeplearning         # Job name
#SBATCH --time=12:00:00                       # Max execution time (HH:MM:SS)
#SBATCH --partition=gpu                       # Specify GPU partition
#SBATCH --gres=gpu:1                          # Request 1 GPU
#SBATCH --mem=24G                             # Memory allocation

# Set up environment
module load python
module load cuda
source .env/bin/activate

# Define constants
N_ITER=10                # Number of iterations to run
ARGS=$1                  # First argument is the Python script argument
ITER=$2                  # Second argument is the iteration number
if [ -z "$ITER" ]; then  # If ITER is not provided, set it to 0
    ITER=0 
fi

# Define log file paths manually, so jobs can append to the same file
OUT_LOG="outfiles/logs/out_job${ARGS}.txt"
ERR_LOG="outfiles/logs/err_job${ARGS}.txt"
if [ "$ITER" -eq 0 ]; then
    # Clear logs if this is the first iteration
    > "$OUT_LOG"
    > "$ERR_LOG"
else
    # Add spacing to separate job outputs in the logs
    echo -e "\n\n===== New Job Resubmission (ITER $ITER/$N_ITER) =====\n\n" >> "$OUT_LOG"
    echo -e "\n\n===== New Job Resubmission (ITER $ITER/$N_ITER) =====\n\n" >> "$ERR_LOG"
fi


# Print environment details
echo "Running on: $(hostname)" >> "$OUT_LOG"
echo "Slurm Job ID: $SLURM_JOB_ID" >> "$OUT_LOG"
echo "CUDA Devices: [$CUDA_VISIBLE_DEVICES]" >> "$OUT_LOG"
echo "Job Arguments: [$ARGS, $ITER]" >> "$OUT_LOG"

# Run the Python script. Redirect stdout and stderr to log files
python -u main.py $ARGS $ITER >> "$OUT_LOG" 2>> "$ERR_LOG"

# Check if Python script ran successfully
EXIT_CODE=$?
if [ "$EXIT_CODE" -ne 0 ]; then
    # If Python script failed, print error message and exit
    echo "Python script failed (exit code: $EXIT_CODE)." >> "$OUT_LOG"
    echo "$EXIT_CODE" >> "$ERR_LOG"
    exit $EXIT_CODE
fi

# If ITER is less than N_ITER, submit next job
ITER=$((ITER+1))
if [ $ITER -lt $N_ITER ]; then
    echo "Submitting next job with iteration $ITER/$N_ITER at $(date)." >> "$OUT_LOG"
    sbatch run_slurm.sh $ARGS $ITER
fi

# Print completion message
echo "Job finished all iterations at $(date)." >> "$OUT_LOG"
