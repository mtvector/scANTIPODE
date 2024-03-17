#!/bin/bash

#SBATCH --job-name=nbconvert_array
#SBATCH --output=nbconvert_%A_%a.out # %A is replaced by job ID, %a by array index
#SBATCH --error=nbconvert_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --array=1-$(cat notebooks.txt | wc -l) # Create an array job for each line in notebooks.txt

# Load necessary modules or software, e.g., Anaconda
# module load anaconda

# Activate your Python environment with Jupyter installed
# source activate your_env

# Read the notebook path for the current array task
NOTEBOOK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" notebooks.txt)

# Command to run jupyter nbconvert
jupyter nbconvert --ExecutePreprocessor.allow_errors=True --to notebook --execute "${NOTEBOOK}" --output "executed_$(basename "${NOTEBOOK}")"
