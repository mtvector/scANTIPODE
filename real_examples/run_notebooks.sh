#!/bin/bash
#SBATCH --job-name nbconvert_array
#SBATCH --output /home/matthew.schmitz/log/nbconvert_%A_%a.out # %A is replaced by job ID, %a by array index
#SBATCH --error /home/matthew.schmitz/log/nbconvert_%A_%a.err
#SBATCH --time 99:00:00
#SBATCH --partition celltypes
#SBATCH --array=115# Create an array job for each line in notebooks.txt
#SBATCH --gres=gpu:1 --constraint="a100|v100"#
#SBATCH --mem 63gb
#SBATCH --ntasks 1
#SBATCH --begin=now#+3hour

source ~/.bashrc

conda activate antipode

# Read the notebook path for the current array task
NOTEBOOK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/Matthew/code/scANTIPODE/real_examples/notebooks.txt)

# Command to run jupyter nbconvert
#jupyter nbconvert --ExecutePreprocessor.allow_errors=True --to html --execute "${NOTEBOOK}" --output ~/Matthew/code/scANTIPODE/examples/outputs/"executed_$(basename "${NOTEBOOK}")" 

# Define output HTML path
OUTPUT_HTML=~/Matthew/code/scANTIPODE/real_examples/outputs/"executed_$(basename "${NOTEBOOK}")" 


jupyter nbconvert --ExecutePreprocessor.allow_errors=True --to html --execute "${NOTEBOOK}" --output "${OUTPUT_HTML}"

# # Check if the output HTML file already exists
# if [ ! -f "${OUTPUT_HTML}.html" ]; then
#     # If it does not exist, convert the notebook to HTML
#     echo "Output HTML (${OUTPUT_HTML}.html) doesn't exist."
#     jupyter nbconvert --ExecutePreprocessor.allow_errors=True --to html --execute "${NOTEBOOK}" --output "${OUTPUT_HTML}"
# else
#     echo "Output HTML ($OUTPUT_HTML) already exists, skipping conversion."
# fi
