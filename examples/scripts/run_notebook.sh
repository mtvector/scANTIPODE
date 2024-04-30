#!/bin/bash
#SBATCH --job-name singnbconv
#SBATCH --output /home/matthew.schmitz/log/nbconvert_%A_%a.out # %A is replaced by job ID, %a by array index
#SBATCH --error /home/matthew.schmitz/log/nbconvert_%A_%a.err
#SBATCH --time 50:00:00
#SBATCH --partition celltypes
#SBATCH --gres=gpu:1 --constraint="a100|v100"#
#SBATCH --mem 255gb
#SBATCH --ntasks 1

source ~/.bashrc

!nvidia-smi

conda activate antipode

# Read the notebook path for the current array task
NOTEBOOK=/allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Matthew/code/scANTIPODE/examples/1.9.1.8.3_JorstadAll-NoReLU.ipynb

# Command to run jupyter nbconvert
jupyter nbconvert --ExecutePreprocessor.allow_errors=True --to html --execute "${NOTEBOOK}" --output ~/Matthew/code/scANTIPODE/examples/outputs/"executed_$(basename "${NOTEBOOK}")" 
