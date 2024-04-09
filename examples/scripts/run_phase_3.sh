#!/bin/bash
#SBATCH --job-name phase3
#SBATCH --output /home/matthew.schmitz/log/phase3_%A_%a.out # %A is replaced by job ID, %a by array index
#SBATCH --error /home/matthew.schmitz/log/phase3_%A_%a.err
#SBATCH --time 72:00:00
#SBATCH --partition celltypes
#SBATCH --array=1-4 # Create an array job for each line in notebooks.txt
#SBATCH --gres=gpu:1 --constraint="a100|v100"#
#SBATCH --mem 128gb
#SBATCH --ntasks 1

source ~/.bashrc

conda activate antipode

# Read the notebook path for the current array task
NOTEBOOK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/Matthew/code/scANTIPODE/examples/scripts/phase_3s.txt)

python ~/Matthew/code/scANTIPODE/examples/scripts/run_phase_3.py /allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Matthew/data/taxtest/HvQvM/HvQvMall_cere_clean_nodoublets.h5ad $NOTEBOOK