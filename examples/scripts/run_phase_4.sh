#!/bin/bash
#SBATCH --job-name phase4
#SBATCH --output /home/matthew.schmitz/log/phase3_%A_%a.out # %A is replaced by job ID, %a by array index
#SBATCH --error /home/matthew.schmitz/log/phase3_%A_%a.err
#SBATCH --time 200:00:00
#SBATCH --partition celltypes
#SBATCH --array=2-2 # Create an array job for each line in notebooks.txt
#SBATCH --gres=gpu:1 --constraint="a100|v100"#
#SBATCH --mem 128gb
#SBATCH --ntasks 1

source ~/.bashrc

conda activate antipode

# Read the notebook path for the current array task
NOTEBOOK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ~/Matthew/code/scANTIPODE/examples/scripts/phase_4s.txt)

python ~/Matthew/code/scANTIPODE/examples/scripts/run_phase_4.py /allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Matthew/data/taxtest/HvQvM/HvQvMall_cere_clean_nodoublets.h5ad $NOTEBOOK batch_name species spliced

#python ~/Matthew/code/scANTIPODE/examples/scripts/run_phase_4.py /allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Matthew/data/cortex_data/jorstad_merge_all.h5ad $NOTEBOOK batch organism UMIs

# python ~/Matthew/code/scANTIPODE/examples/scripts/run_phase_4.py /home/matthew.schmitz/Matthew/data/cortex_data/v1_combination.h5ad $NOTEBOOK batch species UMIs

