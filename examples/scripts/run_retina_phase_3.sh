#!/bin/bash
#SBATCH --job-name phase3
#SBATCH --output /home/matthew.schmitz/log/phase3_%A_%a.out # %A is replaced by job ID, %a by array index
#SBATCH --error /home/matthew.schmitz/log/phase3_%A_%a.err
#SBATCH --time 72:00:00
#SBATCH --partition celltypes
#SBATCH --gres=gpu:1 --constraint="a100|v100"#
#SBATCH --mem 128gb
#SBATCH --ntasks 1

source ~/.bashrc

conda activate antipode

# Read the notebook path for the current array task
NOTEBOOK=/allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Matthew/models/1.9.1.8.1_RunRetinaAllNoPsi

python ~/Matthew/code/scANTIPODE/examples/scripts/run_phase_3.py /allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Matthew/data/shekar_retina/retina_filtered.h5ad $NOTEBOOK

