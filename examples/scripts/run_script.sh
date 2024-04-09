#!/bin/bash
#SBATCH --job-name antipode_script
#SBATCH --output /home/matthew.schmitz/log/antipode_script_%A.out # %A is replaced by job ID, %a by array index
#SBATCH --error /home/matthew.schmitz/log/antipode_script_%A.err
#SBATCH --time 72:00:00
#SBATCH --partition celltypes
#SBATCH --gres=gpu:1 --constraint="a100|v100"
#SBATCH --mem 128gb
#SBATCH --ntasks 1

source ~/.bashrc

!nvidia-smi

conda activate pyro

# Command to run jupyter nbconvert
#jupyter nbconvert --to script ~/Matthew/code/scANTIPODE/examples/RunDev-CleanedFixMemLeak-Longer-cere-NoDN.ipynb

python ~/Matthew/code/scANTIPODE/examples/RunDev-CleanedFixMemLeak-Longer-cere-NoDN.py