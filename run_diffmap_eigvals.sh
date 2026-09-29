#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-2000
#SBATCH --output=logs/bact_%A_%a_out.txt
#SBATCH --error=logs/bact_%A_%a_err.txt
#SBATCH --job-name=bact

python calc_dmap_eigvals.py --DatasetName bacteria_all2/ -idx ${SLURM_ARRAY_TASK_ID}