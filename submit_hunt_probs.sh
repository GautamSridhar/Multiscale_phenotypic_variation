#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-16
#SBATCH --error=logs/zf_hunts_%A_%a_err.txt
#SBATCH --output=logs/zf_hunts_%A_%a_out.txt
#SBATCH --job-name=zf_hunts

python estimate_hunt_probs.py -dn Zebrafish -idx ${SLURM_ARRAY_TASK_ID} -len_sim 5000 -n_bouts 3