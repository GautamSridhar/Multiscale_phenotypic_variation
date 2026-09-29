#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0,1,2
#SBATCH --error=logs/bact_bs_detect_%A_%a_err.txt
#SBATCH --output=logs/bact_bs_detect_%A_%a_out.txt
#SBATCH --job-name=bact_bs_detect

python bootstrap_detect_probs.py -dn bacteria_all2 -idx ${SLURM_ARRAY_TASK_ID} -len_sim 5000 -n_bouts 1000