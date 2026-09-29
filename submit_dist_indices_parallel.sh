#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-100
#SBATCH --error=logs/3well_pchip_%A_%a_err.txt
#SBATCH --output=logs/3well_pchip_%A_%a_out.txt
#SBATCH --job-name=3well_pchip
 

# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn CC_multiparam/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn Zebrafish/ -t 3 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn bacteria_all2/ -t 2 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T10/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T17/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T31/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T56/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T100/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T158/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T251/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T398/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T630/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T1000/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T1584/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T2511/ -t 1 -nboot 100
python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T3981/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T6309/ -t 1 -nboot 100
# python markov_distance_indices_parallel.py -idx ${SLURM_ARRAY_TASK_ID} -dn 3well_pchip/T10000/ -t 1 -nboot 100