#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=5:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --array=3981
#SBATCH --output=logs/3well_pchip_cg_%A_%a_out.txt
#SBATCH --error=logs/3well_pchip_cg_%A_%a_err.txt
#SBATCH --job-name=zf_cg

# python get_coarse_labels.py --DatasetName CC_multiparam/ -t 1 -cg 100 -dt 0.1 -ne 1
# python get_coarse_labels.py --DatasetName bacteria_all2/ -t 2 -cg 200 -dt 0.1 -ne 9
# python get_coarse_labels.py --DatasetName Zebrafish/ -t 3 -cg 200 -dt 1 -ne 7 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T10/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T17/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
# python get_coarse_labels.py --DatasetName 3well_pchip/T${SLURM_ARRAY_TASK_ID}/ -t 1 -cg 100 -dt 0.05 -ne 2 -s 1000
