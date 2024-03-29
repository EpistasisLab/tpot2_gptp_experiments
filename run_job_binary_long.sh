#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 10:00:00
#SBATCH --mem=1000G
#SBATCH --job-name=tpot2bench
#SBATCH -p moore,defq
#SBATCH --exclusive

#SBATCH -o ./logs/output.%j_%a.out # STDOUT
#SBATCH --array=1-10

source /common/ribeirop/minconda3/etc/profile.d/conda.sh
conda activate tpot2env

srun -u python run_tpot_on_openml_for_paper_binary_long.py \
--savepath results/results_binary \
--num_runs 5 \