#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 16:40:00
#SBATCH --mem=0
#SBATCH --job-name=tpot2bench
#SBATCH -p moore,defq
#SBATCH --exclusive

#SBATCH -o ./logs/output.%j_%a.out # STDOUT
#SBATCH --array=1-42

source /common/ribeirop/minconda3/etc/profile.d/conda.sh
conda activate tpot2env

srun -u python run_tpot_on_openml_for_paper_binary.py \
--n_jobs 48 \
--savepath results/results_binary \
--num_runs 5 \