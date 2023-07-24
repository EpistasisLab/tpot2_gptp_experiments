#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 5:00:00
#SBATCH --mem=0
#SBATCH --job-name=tpot2bench
#SBATCH -p moore,defq
#SBATCH --exclusive

#SBATCH -o ./logs/output.%j_%a.out # STDOUT
#SBATCH --array=1-34

source /common/ribeirop/minconda3/etc/profile.d/conda.sh
conda activate tpot2env

srun -u python run_tpot_on_openml_for_paper_multiclass.py \
--savepath results/results_multi \
--num_runs 5 \