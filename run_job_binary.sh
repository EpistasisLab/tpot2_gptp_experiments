#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 5:00:00
#SBATCH --mem=0
#SBATCH --job-name=tpot2bench
#SBATCH -p preemptable
#SBATCH --exclusive
#SBATCH --exclude "esplhpc-cp040"
#SBATCH -o ./logs/output.%j_%a.out # STDOUT
#SBATCH --array=1-200


conda activate tpot2env

srun -u python run_tpot_on_openml_for_paper_binary.py \
--n_jobs 48 \
--savepath results/results_binary \
--num_runs 5 \