#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=output_run/test-%j.out
#SBATCH --error=output_run/test-%j.err

module load CUDA/12.1.1 
# srun ncu -o profile ./bin/gpu/gpu_csr 2 ./input_files/sls/sls.mtx
srun ./bin/gpu/gpu_csr 5 ./input_files/integer/Trec5.mtx
