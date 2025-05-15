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
srun ./bin/gpu/gpu_csr ./input_files/pattern/can_268.mtx
