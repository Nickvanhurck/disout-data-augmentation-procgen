#!/bin/bash

#SBATCH --partition=batch
#SBATCH --job-name=nvanhurck
#SBATCH --output=out/output-%A.txt
#SBATCH --error=out/error-%A.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nick.van.hurck@vub.be
#SBATCH --mem=16g
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1


module load Python/3.8.6-GCCcore-10.2.0

export OMP_NUM_THREADS=1

srun python test_bigfish.py
srun python test_coinrun.py
srun python test_heist.py
srun python test_starpilot.py
