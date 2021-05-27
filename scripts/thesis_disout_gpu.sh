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

for i in 1 2 3 4 5 6 7 8
do
  # srun python train_baseline.py
  srun python train_disout.py
  # srun python train_data_aug.py
  # srun python train_disout_data_aug.py
done


# srun python train_baseline.py
# srun python train_disout.py
# srun python train_data_aug.py
# srun python train_disout_data_aug.py

