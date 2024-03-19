#!/bin/bash

#SBATCH -A roysam
#SBATCH -J DDPM_shallow_infer
#SBATCH -t 1-00:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=5gb
#SBATCH --array=1-75
module load torchvision/0.14.0-foss-2021b-CUDA-11.4.1 

python main2.py --num_channels=64 --num_ensemble=3 --subsetID="$SLURM_ARRAY_TASK_ID" --dsdir_GT=SKOV3_CARTCD28_raw_RN_1hr_for_deblur.txt