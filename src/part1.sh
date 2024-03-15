#!/bin/bash

#SBATCH -A varadarajan
#SBATCH -J debug
#SBATCH -o /project/varadarajan/kwu14/repo/blurry_classification/20230505_COAT.o%j.txt
#SBATCH -t 0-12:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=5gb
#SBATCH --mail-user=kwu14@cougarnet.uh.edu
#SBATCH --mail-type=all


module add torchvision/0.15.2-foss-2022a-CUDA-11.7.0

python  main1.py --model=COAT --mode=TIMING_stack --weights=COAT_testweights.pth --TIMING_dir=/project/varadarajan/kwu14/DT-HPC/SKOV3_CARTCD28/ --prefix=SKOV3_CARTCD28_raw_RN_1hr --TIMING_subset=1 --size=224 --first_hr=1