#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah


export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer


python new_main.py --space nb101 --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space nb201 --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison
python new_main.py --space DARTS --representation adj_gin_zcp --loss_type pwl --sample_sizes 4 8 16 32 --batch_size 8 --transfer_comparison