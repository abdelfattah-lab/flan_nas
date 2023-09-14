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
python universal_main.py --space DARTS --transfer_space ENAS --representation adj_gin_zcp --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space PNAS_fix-w-d --transfer_space DARTS_fix-w-d --representation adj_gin_zcp --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
python universal_main.py --space NASNet --transfer_space ENAS_fix-w-d --representation adj_gin_zcp --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4
