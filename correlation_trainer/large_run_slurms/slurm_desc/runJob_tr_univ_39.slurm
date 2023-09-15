#!/bin/bash
#SBATCH -J tr_univ_39
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=80000
#SBATCH -t 48:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 2
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

python universal_main.py --space DARTS --transfer_space ENAS --representation adj_gin_arch2vec --loss_type pwl --sample_size 2048 --batch_size 256 --transfer_sample_sizes 4 8 16 32 --batch_size 8 --transfer_epochs 30 --transfer_lr 3e-4 --joint_repr
