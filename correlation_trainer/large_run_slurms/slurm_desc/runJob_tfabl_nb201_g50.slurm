#!/bin/bash
#SBATCH -J tfabl_nb201_g50
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=80000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 4
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

python universal_main.py --seed 42 --name_desc epochs_ablation --gnn_type ensemble --sample_size 256 --transfer_sample_sizes 4 16 --batch_size 8 --space nb101 --test_size 500 --sourcetest_size 128 --transfer_space nb201 --representation adj_gin --joint_repr --transfer_epochs 50 --transfer_lr 0.001
