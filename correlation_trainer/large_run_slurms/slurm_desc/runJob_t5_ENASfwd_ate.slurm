#!/bin/bash
#SBATCH -J t5_ENASfwd_ate
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=80000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 4
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

python universal_main.py --seed 42 --name_desc table4 --gnn_type ensemble --sample_size 512 --sourcetest_size 128 --transfer_sample_sizes 1 2 3 4 5 6 7 8 16 32 64 128 --batch_size 8 --space NASNet --transfer_space ENAS_fix-w-d --representation adj_gin_a2vcatezcp --joint_repr --num_trials 7
