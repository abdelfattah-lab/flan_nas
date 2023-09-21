#!/bin/bash
#SBATCH -J exptag_enct_enas_0
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=30000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 2
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

python new_main.py --seed 42 --name_desc exp_tagates_enas --gnn_type ensemble --back_dense --sample_sizes 25 50 125 --batch_size 5 --test_size 500 --space ENAS --representation "adj_gin"
