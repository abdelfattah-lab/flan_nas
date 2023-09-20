#!/bin/bash
#SBATCH -J exp7_nas_s_nb2_nb1_cate
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=80000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 2
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/nas_search

python search.py --seed 42 --name_desc exp7 --source_space nb201 --target_space nb101 --gnn_type ensemble --back_dense --periter_samps 8 --samp_lim 512 --representation adj_gin_cate --no_modify_emb_pretransfer
