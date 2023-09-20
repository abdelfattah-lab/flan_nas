#!/bin/bash
#SBATCH -J exp8_nas_s_nb1_nb2_arch2vec
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=80000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 2
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/nas_search

python search.py --seed 42 --name_desc exp8 --source_space nb101 --target_space nb201 --gnn_type ensemble --back_dense --periter_samps 8 --samp_lim 512 --representation adj_gin_arch2vec --joint_repr --no_modify_emb_pretransfer
