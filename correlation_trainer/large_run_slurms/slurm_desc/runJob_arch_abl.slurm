#!/bin/bash
#SBATCH -J arch_abl
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/%j.err
#SBATCH -N 1
#SBATCH --mem=30000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --gres=gpu:1
#SBATCH -n 4
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

python main_abl.py --seed 42 --name_desc arch_abl --gnn_type ensemble --sample_sizes 8 16 32 --batch_size 8  --representation adj_gin  --space nb201 --timesteps 3  --no_residual --no_leakyrelu --no_unique_attention_projection --test_size 1000
