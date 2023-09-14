#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH --mem=30000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah


export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

# adj_gin_arch2vec

python new_main.py --space ENAS_fix-w-d --sample_size 1024 2048 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space NASNet --sample_size 1024 2048 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space PNAS --sample_size 1024 2048 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space PNAS_fix-w-d --sample_size 1024 2048 --seed 42 --representation adj_gin_arch2vec


python new_main.py --space ENAS_fix-w-d --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space NASNet --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space PNAS --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space PNAS_fix-w-d --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
