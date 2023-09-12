#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=40000
#SBATCH -t 4:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah

export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

python new_main.py --space nb201 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 8 40 80 --batch_size 16 --test_size 7813

python new_main.py --space nb201 --representation arch2vec --test_tagates --loss_type pwl --sample_sizes 8 40 80 --batch_size 16 --test_size 7813
