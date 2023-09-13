#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=30000
#SBATCH -t 16:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah

export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

# adj_gin_arch2vec
python new_main.py --space tb101 --task autoencoder --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space tb101 --task class_object --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space tb101 --task class_scene --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space tb101 --task jigsaw --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space tb101 --task room_layout --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec
python new_main.py --space tb101 --task segmentsemantic --sample_size 16 32 128 256 512 --seed 42 --representation adj_gin_arch2vec

# python new_main.py --space nb101 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 73 364 729 --batch_size 16

# python new_main.py --space nb101 --representation zcp --test_tagates --loss_type pwl --sample_sizes 73 364 729 --batch_size 16
