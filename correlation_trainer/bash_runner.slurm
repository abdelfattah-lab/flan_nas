#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o large_scale_run_logs/help_tests_%j.out
#SBATCH -e large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=40000
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=abdelfattah
#SBATCH --exclude=abdelfattah-compute-01


source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/nas_embedding_suite/correlation_trainer

# bash tsne_visualizer.bash

python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation zcp --test_tagates True

python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjgin_zcp --test_tagates True

python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjmlp_zcp --test_tagates True

python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation zcp

python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjgin_zcp

python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjmlp_zcp

