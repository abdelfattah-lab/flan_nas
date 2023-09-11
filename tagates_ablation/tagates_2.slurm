#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o large_scale_run_logs/help_tests_%j.out
#SBATCH -e large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00
#SBATCH --account=abdelfattah


source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer
export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1 --op_attention --residual_support --zcp
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2 --op_attention --residual_support --zcp
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3 --op_attention --residual_support --zcp


python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb \
        --timestep 1 --residual_support --op_attention 

python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb \
        --timestep 1 --residual_support --op_attention --zcp
