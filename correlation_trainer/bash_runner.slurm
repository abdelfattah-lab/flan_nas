#!/bin/bash
#SBATCH -J help_tests
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/correlation_trainer/large_scale_run_logs/help_tests_%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=40000
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=abdelfattah
#SBATCH --exclude=abdelfattah-compute-01


source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate unr

cd /home/ya255/projects/iclr_nas_embedding/correlation_trainer

python new_main.py --space nb201 --representation adj_gin --test_tagates --loss_type pwl --sample_sizes 8 40 80 --batch_size 16 --test_size 7813
python new_main.py --space nb201 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 8 40 80 --batch_size 16 --test_size 7813
python new_main.py --space nb201 --representation adj_gin_cate --test_tagates --loss_type pwl --sample_sizes 8 40 80 --batch_size 16 --test_size 7813
python new_main.py --space nb201 --representation adj_gin_arch2vec --test_tagates --loss_type pwl --sample_sizes 8 40 80 --batch_size 16 --test_size 7813

python new_main.py --space nb101 --representation adj_gin --test_tagates --loss_type pwl --sample_sizes 73 364 729  --batch_size 16
python new_main.py --space nb101 --representation adj_gin_zcp --test_tagates --loss_type pwl --sample_sizes 73 364 729 --batch_size 16
python new_main.py --space nb101 --representation adj_gin_cate --test_tagates --loss_type pwl --sample_sizes 73 364 729 --batch_size 16
python new_main.py --space nb101 --representation adj_gin_arch2vec --test_tagates --loss_type pwl --sample_sizes 73 364 729 --batch_size 16

# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1 --residual_support
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2 --residual_support
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3 --residual_support
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1 --op_attention --residual_support
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2 --op_attention --residual_support
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3 --op_attention --residual_support
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1 --op_attention --residual_support --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2 --op_attention --residual_support --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3 --op_attention --residual_support --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1 --op_attention --residual_support --zcp
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2 --op_attention --residual_support --zcp
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3 --op_attention --residual_support --zcp
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1 --op_attention --residual_support --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2 --op_attention --residual_support --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3 --op_attention --residual_support --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1 --op_attention --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2 --op_attention --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3 --op_attention --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 1  --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 2  --zcp  --zs_as_p
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates --op_tagates --opmat_tagates --device cpu --op_emb --timestep 3  --zcp  --zs_as_p

# # bash tsne_visualizer.bash

# python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation zcp --test_tagates True

# python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjgin_zcp --test_tagates True

# python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjmlp_zcp --test_tagates True

# python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation zcp

# python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjgin_zcp

# python main.py --seed 42 --batch_size 32 --hidden_size 256 --num_layers 3 --epochs 200 --space nb101 --representation adjmlp_zcp

