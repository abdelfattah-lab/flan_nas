#!/bin/bash

# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --accs_tagates
python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --adj_tagates --accs_tagates
python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --op_tagates 
python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --opmat_tagates --accs_tagates
python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --opmat_tagates --opmat_only --accs_tagates
python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --op_tagates --opmat_tagates --accs_tagates
python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --adj_tagates --op_tagates --opmat_tagates --accs_tagates
python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --adj_tagates --op_tagates --zcp_tagates --accs_tagates
# python nb101_repl.py --space nb101 --representation adjgin_zcp --epochs 150 --loss_type pwl --test_tagates --adj_tagates --op_tagates --zcp_tagates --accs_tagates

# spaces=("Amoeba" "DARTS" "DARTS_fix-w-d" "DARTS_lr-wd" "ENAS" "ENAS_fix-w-d" "NASNet" "PNAS" "PNAS_fix-w-d" "nb101" "nb201" "nb301" "tb101")
# representations=("adj_mlp" "cate" "zcp" "arch2vec")

# for space in "${spaces[@]}"; do
#   for representation in "${representations[@]}"; do
#     echo "Running tsne_visualizer.py with --space $space and --representation $representation"
#     python tsne_visualizer.py --space "$space" --representation "$representation" --device "cpu"
#   done
# done

