#!usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python run.py --do_train --parallel --train_data data/all_ss/train_data.pt --train_pair data/all_ss/train_pair_k2_d50000_metric_params_0_1413343.pt --valid_data data/all_ss/test_data.pt --valid_pair data/all_ss/test_pair_k2_d50000_metric_params_0_1413343.pt --dataset all_ss --search_space all_ss --n_vocab 20 --graph_d_model 32 --pair_d_model 32 --check_point_freq 2760