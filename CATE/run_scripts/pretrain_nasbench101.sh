#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python run.py --do_train --parallel --train_data data/nasbench101/train_data.pt --train_pair data/nasbench101/train_pair_k2_d9999_metric_params.pt  --valid_data data/nasbench101/test_data.pt --valid_pair data/nasbench101/test_pair_k2_d9999_metric_params.pt --dataset nasbench101 --search_space nasbench101 --n_vocab 5 --graph_d_model 32 --pair_d_model 32
