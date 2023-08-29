#!/bin/bash

spaces=("Amoeba" "DARTS" "DARTS_fix-w-d" "DARTS_lr-wd" "ENAS" "ENAS_fix-w-d" "NASNet" "PNAS" "PNAS_fix-w-d" "nb101" "nb201" "nb301" "tb101")
representations=("adj_mlp" "cate" "zcp" "arch2vec")

for space in "${spaces[@]}"; do
  for representation in "${representations[@]}"; do
    echo "Running tsne_visualizer.py with --space $space and --representation $representation"
    python tsne_visualizer.py --space "$space" --representation "$representation" --device "cpu"
  done
done

