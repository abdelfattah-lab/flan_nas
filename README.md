# FLAN: Encodings for Prediction-based Neural Architecture Search

This repository contains the implementation and necessary resources to reproduce the results presented in our paper "FLAN: Flow Attention for NAS." In this study, we investigate various neural network (NN) encoding methods to enhance the efficiency of Neural Architecture Search (NAS) accuracy predictors. We introduce a novel hybrid encoder, FLAN, and demonstrate its superior performance across multiple NAS benchmarks.

## Abstract

Predictor-based methods have significantly improved NAS optimization. The efficacy of these predictors hinges on the method of encoding neural network architectures. Traditional encodings utilized adjacency matrices, while recent encodings adopt diverse approaches including unsupervised pretraining and zero-cost proxies. We categorize neural encodings into structural, learned, and score-based. We introduce \textit{unified encodings}, extending NAS predictors across multiple search spaces. Our analysis spans over 1.5 million architectures across various NAS benchmarks. FLAN leverages insights on predictor design, transfer learning, and unified encodings, substantially reducing the cost of training NAS accuracy predictors. Our encodings for all networks are open-sourced.

## Reproducing Results

To replicate the findings of our paper, follow the setup and execution guidelines below:

### Environment Setup

Ensure that the `env_setup.py` script is executed correctly to set up the environment. Please modify the file to the appropriate path.

### Dataset Preparation

- Download the NDS dataset from [this link](https://dl.fbaipublicfiles.com/nds/data.zip) and place it in the `nas_embedding_suite` folder. The expected structure is `NDS/nds_data/*.json`.
- Download and unzip `flan_embeddings_04_03_24.zip` to `./nas_embedding_suite/` from [Google Drive](https://drive.google.com/file/d/1oJyH0zox_cbRUX-hgzkliOLAUaz3gIxw/view?usp=sharing).

### Execution Instructions

- Reference commands for training and testing the predictors are located in `./correlation_trainer/large_run_slurms/unified_joblist.log` and `./correlation_trainer/large_run_slurms/unified_nas_joblist.log`.
- For executing the processes on a local SLURM setup, refer to the `parallelized_executor.sh` script and make the appropriate modifications for your set-up.
- The commands can be adapted and run without SLURM as well.
