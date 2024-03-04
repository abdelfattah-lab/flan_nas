# FLAN: Encodings for Prediction-based Neural Architecture Search

This repository contains the implementation and necessary resources to reproduce the results presented in our paper "FLAN: Flow Attention for NAS." In this study, we investigate various neural network (NN) encoding methods to enhance the efficiency of Neural Architecture Search (NAS) accuracy predictors. We introduce a novel hybrid encoder, FLAN, and demonstrate its superior performance across multiple NAS benchmarks.

## Abstract

Predictor-based methods have significantly improved NAS optimization. The efficacy of these predictors hinges on the method of encoding neural network architectures. Traditional encodings utilized adjacency matrices, while recent encodings adopt diverse approaches including unsupervised pretraining and zero-cost proxies. We categorize neural encodings into structural, learned, and score-based. We introduce unified encodings, extending NAS predictors across multiple search spaces. Our analysis spans over 1.5 million architectures across various NAS benchmarks. FLAN leverages insights on predictor design, transfer learning, and unified encodings, substantially reducing the cost of training NAS accuracy predictors. Our encodings for all networks are open-sourced.

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
- The commands can be trivially adapted and run without SLURM as well.

## Example Executions

Below are specific example commands that demonstrate how to execute various processes within the framework. These examples cover training from scratch, utilizing supplementary encodings, transferring predictors between spaces, and running NAS on a given search space.

### Run Training from Scratch

To train a model from scratch using a specific seed, network representation, and a set of sample sizes:

```bash
python new_main.py --seed 42 --name_desc table1_s --gnn_type ensemble --sample_sizes 72 364 729 --batch_size 8 --space nb101 --representation adj_gin --test_size 7290 --num_trials 5
```

### Run Training from Scratch with Supplementary Encodings

To include supplementary encodings in your training:

```bash
python new_main.py --seed 42 --name_desc table1_s --gnn_type ensemble --sample_sizes 72 364 729 --batch_size 8 --space nb101 --representation adj_gin_a2vcatezcp --test_size 7290 --num_trials 5
```


### Transfer Predictor from a Source Space to a Target Space

For transferring a trained predictor to a different NAS space:

```bash
python universal_main.py --seed 42 --name_desc table1_tf --gnn_type ensemble --sample_size 512 --sourcetest_size 128 --transfer_sample_sizes 8 40 78 --batch_size 8 --space nb101 --transfer_space nb201 --representation adj_gin --joint_repr --test_size 7813 --num_trials 5
```

### Transfer Predictor with Supplementary Encodings from a Source Space to a Target Space

To transfer a predictor that includes supplementary encodings:

```bash
python universal_main.py --seed 42 --name_desc table1_tf --gnn_type ensemble --sample_size 512 --sourcetest_size 128 --transfer_sample_sizes 8 40 78 --batch_size 8 --space nb101 --transfer_space nb201 --representation adj_gin_a2vcatezcp --joint_repr --test_size 7813 --num_trials 5
```

### Run NAS on a Search Space

To conduct NAS within a specific search space:

```bash
python search.py --seed 42 --name_desc allnas --target_space nb101 --gnn_type ensemble --periter_samps 8 --samp_lim 512 --representation adj_gin_zcp --epochs 40
```

### Run NAS after Transfer Learning a Predictor on a Target Search Space

To run NAS using a predictor that has been transferred and fine-tuned on a new search space:

```bash
python search.py --seed 42 --name_desc allnas_t --source_space Amoeba --target_space PNAS_fix-w-d --gnn_type ensemble --periter_samps 8 --samp_lim 512 --representation adj_gin_arch2vec_cate --joint_repr --epochs 40
```

