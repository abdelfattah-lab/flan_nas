#!/bin/bash
#SBATCH -J allnasspace
#SBATCH -o /home/ya255/projects/iclr_nas_embedding/CATE/large_scale_run_logs/large_allss_%j.out
#SBATCH -e /home/ya255/projects/iclr_nas_embedding/CATE/large_scale_run_logs/large_allss_%j.err
#SBATCH -N 1
#SBATCH --mem=80000
#SBATCH -t 48:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-01
#SBATCH --gres=gpu:1
#SBATCH -n 4

export PROJ_BPATH="/home/ya255/projects/iclr_nas_embedding"

source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate cate

cd /home/ya255/projects/iclr_nas_embedding/CATE

$1