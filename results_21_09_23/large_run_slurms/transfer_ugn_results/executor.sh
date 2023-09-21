#!/bin/bash

sbatch --requeue m0.slurm
sbatch --requeue m1.slurm
sbatch --requeue m2.slurm
sbatch --requeue m3.slurm
sbatch --requeue m4.slurm
sbatch --requeue m5.slurm
sbatch --requeue m6.slurm
sbatch --requeue m7.slurm

sbatch --requeue m0_1.slurm
sbatch --requeue m1_1.slurm
sbatch --requeue m2_1.slurm
sbatch --requeue m3_1.slurm
sbatch --requeue m4_1.slurm
sbatch --requeue m5_1.slurm
sbatch --requeue m6_1.slurm
sbatch --requeue m7_1.slurm
