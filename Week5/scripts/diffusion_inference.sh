#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-24:00 # Runtime in D-HH:MM
#SBATCH -p mlow # Partition to submit to
#SBATCH -q masterlow # Required to requeue other users mlow queue jobs
                      # With this parameter only 1 job will be running in queue mhigh
                      # By defaulf the value is masterlow if not defined
#SBATCH --mem 4096 # 4GB memory
#SBATCH --gres gpu:1 # Request of 1 gpu
#SBATCH -o logs/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e logs/%x_%u_%j.err # File to which STDERR will be written


python -m src.main --config configs/diffusion/compare_models_sd_5steps.yaml diffusion_inference