#!/bin/bash
#SBATCH --job-name=KCAteam_cs4321_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate tfEnv

python trainer/report.py \
--model_dir="/data/cs4321/KCAteam/models/midterm_georgios.andrianopoulos.gr_2022-08-10_23-49-36-598234480/" \
--tsne_ds="True" \
--model_type="MobileNetV2"