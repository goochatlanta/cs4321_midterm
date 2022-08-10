#!/bin/bash
#SBATCH --job-name=KCAteam_cs4321_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate tfEnv


python trainer/task.py \
--model_dir="/data/cs4321/KCAteam/models/midterm_$(echo $USER)_$(date +%Y-%m-%d_%H-%M-%S-%N)/" \
--model_type="resnet50" \
--num_epochs=10 \
--batch_size=32 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="tensor_board, csv_log, checkpoint" \
--data_augmentation="random_flip, MixUp, random_augmentation" \
--num_fine_epochs=6 \
--unfrozen_layers=100 \
--length_of_dense_layers=256 \
#--only_test_model_dir="/data/cs4321/KCAteam/models/midterm_georgios.andrianopoulos.gr_2022-08-07_03-29-09/"





