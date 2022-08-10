#!/bin/bash
#SBATCH --job-name=KCAteam_cs4321_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate tfEnv

python trainer/task.py \
--model_dir="/data/cs4321/KCAteam/models/midterm_georgios.andrianopoulos.gr_2022-08-09_12-05-44-382227820/" \
--model_type="MobileNetV2" \
--num_epochs=61 \
--batch_size=32 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="tensor_board, csv_log, checkpoint" \
--data_augmentation="random_flip, MixUp" \
--num_fine_epochs=1 \
--unfrozen_layers=10 \
--length_of_dense_layers=256 \
#--only_fine_tuning='True'
#--con_fine_tunning='True'
#--model_dir="/data/cs4321/KCAteam/models/midterm_$(echo $USER)_$(date +%Y-%m-%d_%H-%M-%S-%N)/" \




