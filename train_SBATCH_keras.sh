#!/bin/bash
#SBATCH --job-name=KCAteam_cs4321_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate tfEnv

python trainer/task.py \
--model_dir="/data/cs4321/KCAteam/models/midterm_$(echo $USER)_$(date +%Y-%m-%d_%H-%M-%S-%N)/" \
--model_type="MobileNetV2" \
--num_epochs=100 \
--batch_size=16 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="tensor_board, csv_log, checkpoint" \
--data_augmentation="random_flip, MixUp" \
--num_fine_epochs=100 \
--unfrozen_layers=10 \
--length_of_dense_layers="256" \
#--con_fine_tunning='True'
#--only_fine_tuning='True'
#--model_dir="/data/cs4321/KCAteam/models/midterm_georgios.andrianopoulos.gr_2022-08-10_23-59-07-634273952/" \




