#!/bin/bash
#SBATCH --job-name=KCAteam_cs4321_run
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=./logs_hamming/titans-out-%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

    source activate tfEnv


python trainer/task.py \
--model_dir="/data/cs4321/KCAteam/models/midterm_$(echo $USER)_$(date +%Y-%m-%d_%H-%M-%S)/" \
--model_type="MobileNetV2_frozen" \
--num_epochs=200 \
--batch_size=32 \
--eval_metrics="accuracy" \
--optimizer="adam" \
--callback_list="tensor_board, csv_log, checkpoint"




