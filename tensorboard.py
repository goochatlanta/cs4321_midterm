
import subprocess
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default='')
parser.add_argument('--delete', type=str, default=None)
args = parser.parse_args()

list_of_files = glob.glob(f'/data/cs4321/KCAteam/models/*{args.user}*')
latest_file = max(list_of_files, key=os.path.getctime)

if args.delete:
    command = f'rm -fr {latest_file}'
else:
    command = f'tensorboard --logdir {latest_file}'
results = subprocess.call(command, shell=True)