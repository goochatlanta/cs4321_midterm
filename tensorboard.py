import subprocess
import glob
import os

list_of_files = glob.glob('/data/cs4321/KCAteam/models/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
command = f'tensorboard --logdir {latest_file}'
results = subprocess.call(command, shell=True)