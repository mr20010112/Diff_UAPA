import os
import minari
import shutil

# Set environment variable for Minari dataset path
os.environ['MINARI_DATA_PATH'] = '/home/mrq/project/diffusion_policy-main-test/data/kitchen/Minari'

# # Download the dataset
# dataset = minari.download_dataset("D4RL/kitchen/complete-v2")

# source = os.path.expanduser("~/.minari/datasets/D4RL/kitchen/complete-v2")
# destination = "/home/mrq/project/diffusion_policy-main-test/data/kitchen/Minari"

# # Move the dataset directory
# shutil.move(source, destination)

dataset = minari.load_dataset('/home/mrq/project/diffusion_policy-main-test/data/kitchen/Minari/complete-v2')
env  = dataset.recover_environment()

1+1 == 2