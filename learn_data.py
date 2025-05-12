import h5py
import numpy as np
import os
from tqdm import tqdm

dataset_dir = 'data/realrobot/normal'
hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.hdf5', '.h5'))]


def h5_to_data(h5_obj):
    result = {}
    for key, item in h5_obj.items():
        if isinstance(item, h5py.Dataset):
            result[key] = item[()]
        elif isinstance(item, h5py.Group):
            result[key] = h5_to_data(item)
    return result

for filename in tqdm(hdf5_files, desc='Processing HDF5 files'):
    file_path = os.path.join(dataset_dir, filename)
    with h5py.File(file_path, 'r') as f:
        data = h5_to_data(f)
        image = data['observations']['images']['cam_left']
    if image.shape[0] != 50:
        print(f"Image shape mismatch in file {filename}: {image.shape}")