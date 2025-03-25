import h5py

# 指定 HDF5 文件路径
file_path = "data/robomimic/datasets/lift/ph/image.hdf5"

# 打开 HDF5 文件
with h5py.File(file_path, "r") as h5_file:
    print(f"Exploring HDF5 file: {file_path}")
    
    # 遍历文件结构
    for name, obj in h5_file.items():
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
    
    # 如果需要深入递归访问
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

    h5_file.visititems(print_structure)
