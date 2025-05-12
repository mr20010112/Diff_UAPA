import h5py
import cv2

# 指定 HDF5 文件路径
file_path = "data/realrobot/expert/1499.hdf5"

# 打开 HDF5 文件
with h5py.File(file_path, "r") as f:
    print(f"Exploring HDF5 file: {f}")
    data = f["observations"]["images"]["cam_right"]
    decompressed_image = cv2.imdecode(f["observations"]["images"]["cam_right"][0], 1)
    print(f"Decompressed image shape: {decompressed_image.shape}")

    
    # 如果需要深入递归访问
    # def print_structure(name, obj):
    #     if isinstance(obj, h5py.Group):
    #         print(f"Group: {name}")
    #     elif isinstance(obj, h5py.Dataset):
    #         print(f"  Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

    # f.visititems(print_structure)
