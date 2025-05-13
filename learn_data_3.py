import h5py
import cv2
import os
import concurrent.futures

# 指定 HDF5 文件路径
file_path = "data/realrobot/expert/1499.hdf5"

def decode_image(data):
    return cv2.imdecode(data, 1)

with h5py.File(file_path, "r") as f:
    data = f["observations"]["images"]["cam_right"]
    # decompressed_image = cv2.imdecode(data[0], 1)
    # print(f"Decompressed image shape: {decompressed_image.shape}")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(decode_image, data)
        decompressed_images = list(results)
                               
    decompressed_image = decompressed_images[21]

    # 保存到当前工作目录下
    output_path = "cam_right.png"
    success = cv2.imwrite(output_path, decompressed_image)
    if success:
        print(f"Image saved to {os.path.abspath(output_path)}")
    else:
        print("Failed to save image")
    
    # 如果需要深入递归访问
    # def print_structure(name, obj):
    #     if isinstance(obj, h5py.Group):
    #         print(f"Group: {name}")
    #     elif isinstance(obj, h5py.Dataset):
    #         print(f"  Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

    # f.visititems(print_structure)
