import h5py
import numpy as np

# 定义文件路径
file_paths = [
    "data/kitchen/Minari/normal/epoch=25/kitchen_data_0.5.h5",
    "data/kitchen/Minari/normal/epoch=25/kitchen_data_0.5_2.h5",
    "data/kitchen/Minari/normal/epoch=25/kitchen_data_0.5_3.h5",
]

# 初始化一个字典，用于存储堆叠后的数据
stacked_data = {}

# 遍历文件并堆叠数据
for file_path in file_paths:
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            # 如果是第一次处理这个键，直接初始化
            if key not in stacked_data:
                stacked_data[key] = f[key][()]
            else:
                # # 堆叠当前文件数据到已有数据中
                # if key == 'terminals':
                #     stacked_data[key] = np.concatenate((stacked_data[key], f[key][()] + stacked_data[key][-1]))
                # else:    
                stacked_data[key] = np.concatenate((stacked_data[key], f[key][()]))

# 保存堆叠后的数据为一个新的 HDF5 文件
output_path = "data/kitchen/Minari/normal/epoch=25/kitchen_data_0.5_complete.h5"
with h5py.File(output_path, 'w') as f:
    for key, value in stacked_data.items():
        f.create_dataset(key, data=value)

print(f"Stacked dataset saved to {output_path}")
