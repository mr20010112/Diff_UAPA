import pickle
import numpy as np

# 加载 pkl 文件
file_path = 'data/d4rl/halfcheetah-medium-expert-v2.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# # 查看数据结构
# print(type(data))  # 数据的顶层结构
# print(len(data))   # 数据的长度（如果是列表或字典）

for key, value in data.items():
    print(f"Key: {key}, Type: {type(value)}")
    if isinstance(value, (list, dict)):
        print(f"Length: {len(value)}")
    elif isinstance(value, np.ndarray):
        print(f"Shape: {value.shape}")
