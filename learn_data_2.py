import torch

# 加载模型的ckpt文件
checkpoint = torch.load('data/outputs/2025.05.12/18.44.09_train_diffusion_real_robot0_real_robot/checkpoints/epoch=0015-train_action_mse_error=0.610.ckpt')

# 查看包含的键值
for key in checkpoint.keys():
    print(f"Key: {key}, Type: {type(checkpoint[key])}, Value: {checkpoint[key]}")

