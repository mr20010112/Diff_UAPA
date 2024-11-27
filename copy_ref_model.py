import torch
import dill
import copy
import os
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy

# 假设 checkpoint 文件路径
checkpoint_path = 'data/experiments/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt'

model: DiffusionTransformerLowdimPolicy
# 加载 checkpoint
checkpoint = torch.load(checkpoint_path, pickle_module=dill)

keys = checkpoint['state_dicts']['model']['model'].keys()

for key in keys:
    #origin_key = key.replace('ref_model.', 'model.', 1)
    #print(origin_key)
    flag = checkpoint['state_dicts']['model']['model'][key]
    checkpoint['state_dicts']['model']['ref_model'][key] = flag

checkpoint_path = checkpoint_path.replace('latest.ckpt', 'latest_rlhf.ckpt')

# 保存回 checkpoint 文件
torch.save(checkpoint, checkpoint_path, pickle_module=dill)