import torch
import dill
import copy
import os
from diffusion_policy.policy.ours_diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy

checkpoint_path = 'data/experiments/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt'

model: DiffusionTransformerLowdimPolicy
checkpoint = torch.load(checkpoint_path, pickle_module=dill)

keys = checkpoint['state_dicts']['model']['model'].keys()

for key in keys:
    #origin_key = key.replace('ref_model.', 'model.', 1)
    #print(origin_key)
    flag = checkpoint['state_dicts']['model']['model'][key]
    checkpoint['state_dicts']['model']['ref_model'][key] = flag

checkpoint_path = checkpoint_path.replace('latest.ckpt', 'latest_rlhf.ckpt')

torch.save(checkpoint, checkpoint_path, pickle_module=dill)