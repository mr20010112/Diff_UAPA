import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import copy
import numpy as np
import math
from typing import Optional, Dict

from diffusion_policy.common.prior_utils_confidence import BetaNetwork
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

def comp_policy(self, ref_policy: BaseLowdimPolicy, target_policy: BaseLowdimPolicy, env: BaseLowdimRunner, beta_model: BetaNetwork):
    ref_runner_log, ref_episode_data = env.run(ref_policy)
    target_runner_log, target_episode_data = env.run(target_policy)

    ref_obs = ref_episode_data['observations']
    ref_action = ref_episode_data['actions']

    target_obs = target_episode_data['observations']
    target_action = target_episode_data['actions']

    ref_s_a = np.concatenate([ref_obs, ref_action], axis=-1)
    target_s_a = np.concatenate([target_obs, target_action], axis=-1)

    def get_beta_value(self, beta_model: BetaNetwork, s_a: np.ndarray, batch_size=5):
        interval = math.ceil(s_a.shape[0] / batch_size)
        alpha, beta = [], []
        for i in range(interval):
            start_pt = i * batch_size
            end_pt = min((i + 1) * batch_size, s_a.shape[0])
            batch_s_a = s_a[start_pt:end_pt, ...]

            batch_alpha, batch_beta = beta_model.get_alpha_beta(torch.from_numpy(batch_s_a).float().to(beta_model.device))

            alpha.append(batch_alpha)
            beta.append(batch_beta)

        return np.array(torch.cat(alpha, dim=0).cpu().numpy()), np.array(torch.cat(beta, dim=0).cpu().numpy())
    
    ref_alpha, ref_beta = get_beta_value(self, beta_model = beta_model, s_a = ref_s_a)
    target_alpha, target_beta = get_beta_value(self, beta_model = beta_model, s_a = target_s_a)

    ref_score, target_score = np.sum(ref_alpha-ref_beta), np.sum(target_alpha-target_beta)

    if ref_score >= target_score:
        print('contain reference policy')
        return ref_policy
    else:
        print('update reference policy')
        return copy.deepcopy(target_policy)