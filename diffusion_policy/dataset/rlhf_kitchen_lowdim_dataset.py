from typing import Dict
import torch
import numpy as np
import copy
import pathlib
import random
import hydra
import h5py
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs
from diffusion_policy.common.pref_replay_buffer import PrefReplayBuffer
from diffusion_policy.common.pref_sampler import PrefSequenceSampler
from typing import Optional, Dict
from diffusion_policy.common.prior_utils import BetaNetwork

#处理 "kitchen" 任务的低维数据集。该类从 .mjl 文件中解析数据，存储在 ReplayBuffer 中，并对数据进行采样

class RLHF_KitchenLowdimDataset(BaseLowdimDataset):
    def __init__(self,
                expert_replay_buffer: ReplayBuffer,
                normal_replay_buffer: ReplayBuffer,
                abs_action=True,
                sequence_length=1,
                gamma=0.9999,
                N=1,
                load_dir=None,
                save_dir=None,
                ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        expert_episode_ends = expert_replay_buffer.episode_ends
        expert_num_episodes = len(expert_episode_ends)
        normal_episode_ends = normal_replay_buffer.episode_ends
        normal_num_episodes = len(normal_episode_ends)
        self.pref_replay_buffer = PrefReplayBuffer.create_empty_numpy()


        if load_dir is None:
            for _ in range(N):
                idx_expert = random.sample(range(expert_num_episodes), 1)[0]
                idx_normal = random.sample(range(normal_num_episodes), 1)[0]

                episode_expert = expert_replay_buffer.get_episode(idx_expert, keys=['obs', 'action', 'reward'], copy=False)
                episode_normal = normal_replay_buffer.get_episode(idx_normal, keys=['obs', 'action', 'reward'], copy=False)

                # Equal length processing for episode 1
                expert_len = len(episode_expert['obs'])
                if expert_len >= sequence_length:
                    start = 0
                    length = sequence_length
                    for key in episode_expert.keys():
                        episode_expert[key] = episode_expert[key][start:start + sequence_length]
                else:
                    for key in episode_expert.keys():
                        episode_expert[key] = np.pad(episode_expert[key], 
                                            ((0, sequence_length - expert_len),) + ((0, 0),) * (episode_expert[key].ndim - 1),
                                            mode='edge')
                        length = expert_len

                # Equal length processing for episode 2
                normal_len = len(episode_normal['obs'])
                if normal_len >= sequence_length:
                    start = 0
                    length_2 = sequence_length
                    for key in episode_normal.keys():
                        episode_normal[key] = episode_normal[key][start:start + sequence_length]
                else:
                    for key in episode_normal.keys():
                        episode_normal[key] = np.pad(episode_normal[key], 
                                            ((0, sequence_length - normal_len),) + ((0, 0),) * (episode_normal[key].ndim - 1),
                                            mode='edge')
                        length_2 = normal_len


                # Set up votes and metadata based on the presence of 'reward' in episode1
                if 'reward' not in episode_expert.keys() or len(episode_expert['reward']) == 0:
                    votes = np.zeros((1,))
                    votes_2 = np.zeros((1,))
                else:
                    flag = np.sum([gamma ** t * reward for t, reward in enumerate(episode_expert['reward'])])
                    flag_2 = np.sum([gamma ** t * reward for t, reward in enumerate(episode_normal['reward'])])
                    if np.abs(flag - flag_2) < 0.1:
                        votes = np.array([0.5])
                        votes_2 = np.array([0.5])
                    elif flag > flag_2:
                        votes = np.array([1.0])
                        votes_2 = np.array([0.0])
                    else:
                        votes = np.array([0.0])
                        votes_2 = np.array([1.0])

                # votes = np.ones((1,))
                # votes_2 = np.zeros((1,))


                # Add preference episode to the replay buffer
                self.pref_replay_buffer.add_pref_episode(
                    data={
                        'obs': episode_expert['obs'],          # First trajectory observations (shape T, obs_dim)
                        'action': episode_expert['action'],     # First trajectory actions (shape T, action_dim)
                        'obs_2': episode_normal['obs'],         # Second trajectory observations
                        'action_2': episode_normal['action'],    # Second trajectory actions
                    },
                    meta_data={
                        'votes': votes,                   # Vote for the first trajectory
                        'votes_2': votes_2,               # Vote for the second trajectory
                        'length': np.array([length]),     # Length of the first trajectory
                        'length_2': np.array([length_2]), # Length of the second trajectory
                        'beta_priori': np.zeros([2]),
                        'beta_priori_2': np.zeros([2]),
                    }
                )

            if save_dir is not None:
                save_data = dict()
                data, meta = self.pref_replay_buffer.data, self.pref_replay_buffer.meta
                save_data.update({key: value for key, value in data.items()})
                save_data.update({key: value for key, value in meta.items()})

                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                def recursively_store(group, data):

                    for key, value in data.items():
                        # 确保键名合法（HDF5 不支持某些特殊字符）
                        sanitized_key = key.replace('/', '_')  # 替换不合法字符

                        if isinstance(value, dict):
                            # 如果值是字典，则创建子组并递归存储
                            subgroup = group.create_group(sanitized_key)
                            recursively_store(subgroup, value)
                        elif isinstance(value, str):
                            # 如果值是字符串，保存为字节类型
                            group.create_dataset(sanitized_key, data=value.encode('utf-8'))
                        elif hasattr(value, "dtype") and value.dtype == 'O':
                            # 如果是对象类型数组，转为字符串数组
                            group.create_dataset(sanitized_key, data=[str(v) for v in value])
                        else:
                            # 其他情况，直接保存
                            group.create_dataset(sanitized_key, data=value)

                with h5py.File(save_dir / 'kitchen_prefdata.h5', 'w') as f:
                    recursively_store(f, save_data)
        else:
            with h5py.File(load_dir, 'r') as f:
                pref_data = f
        
                observation = pref_data['obs'][:]
                action = pref_data['action'][:]
                observation_2 = pref_data['obs_2'][:]
                action_2 = pref_data['action_2'][:]
                votes = pref_data['votes'][:]
                votes_2 = pref_data['votes_2'][:]
                length = pref_data['length'][:]
                length_2 = pref_data['length_2'][:]
                beta_priori = pref_data['beta_priori'][:]
                beta_priori_2 = pref_data['beta_priori_2'][:]

            N = observation.shape[0]
            sequence_length = observation.shape[1]
            
            for i in range(N):
                self.pref_replay_buffer.add_pref_episode(
                    data={
                        'obs': observation[i],         # First trajectory observations
                        'action': action[i],           # First trajectory actions (shape T, action_dim)
                        'obs_2': observation_2[i],     # Second trajectory observations
                        'action_2': action_2[i],       # Second trajectory actions
                    },
                    meta_data={
                        'votes': votes[i],                   # Vote for the first trajectory
                        'votes_2': votes_2[i],               # Vote for the second trajectory
                        'length': length[i],                 # Length of the first trajectory
                        'length_2': length_2[i],             # Length of the second trajectory
                        'beta_priori': beta_priori[i],      # Beta priori for the first trajectory
                        'beta_priori_2': beta_priori_2[i],  # Beta priori for the second trajectory
                    }
                )


        #self.data = pref_dataset
        self.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer,
            sequence_length=sequence_length,
        )

        self.length = N


    # def get_validation_dataset(self):
    #     val_set = copy.copy(self)
    #     val_set.sampler = SequenceSampler(
    #         replay_buffer=self.replay_buffer, 
    #         sequence_length=self.horizon,
    #         pad_before=self.pad_before, 
    #         pad_after=self.pad_after,
    #         episode_mask=~self.train_mask
    #         )
    #     val_set.train_mask = ~self.train_mask
    #     return val_set

    # def get_normalizer(self, mode='limits', **kwargs):

    #     if 'range_eps' not in kwargs:
    #         # to prevent blowing up dims that barely change
    #         kwargs['range_eps'] = 5e-2
    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=self.replay_buffer.data, last_n_dims=1, mode=mode, **kwargs)
    #     return normalizer

    def get_beta_priori(self):
        data = self.pref_replay_buffer.data
        pref_data = data.copy()
        meta = self.pref_replay_buffer.meta
        pref_data.update(meta)
        if 'episode_ends' in pref_data.keys():
            del pref_data['episode_ends']
        beta_model = BetaNetwork(observation_dim=len(data['obs'][0, 0, :]), action_dim=len(data['action'][0, 0, :]), lr=1e-3, device='cuda:0',
                 model_type='Transformer', beta_coef=0.1)

        beta_model.fit_data(num_epochs=200, batch_size=50, dataset=pref_data, save_dir='data/beta_model/kitchen'
                            , load_dir='data/beta_model/kitchen/itr_200/beta_model.pth') #load_dir='data/beta_model/kitchen/itr_200/beta_model.pth',

        obs_1 = data['obs']
        obs_2 = data['obs']
        action_1 = data['action']
        action_2 = data['action']
        s_a_1 = np.concatenate([obs_1, action_1], axis=-1)
        s_a_2 = np.concatenate([obs_2, action_2], axis=-1)

        alpha, beta = beta_model.get_alpha_beta(torch.from_numpy(s_a_1).float().to('cuda:0'))
        alpha_2, beta_2 = beta_model.get_alpha_beta(torch.from_numpy(s_a_2).float().to('cuda:0'))

        max_value = torch.max(torch.cat([alpha, beta, alpha_2, beta_2]))
        min_value = torch.min(torch.cat([alpha, beta, alpha_2, beta_2]))

        def scale_to_range(x, min_val, max_val, target_min=1, target_max=25):
            return target_min + (x - min_val) * (target_max - target_min) / (max_val - min_val)

        alpha = scale_to_range(alpha, min_value, max_value)
        beta = scale_to_range(beta, min_value, max_value)
        alpha_2 = scale_to_range(alpha_2, min_value, max_value)
        beta_2 = scale_to_range(beta_2, min_value, max_value)

        self.pref_replay_buffer.meta['beta_priori'] = np.array([alpha.cpu().numpy(), beta.cpu().numpy()]).T
        self.pref_replay_buffer.meta['beta_priori_2'] = np.array([alpha_2.cpu().numpy(), beta_2.cpu().numpy()]).T

    def get_all_actions(self) -> torch.Tensor:
        actions = np.cat(self.pref_replay_buffer.data['action'], self.pref_replay_buffer.data['action_2'], dim = 0)
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        return self.sampler.__len__()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = dict()
        torch_data = self.sampler.sample_sequence(idx)
        return torch_data