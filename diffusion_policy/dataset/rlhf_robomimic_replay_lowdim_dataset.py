from typing import Dict, List
import torch
import numpy as np
import h5py
import random
from tqdm import tqdm
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.pref_replay_buffer import PrefReplayBuffer
from diffusion_policy.common.pref_sampler import PrefSequenceSampler
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RLHF_RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            normal_replay_buffer: ReplayBuffer,
            expert_replay_buffer: ReplayBuffer,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            sequence_length=1,
            N=1
        ):

        expert_episode_ends = expert_replay_buffer.episode_ends
        expert_num_episodes = len(expert_episode_ends)
        normal_episode_ends = normal_replay_buffer.episode_ends
        normal_num_episodes = len(normal_episode_ends)
        self.pref_replay_buffer = PrefReplayBuffer.create_empty_numpy()

        for _ in range(N):
            idx_expert = random.sample(len(expert_num_episodes), 1)
            idx_normal = random.sample(len(normal_num_episodes), 1)

            episode_expert = expert_replay_buffer.get_episode(idx_expert, keys=['obs', 'action', 'reward'], copy=False)
            episode_normal = normal_replay_buffer.get_episode(idx_normal, keys=['obs', 'action', 'reward'], copy=False)

            # Equal length processing for episode 1
            expert_len = len(episode_expert['obs'])
            if expert_len >= sequence_length:
                start = random.randint(0, expert_len - sequence_length)
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
                start = random.randint(0, normal_len - sequence_length)
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


            # Add preference episode to the replay buffer
            self.pref_replay_buffer.add_pref_episode(
                data={
                    'obs': episode_expert['obs'],          # First trajectory observations (shape T, obs_dim)
                    'action': episode_expert['action'],     # First trajectory actions (shape T, action_dim)
                    'obs_2': episode_normal['obs'],         # Second trajectory observations
                    'action_2': episode_normal['action']    # Second trajectory actions
                },
                meta_data={
                    'votes': votes,                   # Vote for the first trajectory
                    'votes_2': votes_2,               # Vote for the second trajectory
                    'length': np.array([length]),     # Length of the first trajectory
                    'length_2': np.array([length_2]), # Length of the second trajectory
                }
            )  

        # val_mask = get_val_mask(
        #     n_episodes=pref_replay_buffer.n_episodes, 
        #     val_ratio=val_ratio,
        #     seed=seed)
        # train_mask = ~val_mask
        # train_mask = downsample_mask(
        #     mask=train_mask, 
        #     max_n=max_train_episodes, 
        #     seed=seed)

        sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer, 
            sequence_length=sequence_length,
            )
        
        self.replay_buffer = self.pref_replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        # self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.N = N
    
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

    def get_all_actions(self) -> torch.Tensor:
        actions = np.cat(self.pref_replay_buffer.data['action'], self.pref_replay_buffer.data['action_2'], dim = 0)
        return torch.from_numpy(actions)
    
    def __len__(self):
        return self.sampler.__len__()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    
