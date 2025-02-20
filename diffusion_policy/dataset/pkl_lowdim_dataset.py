from typing import Dict
import torch
import numpy as np
import copy
from tqdm import tqdm
import pickle
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class PklLowdimDataset(BaseLowdimDataset):
    def __init__(self,
                 dataset_path=None,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 abs_action=True,
                 seed=42,
                 val_ratio=0.0):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        # Load data from the .pkl file
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        observations = data['observations']
        actions = data['actions']
        rewards = data['seg_reward_1']

        observations_2 = data['observations_2']
        actions_2 = data['actions_2']
        rewards_2 = data['seg_reward_2']

        self.replay_buffer = ReplayBuffer.create_empty_numpy()

        for i in range(len(observations)):
            episode = {
                'obs': observations[i],
                'action': actions[i].astype(np.float32),
                'reward': rewards[i].astype(np.float32),
            }
            self.replay_buffer.add_episode(episode)

        for i in range(len(observations_2)):
            episode = {
                'obs': observations_2[i],
                'action': actions_2[i].astype(np.float32),
                'reward': rewards_2[i].astype(np.float32),
            }
            self.replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action'],
            'reward': self.replay_buffer['reward'],
        }
        if 'range_eps' not in kwargs:
            # Prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
