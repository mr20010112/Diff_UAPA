from typing import Dict
import torch
import numpy as np
import copy
import pathlib
import h5py
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs

#处理 "kitchen" 任务的低维数据集。该类从 .mjl 文件中解析数据，存储在 ReplayBuffer 中，并对数据进行采样

class KitchenHdf5LowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0, #添加到观测值中的噪声比例，模拟机器人噪声
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        #包含机器人状态维度上噪声振幅的数组
        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
            0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)

        with h5py.File(dataset_dir, 'r') as f:
            actions = f['actions'][:]
            observations = f['observations'][:]
            rewards = f['rewards'][:]
            terminals = f['terminals'][:]
            # timeouts = f['timeouts'][:]

        actions = actions * (np.ones([len(actions), 9]) * 2) + np.zeros([len(actions), 9])
        actions = actions * 0.08 + observations[:, :9]
        observations[:, 30:] = np.zeros([len(observations), 30]) 


        if robot_noise_ratio > 0:
            # add observation noise to match real robot
            noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                low=-1., high=1., size=(observations.shape[0], 30))
            observations[:,:30] += noise

        episode_ends = np.array((terminals == True).nonzero()[0])
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in range(len(episode_ends)):
            end = episode_ends[i]
            if i == 0:
                start = 0
            else:
                start = episode_ends[i-1]  
            try:
                episode = {
                    'obs': observations[start:end],
                    'action': actions[start:end].astype(np.float32),
                    'reward': rewards[start:end].astype(np.float32),
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

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
            # to prevent blowing up dims that barely change
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
