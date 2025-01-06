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
            replay_buffer_1: ReplayBuffer,
            replay_buffer_2: ReplayBuffer,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            gamma=0.9999,
            max_train_episodes=None,
            load_dir=None,
            sequence_length=1,
            N=1
        ):

        episode_ends_1 = replay_buffer_1.episode_ends
        num_episodes_1 = len(episode_ends_1)
        episode_ends_2 = replay_buffer_2.episode_ends
        num_episodes_2 = len(episode_ends_2)
        self.pref_replay_buffer = PrefReplayBuffer.create_empty_numpy()


        if load_dir is None:
            for _ in range(N):
                idx_1 = random.sample(range(num_episodes_1), 1)[0]
                idx_2 = random.sample(range(num_episodes_2), 1)[0]

                episode_1 = replay_buffer_1.get_episode(idx_1, keys=['obs', 'action', 'reward'], copy=False)
                episode_2 = replay_buffer_2.get_episode(idx_2, keys=['obs', 'action', 'reward'], copy=False)

                # Equal length processing for episode 1
                episode_1_len = len(episode_1['obs'])
                if episode_1_len >= sequence_length:
                    start = 0
                    length = sequence_length
                    for key in episode_1.keys():
                        episode_1[key] = episode_1[key][start:start + sequence_length]
                else:
                    for key in episode_1.keys():
                        episode_1[key] = np.pad(episode_1[key], 
                                            ((0, sequence_length - episode_1_len),) + ((0, 0),) * (episode_1[key].ndim - 1),
                                            mode='edge')
                        length = episode_1_len

                # Equal length processing for episode 2
                episode_2_len = len(episode_2['obs'])
                if episode_2_len >= sequence_length:
                    start = 0
                    length_2 = sequence_length
                    for key in episode_2.keys():
                        episode_2[key] = episode_2[key][start:start + sequence_length]
                else:
                    for key in episode_2.keys():
                        episode_2[key] = np.pad(episode_2[key], 
                                            ((0, sequence_length - episode_2_len),) + ((0, 0),) * (episode_2[key].ndim - 1),
                                            mode='edge')
                        length_2 = episode_2_len


                # Set up votes and metadata based on the presence of 'reward' in episode1
                if 'reward' not in episode_1.keys() or len(episode_1['reward']) == 0:
                    votes = np.zeros((1,))
                    votes_2 = np.zeros((1,))
                else:
                    votes = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_1['reward'])])
                    votes_2 = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_2['reward'])])


            # Add preference episode to the replay buffer
                self.pref_replay_buffer.add_pref_episode(
                    data={
                        'obs': episode_1['obs'],          # First trajectory observations (shape T, obs_dim)
                        'action': episode_1['action'],     # First trajectory actions (shape T, action_dim)
                        'obs_2': episode_2['obs'],         # Second trajectory observations
                        'action_2': episode_2['action'],    # Second trajectory actions
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

    def construct_pref_data(self):
        data = self.pref_replay_buffer.data
        pref_data = data.copy()
        meta = self.pref_replay_buffer.meta
        pref_data.update(meta)
        if 'episode_ends' in pref_data.keys():
            del pref_data['episode_ends']

        return pref_data

    def set_beta_priori(self, data_size=100):
        pref_data = self.construct_pref_data()
        self.beta_model = BetaNetwork(data=pref_data,
                                 device=self.gpu_device,
                                 data_size=data_size)

    def update_beta_priori(self, batch_size=3):

        def scale_to_range(x, min_val, max_val, target_min=1, target_max=10):

            if min_val == max_val:
                raise ValueError("min_val and max_val must be different to avoid division by zero.")
            return target_min + (x - min_val) * (target_max - target_min) / (max_val - min_val)

        # Define unified scaling logic
        def scale_tensor(x, global_min, global_max, target_min=1, target_max=10):

            # Ensure the tensor is a floating-point tensor
            if not torch.is_floating_point(x):
                x = x.float()

            local_min, local_max = torch.min(x), torch.max(x)

            # Handle division by zero for local range
            if local_min == local_max:
                return torch.full_like(x, target_min)  # Return tensor filled with target_min

            # Handle division by zero for global range
            # if global_min < 1:
            #     global_min = 1  # Replace 0 with a small positive value to avoid division by zero
            if global_min == global_max:
                raise ValueError("global_min and global_max must be different to avoid division by zero.")

            # Compute scaled local range
            scaled_min = (local_min / global_min) * target_min
            scaled_max = (local_max / global_max) * target_max

            # Apply scaling
            return scale_to_range(x, local_min, local_max, scaled_min, scaled_max)

        obs_1 = self.pref_replay_buffer.data['obs']
        obs_2 = self.pref_replay_buffer.data['obs_2']
        action_1 = self.pref_replay_buffer.data['action']
        action_2 = self.pref_replay_buffer.data['action_2']
        s_a_1 = np.concatenate([obs_1, action_1], axis=-1)
        s_a_2 = np.concatenate([obs_2, action_2], axis=-1)

        interval = math.ceil(s_a_1.shape[0] / batch_size)
        alpha, beta = [], []
        alpha_2, beta_2 = [], []
        for i in range(interval):
            start_pt = i * batch_size
            end_pt = min((i + 1) * batch_size, s_a_1.shape[0])
            batch_s_a_1 = s_a_1[start_pt:end_pt, ...]
            batch_s_a_2 = s_a_2[start_pt:end_pt, ...]

            batch_alpha, batch_beta = self.beta_model.get_alpha_beta(torch.from_numpy(batch_s_a_1).float().to(self.beta_model.device))
            batch_alpha_2, batch_beta_2 = self.beta_model.get_alpha_beta(torch.from_numpy(batch_s_a_2).float().to(self.beta_model.device))

            alpha.append(batch_alpha)
            beta.append(batch_beta)
            alpha_2.append(batch_alpha_2)
            beta_2.append(batch_beta_2)

        alpha = torch.cat(alpha, dim=0)+1
        beta = torch.cat(beta, dim=0)+1
        alpha_2 = torch.cat(alpha_2, dim=0)+1
        beta_2 = torch.cat(beta_2, dim=0)+1

        mean_value = torch.mean(torch.cat([alpha, beta, alpha_2, beta_2]))
        std_value = torch.std(torch.cat([alpha, beta, alpha_2, beta_2]))

        alpha = torch.clamp(alpha, max=mean_value+3*std_value)
        beta = torch.clamp(beta, max=mean_value+3*std_value)
        alpha_2 = torch.clamp(alpha_2, max=mean_value+3*std_value)
        beta_2 = torch.clamp(beta_2, max=mean_value+3*std_value)

        max_value = torch.max(torch.cat([alpha, beta, alpha_2, beta_2]))
        min_value = torch.min(torch.cat([alpha, beta, alpha_2, beta_2]))

        target_min, target_max = 1, 10

        alpha = scale_tensor(alpha, min_value, max_value, target_min, target_max)
        beta = scale_tensor(beta, min_value, max_value, target_min, target_max)
        alpha_2 = scale_tensor(alpha_2, min_value, max_value, target_min, target_max)
        beta_2 = scale_tensor(beta_2, min_value, max_value, target_min, target_max)

        self.pref_replay_buffer.meta['beta_priori'] = np.array([alpha.cpu().numpy(), beta.cpu().numpy()]).T
        self.pref_replay_buffer.meta['beta_priori_2'] = np.array([alpha_2.cpu().numpy(), beta_2.cpu().numpy()]).T
        self.pref_replay_buffer.root['meta']['beta_priori'] = np.array([alpha.cpu().numpy(), beta.cpu().numpy()]).T
        self.pref_replay_buffer.root['meta']['beta_priori_2'] = np.array([alpha_2.cpu().numpy(), beta_2.cpu().numpy()]).T

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer, 
            sequence_length=self.sequence_length,
            episode_mask=~self.train_mask,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

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
    
