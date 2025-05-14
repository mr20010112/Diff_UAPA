from typing import Dict
import torch
import numpy as np
import copy
import pathlib
import random
import hydra
import h5py
import math
import math
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
import concurrent.futures
import cv2
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs
from diffusion_policy.common.pref_realrobot_replay_buffer import Pref_RealRobotReplayBuffer
from diffusion_policy.common.pref_realrobot_sampler import PrefSequenceSampler, PrefSliceSampler
from typing import Optional, Dict
from diffusion_policy.common.prior_utils_vae import RealRobotBetaNetwork
from diffusion_policy.model.vision.realrobot_image_obs_encoder import RealRobotImageObsEncoder

#处理 "kitchen" 任务的低维数据集。该类从 .mjl 文件中解析数据，存储在 ReplayBuffer 中，并对数据进行采样
class RLHF_RealRobotDataset(BaseImageDataset):
    def __init__(self,
                dataset_1: BaseImageDataset,
                dataset_2: BaseImageDataset,
                abs_action=True,
                sequence_length=1,
                gamma=0.9999,
                N=1,
                seed=42,
                val_ratio=0.0,
                load_dir=None,
                save_dir=None,
                gpu_device = 'cuda:0',
                ):
        super().__init__()

        def flatten_dataset_dict(d, parent_key='', sep='/'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        new_key2 = f"{new_key}{sep}{k2}"
                        if isinstance(v2, dict):
                            for k3, v3 in v2.items():
                                new_key3 = f"{new_key2}{sep}{k3}"
                                items.append((new_key3, v3))
                        else:
                            items.append((new_key2, v2))
                else:
                    items.append((new_key, v))
            return dict(items)

        if not abs_action:
            raise NotImplementedError()

        episode_ends_1 = dataset_1.replay_buffer.episode_ends
        num_episodes_1 = len(episode_ends_1)
        episode_ends_2 = dataset_2.replay_buffer.episode_ends
        num_episodes_2 = len(episode_ends_2)
        self.pref_replay_buffer = Pref_RealRobotReplayBuffer.create_empty_numpy()


        # if load_dir is None:
        for _ in tqdm(range(N), desc='Processing HDF5 data'):
            idx_1 = random.sample(range(num_episodes_1), 1)[0]
            idx_2 = random.sample(range(num_episodes_2), 1)[0]

            episode_1 = dataset_1.replay_buffer.get_episode(idx_1)
            episode_2 = dataset_2.replay_buffer.get_episode(idx_2)

            # Equal length processing for episode 1
            episode_1_len = len(episode_1['action'])
            if episode_1_len >= sequence_length:
                start = 0
                length = sequence_length
                for key in episode_1.keys():
                    if key == 'observations':
                        for sub_key in episode_1['observations'].keys():
                            if sub_key == 'images':
                                for sub_sub_key in episode_1['observations']['images'].keys():
                                    episode_1['observations']['images'][sub_sub_key] = episode_1['observations']['images'][sub_sub_key][start:start + sequence_length]
                            else:
                                episode_1['observations'][sub_key] = episode_1['observations'][sub_key][start:start + sequence_length]
                    else:
                        episode_1[key] = episode_1[key][start:start + sequence_length]
            else:
                for key in episode_1.keys():
                    episode_1[key] = np.pad(episode_1[key], 
                                        ((0, sequence_length - episode_1_len),) + ((0, 0),) * (episode_1[key].ndim - 1),
                                        mode='edge')
                    length = episode_1_len
            episode_1['obs'] = episode_1['observations']
            del episode_1['observations']
            data = flatten_dataset_dict(episode_1)

            # Equal length processing for episode 2
            episode_2_len = len(episode_2['action'])
            if episode_2_len >= sequence_length:
                start = 0
                length_2 = sequence_length
                for key in episode_2.keys():
                    if key == 'observations':
                        for sub_key in episode_2['observations'].keys():
                            if sub_key == 'images':
                                for sub_sub_key in episode_2['observations']['images'].keys():
                                    episode_2['observations']['images'][sub_sub_key] = episode_2['observations']['images'][sub_sub_key][start:start + sequence_length]
                            else:
                                episode_2['observations'][sub_key] = episode_2['observations'][sub_key][start:start + sequence_length]
                    else:
                        episode_2[key] = episode_2[key][start:start + sequence_length]
            else:
                for key in episode_2.keys():
                    episode_2[key] = np.pad(episode_2[key], 
                                        ((0, sequence_length - episode_2_len),) + ((0, 0),) * (episode_2[key].ndim - 1),
                                        mode='edge')
                    length_2 = episode_2_len
            episode_2['action_2'] = episode_2['action']
            episode_2['obs_2'] = episode_2['observations']
            episode_2['compress_len_2'] = episode_2['compress_len']
            data.update(flatten_dataset_dict(episode_2))


            # Set up votes and metadata based on the presence of 'reward' in episode1
            if 'reward' not in episode_1.keys():
                votes = np.ones((1,))
                votes_2 = np.zeros((1,))
            else:
                votes = np.sum([gamma ** t * reward for t, reward in enumerate(episode_1['reward'])])
                votes_2 = np.sum([gamma ** t * reward for t, reward in enumerate(episode_2['reward'])])


            # Add preference episode to the replay buffer
            self.pref_replay_buffer.add_pref_episode(
                data=data,
                meta_data={
                    'votes': votes,                   # Vote for the first trajectory
                    'votes_2': votes_2,               # Vote for the second trajectory
                    'beta_priori': np.zeros([2]),
                    'beta_priori_2': np.zeros([2]),
                }
            )

        val_mask = get_val_mask(
            n_episodes=N, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        #self.data = pref_dataset
        self.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer,
            sequence_length=sequence_length,
            episode_mask=train_mask,
        )

        self.gpu_device = gpu_device
        self.length = N
        self.train_mask = train_mask
        self.sequence_length = sequence_length
        # self.beta_model: Optional[BetaNetwork] = None


    # def get_normalizer(self, mode='limits', **kwargs):

    #     if 'range_eps' not in kwargs:
    #         # to prevent blowing up dims that barely change
    #         kwargs['range_eps'] = 5e-2
    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=self.replay_buffer.data, last_n_dims=1, mode=mode, **kwargs)
    #     return normalizer

    def unflatten_dataset_dict(self, flat_dict, delimiter='/'):
        result = {}
        for compound_key, value in flat_dict.items():
            keys = compound_key.split(delimiter)
            current = result
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        
        return result

    def construct_pref_data(self):
        data = self.pref_replay_buffer.data
        pref_data = data.copy()
        meta = self.pref_replay_buffer.meta
        pref_data.update(meta)
        if 'episode_ends' in pref_data.keys():
            del pref_data['episode_ends']
        pref_data = self.unflatten_dataset_dict(pref_data)

        return pref_data

    def set_beta_priori(self, obs_encoder: RealRobotImageObsEncoder, normalizer=None):
        data = self.pref_replay_buffer.data
        action_dim = data['action'].shape[-1]
        observation_dim = obs_encoder.output_shape()[0]
        del data
        self.beta_model = RealRobotBetaNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            obs_encoder=obs_encoder,
            normalizer=normalizer,
        )

    def update_beta_priori(self, batch_size=3):

        def unflatten_dataset_dict(flat_dict, delimiter='/'):
            result = {}
            for compound_key, value in flat_dict.items():
                keys = compound_key.split(delimiter)
                current = result
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
            
            return result
        
        def decode_image(data):
            return cv2.imdecode(data, 1)
        
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
        
        with torch.no_grad():
            data = self.pref_replay_buffer.data
            data = unflatten_dataset_dict(flat_dict=data)

            obs_1 = data['obs']
            obs_2 = data['obs_2']
            action_1 = data['action']
            action_2 = data['action_2']
            compress_len = data["compress_len"]
            compress_len_2 = data["compress_len_2"]
            votes_1 = self.pref_replay_buffer.meta["votes"]
            votes_2 = self.pref_replay_buffer.meta["votes_2"]
            del data

            qpos_keys = [key for key in obs_1.keys() if key != 'images']
            camera_keys = obs_1["images"].keys()

            interval = math.ceil(action_1.shape[0] / batch_size)

            alpha = []
            beta = []
            alpha_2 = []
            beta_2 = []
            for i in tqdm(range(interval), desc="Updating beta priori"):
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, action_1.shape[0])
                indices = np.arange(start_pt, end_pt)
                
                batch_compress_len = compress_len[indices]
                batch_compress_len_2 = compress_len_2[indices]
                
                batch_act_1 = torch.from_numpy(action_1[indices]).float()
                batch_act_2 = torch.from_numpy(action_2[indices]).float()
                batch_votes_1 = torch.from_numpy(votes_1[indices]).float()
                batch_votes_2 = torch.from_numpy(votes_2[indices]).float()
                
                batch_obs_1 = {}
                batch_obs_2 = {}
                
                for key in qpos_keys:
                    batch_obs_1[key] = torch.from_numpy(obs_1[key][indices]).float()
                    batch_obs_2[key] = torch.from_numpy(obs_2[key][indices]).float()

                for key in camera_keys:
                    batch_image_data_1 = obs_1["images"][key][indices]
                    batch_decompressed_images_1 = []
                    
                    for k in range(len(batch_image_data_1)):
                        img_data = batch_image_data_1[k, :, :int(batch_compress_len[k, 0])].copy()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            results = executor.map(decode_image, img_data)
                            decompressed_images = list(results)
                        batch_decompressed_images_1.append(decompressed_images)
                    
                    batch_decompressed_images_1 = np.array(batch_decompressed_images_1)
                    batch_decompressed_images_1 = np.einsum('b k h w c -> b k c h w', batch_decompressed_images_1)
                    batch_obs_1[key] = torch.from_numpy(batch_decompressed_images_1 / 255.0).float()
                    
                    del batch_decompressed_images_1, batch_image_data_1
                    
                    batch_image_data_2 = obs_2["images"][key][indices]
                    batch_decompressed_images_2 = []
                    
                    for k in range(len(batch_image_data_2)):
                        img_data_2 = batch_image_data_2[k, :, :int(batch_compress_len_2[k, 0])].copy()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            results_2 = executor.map(decode_image, img_data_2)
                            decompressed_images_2 = list(results_2)
                        batch_decompressed_images_2.append(decompressed_images_2)
                    
                    batch_decompressed_images_2 = np.array(batch_decompressed_images_2)
                    batch_decompressed_images_2 = np.einsum('b k h w c -> b k c h w', batch_decompressed_images_2)
                    batch_obs_2[key] = torch.from_numpy(batch_decompressed_images_2 / 255.0).float()
                    
                    del batch_decompressed_images_2, batch_image_data_2

                batch_obs_1 = self.beta_model.normalizer.normalize(batch_obs_1)
                for key in batch_obs_1.keys():
                    batch_obs_1[key] = batch_obs_1[key].to(self.beta_model.device)
                
                batch_act_1 = self.beta_model.normalizer['action'].normalize(batch_act_1).to(self.beta_model.device)
                
                this_nobs_1 = dict_apply(batch_obs_1, lambda x: x.reshape(-1, *x.shape[2:]))
                nobs_features_1 = self.beta_model.obs_encoder(this_nobs_1)
                global_cond_1 = nobs_features_1.reshape(batch_size, self.sequence_length, -1)
                
                batch_s_a_1 = torch.cat([global_cond_1, batch_act_1], dim=-1)
                del this_nobs_1, nobs_features_1, global_cond_1, batch_obs_1, batch_act_1

                
                batch_obs_2 = self.beta_model.normalizer.normalize(batch_obs_2)
                for key in batch_obs_2.keys():
                    batch_obs_2[key] = batch_obs_2[key].to(self.beta_model.device)
                
                batch_act_2 = self.beta_model.normalizer['action'].normalize(batch_act_2).to(self.beta_model.device)
                
                this_nobs_2 = dict_apply(batch_obs_2, lambda x: x.reshape(-1, *x.shape[2:]))
                nobs_features_2 = self.beta_model.obs_encoder(this_nobs_2)
                global_cond_2 = nobs_features_2.reshape(batch_size, self.sequence_length, -1)
                
                batch_s_a_2 = torch.cat([global_cond_2, batch_act_2], dim=-1)
                del this_nobs_2, nobs_features_2, global_cond_2, batch_obs_2, batch_act_2
                
                batch_alpha, batch_beta = self.beta_model.get_alpha_beta(batch_s_a_1.float().to(self.beta_model.device))
                batch_alpha_2, batch_beta_2 = self.beta_model.get_alpha_beta(batch_s_a_2.float().to(self.beta_model.device))

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

            target_min, target_max = 1, 2

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
        actions = np.concatenate(self.pref_replay_buffer.data['action'], self.pref_replay_buffer.data['action_2'], dim = 0)
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        return self.sampler.__len__()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = dict()
        torch_data = self.sampler.sample_sequence(idx)
        return torch_data


class RLHF_RealRobotSliceDataset(BaseImageDataset):
    def __init__(self,
                dataset_1: BaseImageDataset,
                dataset_2: BaseImageDataset,
                sequence_length=1,
                horizon=1,
                stride=1,
                gamma=0.9999,
                N=1,
                seed=42,
                val_ratio=0.0,
                load_dir=None,
                save_dir=None,
                gpu_device = 'cuda:0',
                ):
        super().__init__()

        def flatten_dataset_dict(d, parent_key='', sep='/'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        new_key2 = f"{new_key}{sep}{k2}"
                        if isinstance(v2, dict):
                            for k3, v3 in v2.items():
                                new_key3 = f"{new_key2}{sep}{k3}"
                                items.append((new_key3, v3))
                        else:
                            items.append((new_key2, v2))
                else:
                    items.append((new_key, v))
            return dict(items)


        episode_ends_1 = dataset_1.replay_buffer.episode_ends
        num_episodes_1 = len(episode_ends_1)
        episode_ends_2 = dataset_2.replay_buffer.episode_ends
        num_episodes_2 = len(episode_ends_2)
        self.pref_replay_buffer = Pref_RealRobotReplayBuffer.create_empty_numpy()
        self.pref_replay_buffer.get_traj_info(sequence_length=sequence_length, horizon=horizon, stride=stride)
        


        # if load_dir is None:
        for _ in tqdm(range(N), desc='Processing HDF5 data'):
            idx_1 = random.sample(range(num_episodes_1), 1)[0]
            idx_2 = random.sample(range(num_episodes_2), 1)[0]

            episode_1 = dataset_1.replay_buffer.get_episode(idx_1)
            episode_2 = dataset_2.replay_buffer.get_episode(idx_2)

            # Equal length processing for episode 1
            episode_1_len = len(episode_1['action'])
            if episode_1_len >= sequence_length:
                start = 0
                length = sequence_length
                for key in episode_1.keys():
                    if key == 'observations':
                        for sub_key in episode_1['observations'].keys():
                            if sub_key == 'images':
                                for sub_sub_key in episode_1['observations']['images'].keys():
                                    episode_1['observations']['images'][sub_sub_key] = episode_1['observations']['images'][sub_sub_key][start:start + sequence_length]
                            else:
                                episode_1['observations'][sub_key] = episode_1['observations'][sub_key][start:start + sequence_length]
                    else:
                        episode_1[key] = episode_1[key][start:start + sequence_length]
            else:
                for key in episode_1.keys():
                    episode_1[key] = np.pad(episode_1[key], 
                                        ((0, sequence_length - episode_1_len),) + ((0, 0),) * (episode_1[key].ndim - 1),
                                        mode='edge')
                    length = episode_1_len
            episode_1['obs'] = episode_1['observations']
            data = flatten_dataset_dict(episode_1)

            # Equal length processing for episode 2
            episode_2_len = len(episode_2['action'])
            if episode_2_len >= sequence_length:
                start = 0
                length_2 = sequence_length
                for key in episode_2.keys():
                    if key == 'observations':
                        for sub_key in episode_2['observations'].keys():
                            if sub_key == 'images':
                                for sub_sub_key in episode_2['observations']['images'].keys():
                                    episode_2['observations']['images'][sub_sub_key] = episode_2['observations']['images'][sub_sub_key][start:start + sequence_length]
                            else:
                                episode_2['observations'][sub_key] = episode_2['observations'][sub_key][start:start + sequence_length]
                    else:
                        episode_2[key] = episode_2[key][start:start + sequence_length]
            else:
                for key in episode_2.keys():
                    episode_2[key] = np.pad(episode_2[key], 
                                        ((0, sequence_length - episode_2_len),) + ((0, 0),) * (episode_2[key].ndim - 1),
                                        mode='edge')
                    length_2 = episode_2_len
            episode_2['action_2'] = episode_2['action']
            episode_2['obs_2'] = episode_2['observations']
            episode_2['compress_len_2'] = episode_2['compress_len']
            data.update(flatten_dataset_dict(episode_2))


            # Set up votes and metadata based on the presence of 'reward' in episode1
            if 'reward' not in episode_1.keys():
                votes = np.ones((1,))
                votes_2 = np.zeros((1,))
            else:
                votes = np.sum([gamma ** t * reward for t, reward in enumerate(episode_1['reward'])])
                votes_2 = np.sum([gamma ** t * reward for t, reward in enumerate(episode_2['reward'])])


            # Add preference episode to the replay buffer
            self.pref_replay_buffer.add_pref_episode(
                data=data,
                meta_data={
                    'votes': votes,                   # Vote for the first trajectory
                    'votes_2': votes_2,               # Vote for the second trajectory
                    'beta_priori': np.zeros([2]),
                    'beta_priori_2': np.zeros([2]),
                }
            )

        val_mask = get_val_mask(
            n_episodes=N, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        #self.data = pref_dataset
        self.sampler = PrefSliceSampler(
            replay_buffer=self.pref_replay_buffer,
            sequence_length=sequence_length,
            episode_mask=train_mask,
            horizon=horizon,
            stride=stride,
        )

        self.pref_replay_buffer.slice_all_data()
        self.gpu_device = gpu_device
        self.length = N
        self.train_mask = train_mask
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.stride = stride
        self.beta_model: Optional[BetaNetwork] = None


    # def get_normalizer(self, mode='limits', **kwargs):

    #     if 'range_eps' not in kwargs:
    #         # to prevent blowing up dims that barely change
    #         kwargs['range_eps'] = 5e-2
    #     normalizer = LinearNormalizer()
    #     normalizer.fit(data=self.replay_buffer.data, last_n_dims=1, mode=mode, **kwargs)
    #     return normalizer

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
        val_set.sampler = PrefSliceSampler(
            replay_buffer=self.pref_replay_buffer, 
            sequence_length=self.sequence_length,
            episode_mask=~self.train_mask,
            horizon=self.horizon,
            stride=self.stride,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_all_actions(self) -> torch.Tensor:
        actions = np.concatenate(self.pref_replay_buffer.data['action'], self.pref_replay_buffer.data['action_2'], dim = 0)
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        traj_num = self.sampler.__len__()
        local_num = math.floor((self.sequence_length - self.horizon) / self.stride) + 1
        return traj_num * local_num

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = dict()
        torch_data = self.sampler.sample_slice(idx)
        return torch_data