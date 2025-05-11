from typing import Dict
import torch
import numpy as np
import copy
import pathlib
import h5py
import os
import cv2
from tqdm import tqdm
import concurrent.futures
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.realrobot_replay_buffer import RealRobotReplayBuffer
from diffusion_policy.common.realrobot_sampler import RealRobotSequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)


class Hdf5RealRobotDataset(BaseImageDataset):
    def __init__(self,
            dataset_dir=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        # camera2robot_matrix = np.array([[ 0.992127, 0.016251, 0.124175, 0.306697],
        # [-0.04194, -0.891172, 0.451723,-0.409919],
        # [ 0.118002,-0.453375,-0.883474, 0.828481],
        # [-0.0, 0.0, -0.0, 1.0]])
        # # NEW VIEW
        # camera2robot_matrix = np.array([[ 0.99754313, 0.0207489, 0.06691178, 0.30499173],
        # [-0.01211929, -0.88961861, 0.45654336, -0.40985323],
        # [ 0.06899874, -0.45623262, -0.88718148, 0.82828758],
        # [-0.0, 0.0, -0.0, 1.0 ]])

        # # NOTE: obtain the inverse matrix from robot_base to cam frame
        # robot2camera_matrix = np.eye(4)
        # R_mat = camera2robot_matrix[:3, :3]
        # t_mat = camera2robot_matrix[:3, 3]
        # R_inv = R_mat.T
        # t_inv = -(R_inv @ t_mat)
        # robot2camera_matrix[:3, :3] = R_inv
        # robot2camera_matrix[:3, 3] = t_inv

        # GRIPPER_LEN = 0.175
        # GRIPPER_ROT = -np.pi/4

        # GRIPPER_LEN = 0.175
        # GRIPPER_ROT = np.pi/4

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

        observation_data = []
        self.replay_buffer = RealRobotReplayBuffer.create_empty_numpy()

        hdf5_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.hdf5', '.h5'))]

        def h5_to_data(h5_obj):
            result = {}
            for key, item in h5_obj.items():
                if isinstance(item, h5py.Dataset):
                    result[key] = item[()]
                elif isinstance(item, h5py.Group):
                    result[key] = h5_to_data(item)
            return result

        for filename in tqdm(hdf5_files, desc='Processing HDF5 files'):
            file_path = os.path.join(dataset_dir, filename)
            with h5py.File(file_path, 'r') as f:
                data = h5_to_data(f)
                del data['compress_len']
                del data['observations']['images']['cam_right'] #also need to be changed in config

                data['action'] = data['action'][:, :14]

                for key in data['observations']['images'].keys():
                    image_data = data['observations']['images'][key]
                    save_length = image_data.shape[-1]
                    pad_image_data = np.pad(image_data, ((0, 0), (0, 200000-save_length)), 'constant', constant_values = (0,0))
                    data['observations']['images'][key] = pad_image_data
                    compress_len = np.tile([save_length], (image_data.shape[0], 1))
                    data.update({'compress_len':compress_len})
                    # decompressed_images = []
                    # with concurrent.futures.ThreadPoolExecutor() as executor:
                    #     results = executor.map(decode_image, image_data)
                    #     decompressed_images = list(results)

                    # decompressed_images = np.array(decompressed_images)
                    # data['observations']['images'][key] = decompressed_images

                data = flatten_dataset_dict(data)
                 

                self.replay_buffer.add_episode(data)


        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = RealRobotSequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = RealRobotSequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self.replay_buffer.data #To Do
        image_keys = [k for k in data.keys() if "image" in k]
        qpos_keys = [k for k in data.keys() if ("observation" in k) and ("image" not in k)]


        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer.data['action'])
        
        for key in image_keys:
            key = key.replace("observations/images/", "")
            normalizer[key] = get_image_range_normalizer()

        for key in qpos_keys:
            new_key = key.replace("observations/", "")        
            normalizer[new_key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer.data[key])

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
