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
                    # 如果是dataset，读出数据赋值
                    result[key] = item[()]
                elif isinstance(item, h5py.Group):
                    # 如果是group，递归调用
                    result[key] = h5_to_data(item)
            return result

        def decode_image(data):
            return cv2.imdecode(data, 1)

        for filename in tqdm(hdf5_files, desc='Processing HDF5 files'):
            file_path = os.path.join(dataset_dir, filename)
            with h5py.File(file_path, 'r') as f:
                # observations = f['observations']
                # action = f['action']
                # original_action_shape = action.shape
                # episode_len = original_action_shape[0]
                # camera_keys = observations['images'].keys()

                # for key in observations:
                #     if key != 'images':
                #         observation_data.append(observations[key])
                # qpos = np.concatenate(observation_data, axis=-1)

                # padded_action = np.zeros((self.max_steps, original_action_shape[1]), dtype=np.float32)
                # padded_action[:episode_len] = action

                # all_cam_images = []
                # for cam_name in camera_keys:
                #     all_cam_images.append(observations['images'][cam_name])
                # all_cam_images = np.concatenate(all_cam_images, axis=0)
                # image_data = np.einsum('k h w c -> k c h w', all_cam_images)
                # image_data = image_data / 255.0
            
                # episode = {
                #     'image': image_data.astype(np.float32),
                #     'action': action.astype(np.float32),
                #     'qpos': qpos.astype(np.float32),
                # }
                data = h5_to_data(f)
                del data['compress_len']
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
        data = {k: v for k, v in data.items() if "observation" not in k}
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
