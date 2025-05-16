if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import h5py

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

import time
import logging
from datetime import datetime

# %%
class DatacollectDiffusionTransformerLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            ckpt_path = pathlib.Path(cfg.checkpoint_dir)
            if ckpt_path.is_file():
                print(f"Resuming from checkpoint {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)
            self.global_step = 0
            self.epoch = 0


        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # device transfer
        device = torch.device(cfg.training.device_gpu)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)


        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1


        all_episodes = {
            'observations': np.empty((0,cfg.task.obs_dim)),  # Time x Features
            'actions': np.empty((0,cfg.task.action_dim)),            # Time x Features
            'rewards': np.empty((0)),            # Time x Features
            'terminals': np.empty((0))           # Time x Features
        }

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if ((self.epoch % cfg.training.rollout_every) == 0 and self.epoch >= 1200) or (self.epoch == 0):
                if (self.epoch % cfg.training.rollout_every) == 0:
                    episode, runner_log = env_runner.run(policy) # runner_log, episode = env_runner.run(policy)
                    all_episodes['observations'] = np.concatenate([all_episodes['observations'], episode['observations']], axis=0)
                    all_episodes['actions'] = np.concatenate([all_episodes['actions'], episode['actions']], axis=0)
                    all_episodes['rewards'] = np.concatenate([all_episodes['rewards'], episode['rewards']], axis=0)
                    all_episodes['terminals'] = np.concatenate([all_episodes['terminals'], episode['terminals']], axis=0)

                self.global_step += 1
                self.epoch += 1

        with h5py.File(f'{cfg.task.name}_data_0.5.h5', 'w') as f:
            # 将字典的每个项存储为数据集
            for key, value in all_episodes.items():
                f.create_dataset(key, data=value)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = DatacollectDiffusionTransformerLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
