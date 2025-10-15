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
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.cpl_bet_lowdim_policy import BETLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, 
    SingleFieldLinearNormalizer
)
from diffusion_policy.common.json_logger import JsonLogger
# from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class DatacollectBETLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # # set seed
        # seed = cfg.collecting.seed
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        # configure model
        self.policy: BETLowdimPolicy
        self.policy = hydra.utils.instantiate(cfg.policy)

        self.global_step = 0
        self.epoch = 0


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        OmegaConf.resolve(cfg)

        ckpt_path = pathlib.Path(cfg.checkpoint_dir)
        if ckpt_path.is_file():
            print(f"Resuming from checkpoint {ckpt_path}")
            self.load_checkpoint(path=ckpt_path, exclude_keys='optimizer')
        self.global_step = 0
        self.epoch = 0

        device = torch.device(cfg.collecting.device_gpu)


        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # device transfer
        device = torch.device(cfg.collecting.device_gpu)
        self.policy.to(device)

        all_episodes = {
            'observations': np.empty((0,cfg.task.obs_dim)),  # Time x Features
            'actions': np.empty((0,cfg.task.action_dim)),            # Time x Features
            'rewards': np.empty((0)),            # Time x Features
            'terminals': np.empty((0))           # Time x Features
        }

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.collecting.num_epochs):
                step_log = dict()
                # ========= eval for this epoch ==========
                self.policy.eval()

                # run rollout
                if (self.epoch % cfg.collecting.rollout_every) == 0:
                    runner_log, episode = env_runner.run(self.policy)
                    all_episodes['observations'] = np.concatenate([all_episodes['observations'], episode['observations']], axis=0)
                    all_episodes['actions'] = np.concatenate([all_episodes['actions'], episode['actions']], axis=0)
                    all_episodes['rewards'] = np.concatenate([all_episodes['rewards'], episode['rewards']], axis=0)
                    all_episodes['terminals'] = np.concatenate([all_episodes['terminals'], episode['terminals']], axis=0)
                
                self.global_step += 1
                self.epoch += 1

        with h5py.File(f'{cfg.task.name}_data_0.5.h5', 'w') as f:
            for key, value in all_episodes.items():
                f.create_dataset(key, data=value)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = DatacollectBETLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
