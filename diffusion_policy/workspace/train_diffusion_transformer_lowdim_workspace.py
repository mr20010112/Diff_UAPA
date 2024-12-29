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
import matplotlib.pyplot as plt

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from torch.cuda.amp import GradScaler, autocast 
from diffusion_policy.model.common.slice import slice_episode
from diffusion_policy.common.compute_all_loss import compute_all_traj_loss
#from diffusion_policy.dataset.rlhf_kitchen_mjl_lowdim_dataset import RLHF_KitchenMjlLowdimDataset

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionTransformerLowdimWorkspace(BaseWorkspace):
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
        #print(cfg.policy)
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
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
            self.global_step = 0
            self.epoch = 0

        device = torch.device(cfg.training.device_gpu)
        ref_model = copy.deepcopy(self.model.model)
        # ref_model.double()
        ref_model.eval() 
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.to(device)

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #add

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.origin_dataset)
        #device = torch.device(cfg.training.device_cpu)
        assert isinstance(dataset, BaseLowdimDataset)
        normalizer = dataset.get_normalizer()

        # configure dataset
        expert_dataset: BaseLowdimDataset
        expert_dataset = hydra.utils.instantiate(cfg.task.expert_dataset)
        #device = torch.device(cfg.training.device_cpu)
        assert isinstance(expert_dataset, BaseLowdimDataset)


        # configure dataset
        normal_dataset: BaseLowdimDataset
        normal_dataset = hydra.utils.instantiate(cfg.task.normal_dataset)
        # expert_normalizer = normal_dataset.get_normalizer()
        assert isinstance(normal_dataset, BaseLowdimDataset)

        pref_dataset: BaseLowdimDataset
        pref_dataset = hydra.utils.instantiate(cfg.task.pref_dataset, expert_replay_buffer=expert_dataset.replay_buffer, \
                                               normal_replay_buffer=normal_dataset.replay_buffer) #cfg.task.perf_dataset
        pref_dataset.set_beta_priori()
        pref_dataset.beta_model.online_update(dataset=pref_dataset.construct_pref_data(), num_epochs=10, warm_up_epochs=2, batch_size=5, lr = 1.0e-6)

        train_dataloader = DataLoader(pref_dataset, **cfg.dataloader)
        del dataset, expert_dataset, normal_dataset

        # configure validation dataset
        val_dataset = pref_dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            # self.ema_model.double()
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # # pytorch assumes stepping LRScheduler every epoch
            # # however huggingface diffusers steps it every batch
            # lr_end=cfg.training.lr_end,
            # num_cycles=cfg.training.num_cycles,
            last_epoch=self.global_step-1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
                "timeout": 300,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device_gpu)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        #device = torch.device(cfg.training.device_gpu)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        device = torch.device(cfg.training.device_gpu)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                self.model.train()
                if self.ema_model is not None:
                    self.ema_model.train()
                train_losses = list()
                # map_loss = []
                # avg_traj_loss = compute_all_traj_loss(replay_buffer = pref_dataset.pref_replay_buffer, model = self.model, ref_model = ref_model)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        avg_traj_loss = 0.0
                        if cfg.training.map.use_map and (not cfg.training.map.map_batch_update):
                            avg_traj_loss = compute_all_traj_loss(replay_buffer = pref_dataset.pref_replay_buffer, model = self.model, ref_model = ref_model)
                        raw_loss = self.model.compute_loss(batch, ref_model=ref_model, avg_traj_loss = avg_traj_loss)
                        # map_loss_batch_numpy = [tensor.detach().cpu().numpy() for tensor in map_loss_batch]
                        # map_loss.append(map_loss_batch_numpy)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log, episode_data = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch, ref_model=ref_model, avg_traj_loss = avg_traj_loss)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        get_obs = batch["obs"]
                        gt_action = batch["action"]
                        
                        
                        start_idx = np.random.randint(0, gt_action.shape[1] - self.model.horizon + 1)
                        end_idx = start_idx + self.model.horizon

                        get_obs = get_obs[:, start_idx:end_idx, :]
                        gt_action = gt_action[:, start_idx:end_idx, :]
                        obs_dict = {'obs': get_obs}

                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        pred_action = pred_action.to(gt_action.device, non_blocking=True)
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
