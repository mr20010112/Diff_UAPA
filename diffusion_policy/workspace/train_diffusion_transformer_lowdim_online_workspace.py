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
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import beta
from typing import Optional, Dict

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.prior_utils_confidence import BetaNetwork
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
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
from diffusion_policy.common.compare_policy import comp_policy
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
        ref_policy = copy.deepcopy(self.model)
        # ref_model.double()
        ref_policy.eval() 
        for param in ref_policy.parameters():
            param.requires_grad = False
        ref_policy.to(device)

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

        # pref_dataset.set_beta_priori(data_size=150)
        # pref_dataset.beta_model.fit_data(num_epochs=50, warm_up_epochs=5, batch_size=5, lr=1.0e-5)
        # pref_dataset.update_beta_priori()

        # cut online groups
        all_votes_1, all_votes_2 = np.array([]), np.array([])
        votes_1, votes_2 = pref_dataset.pref_replay_buffer.meta['votes'], pref_dataset.pref_replay_buffer.meta['votes_2']
        ratio_alpha, ratio_beta = votes_1 / (votes_1 + votes_2), votes_2 / (votes_1 + votes_2)
        votes_alpha = np.maximum(ratio_alpha*8, 1e-6)  # 将小于等于 0 的值替换为一个小正数
        votes_beta = np.maximum(ratio_beta*8, 1e-6)

        for local_epoch_idx in range(cfg.training.online.num_groups):
            ratio = np.stack([np.random.beta(votes_alpha[i], votes_beta[i]) for i in range(votes_alpha.shape[0])])
            local_votes_1 = np.round(cfg.training.online.all_votes / cfg.training.online.num_groups * ratio)
            local_votes_2 = np.round(cfg.training.online.all_votes / cfg.training.online.num_groups - local_votes_1)

            if local_epoch_idx == 0:
                all_votes_1 = local_votes_1.T
                all_votes_2 = local_votes_2.T
            else:
                all_votes_1 = np.concatenate((all_votes_1, local_votes_1.T), axis=0)
                all_votes_2 = np.concatenate((all_votes_2, local_votes_2.T), axis=0)


        sum_votes_1, sum_votes_2 = np.sum(all_votes_1, axis=0, keepdims=True), np.sum(all_votes_2, axis=0, keepdims=True)
        sum_votes_1, sum_votes_2 = np.maximum(sum_votes_1, 1), np.maximum(sum_votes_2, 1)
        all_votes_1, all_votes_2 = np.round(all_votes_1 / sum_votes_1 * (ratio_alpha.T * cfg.training.online.all_votes)), \
                                    np.round(all_votes_2 / sum_votes_2 * (ratio_beta.T * cfg.training.online.all_votes))


        # add noise to votes
        for local_epoch_idx in range(cfg.training.online.num_groups):
            if local_epoch_idx % cfg.training.online.reverse_freq == 0:
                # delta = np.round(cfg.training.online.all_votes / cfg.training.online.num_groups * cfg.training.online.reverse_rate)
                X = stats.truncnorm(-3, 3, loc=cfg.training.online.reverse_rate, scale=cfg.training.online.reverse_rate/3)
                noise_ratio = X.rvs(all_votes_1.shape[1])

                # condiction = (all_votes_1[local_epoch_idx] > all_votes_2[local_epoch_idx])
                all_votes_1[local_epoch_idx], all_votes_2[local_epoch_idx] = all_votes_1[local_epoch_idx] + np.round((all_votes_2[local_epoch_idx] - all_votes_1[local_epoch_idx])*noise_ratio), \
                                                                            all_votes_2[local_epoch_idx] + np.round((all_votes_1[local_epoch_idx] - all_votes_2[local_epoch_idx])*noise_ratio)
                

                all_votes_1[local_epoch_idx] = np.maximum(all_votes_1[local_epoch_idx], 0)
                all_votes_2[local_epoch_idx] = np.maximum(all_votes_2[local_epoch_idx], 0)
    
        init_votes_1 = np.sum(all_votes_1, axis=0, keepdims=True).T / (cfg.training.online.all_votes / 5)
        init_votes_2 = np.sum(all_votes_2, axis=0, keepdims=True).T / (cfg.training.online.all_votes / 5)

        pref_dataset.pref_replay_buffer.meta['votes'] = init_votes_1
        pref_dataset.pref_replay_buffer.meta['votes_2'] = init_votes_2
        pref_dataset.pref_replay_buffer.root['meta']['votes'] = init_votes_1
        pref_dataset.pref_replay_buffer.root['meta']['votes_2'] = init_votes_2

        pref_dataset.set_beta_priori(data_size=150)
        pref_dataset.beta_model.online_update(dataset=pref_dataset.construct_pref_data(), num_epochs=40, warm_up_epochs=5, batch_size=5, lr=1.0e-5)
        pref_dataset.update_beta_priori()

        train_dataloader = DataLoader(pref_dataset, **cfg.dataloader)
        del dataset, expert_dataset, normal_dataset

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            # self.ema_model.double()
            self.ema_model.set_normalizer(normalizer)

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

        # configure test-env runner
        test_env_runner: BaseLowdimRunner
        test_env_runner = hydra.utils.instantiate(
            cfg.task.test_env_runner,
            output_dir=self.output_dir)
        assert isinstance(test_env_runner, BaseLowdimRunner)

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
            for online_epoch_idx in range(cfg.training.online.num_groups):
                print(f"Round {online_epoch_idx + 1} of {cfg.training.online.num_groups} for online training")
                    
                # if online_epoch_idx > 0:
                #     if cfg.training.map.use_map:
                #         ref_policy = comp_policy(self, ref_policy = ref_policy, target_policy = self.model, env = test_env_runner, beta_model = pref_dataset.beta_model)
                #     else:
                #         ref_policy = copy.deepcopy(self.model)

                #     ref_policy.eval() 
                #     for param in ref_policy.parameters():
                #         param.requires_grad = False
                #     ref_policy.to(device)

                if not cfg.training.online.update_history:
                    local_votes_1 = np.array(all_votes_1[online_epoch_idx].T / (all_votes_1[online_epoch_idx].T + \
                                            (all_votes_2[online_epoch_idx].T)), dtype=np.float32).reshape(-1, 1)
                    
                    local_votes_2 = np.array(all_votes_2[online_epoch_idx].T / (all_votes_1[online_epoch_idx].T + \
                                            (all_votes_2[online_epoch_idx].T)), dtype=np.float32).reshape(-1, 1)
                else:
                    local_votes_1 = np.array(all_votes_1[:online_epoch_idx+1].T / (all_votes_1[:online_epoch_idx+1].T + \
                                            all_votes_2[:online_epoch_idx+1].T), dtype=np.float32).reshape(-1, 1)
                    
                    local_votes_2 = np.array(all_votes_2[:online_epoch_idx+1].T / (all_votes_1[:online_epoch_idx+1].T + \
                                            all_votes_2[:online_epoch_idx+1].T), dtype=np.float32).reshape(-1, 1)

                pref_dataset.pref_replay_buffer.meta['votes'] = local_votes_1
                pref_dataset.pref_replay_buffer.meta['votes_2'] = local_votes_2
                pref_dataset.pref_replay_buffer.root['meta']['votes'] = local_votes_1
                pref_dataset.pref_replay_buffer.root['meta']['votes_2'] = local_votes_2

                train_dataloader = DataLoader(pref_dataset, **cfg.dataloader)

                self.optimizer = self.model.get_optimizer(**cfg.optimizer)

                lr_scheduler = get_scheduler(
                    cfg.training.lr_scheduler,
                    optimizer=self.optimizer,
                    num_warmup_steps=cfg.training.lr_warmup_steps,
                    num_training_steps=(
                        len(train_dataloader) * cfg.training.num_epochs) \
                            // cfg.training.gradient_accumulate_every,
                    last_epoch=-1,
                )

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
                                avg_traj_loss = compute_all_traj_loss(replay_buffer = pref_dataset.pref_replay_buffer, \
                                                                      model = self.model, ref_model = ref_policy.model)
                            raw_loss = self.model.compute_loss(batch, ref_model=ref_policy.model, avg_traj_loss = avg_traj_loss)
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

                            is_last_batch = (batch_idx == (len(train_dataloader)*cfg.training.online.num_groups-1))
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
