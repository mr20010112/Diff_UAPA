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
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class PbrlBETLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.policy: BETLowdimPolicy
        self.policy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = self.policy.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        OmegaConf.resolve(cfg)

        # resume training
        if cfg.training.resume:
            ckpt_path = pathlib.Path(cfg.checkpoint_dir)
            if ckpt_path.is_file():
                print(f"Resuming from checkpoint {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)
            self.optimizer = self.policy.get_optimizer(**cfg.optimizer)
            self.global_step = 0
            self.epoch = 0

        device = torch.device(cfg.training.device_gpu)
        ref_policy = copy.deepcopy(self.policy)
        # ref_policy.double()
        ref_policy.eval() 
        for param in ref_policy.parameters():
            param.requires_grad = False
        ref_policy.to(device)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.origin_dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # set normalizer
        normalizer = None
        if cfg.training.enable_normalizer:
            normalizer = dataset.get_normalizer()
        else:
            normalizer = LinearNormalizer()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()

        self.policy.set_normalizer(normalizer)

        # fit action_ae (K-Means)
        self.policy.fit_action_ae(
                normalizer['action'].normalize(
                    dataset.get_all_actions()))

        # configure dataset
        dataset_1: BaseLowdimDataset
        dataset_1 = hydra.utils.instantiate(cfg.task.dataset_1)
        assert isinstance(dataset_1, BaseLowdimDataset)


        # configure dataset
        dataset_2: BaseLowdimDataset
        dataset_2 = hydra.utils.instantiate(cfg.task.dataset_2)
        assert isinstance(dataset_2, BaseLowdimDataset)

        pref_dataset: BaseLowdimDataset
        pref_dataset = hydra.utils.instantiate(cfg.task.pref_dataset, replay_buffer_1=dataset_1.replay_buffer, \
                                               replay_buffer_2=dataset_2.replay_buffer) #cfg.task.perf_dataset

        # cut online groups
        votes_1, votes_2 = pref_dataset.pref_replay_buffer.meta['votes'], pref_dataset.pref_replay_buffer.meta['votes_2']
        var = (votes_1 * votes_2) / (((votes_1 + votes_2) ** 2) * (votes_1 + votes_2 + 1))
        var_flat = var.flatten()
        count = int(len(var_flat) * cfg.training.online.reverse_ratio)
        threshold = np.partition(var_flat, -count)[-count]
        indices = np.where(var_flat >= threshold)[0]
        ratio_1, ratio_2 = votes_1 / (votes_1 + votes_2), votes_2 / (votes_1 + votes_2)

        all_votes_1 = np.array([np.round(ratio_1 * (cfg.training.online.all_votes / cfg.training.online.num_groups)) for _ in range(cfg.training.online.num_groups)])
        all_votes_2 = np.array([np.round(ratio_2 * (cfg.training.online.all_votes / cfg.training.online.num_groups)) for _ in range(cfg.training.online.num_groups)])


        # add noise to votes
        for local_epoch_idx in range(cfg.training.online.num_groups):
            if local_epoch_idx % cfg.training.online.reverse_freq == 0:
                # delta = np.round(cfg.training.online.all_votes / cfg.training.online.num_groups * cfg.training.online.reverse_rate)
                X = stats.truncnorm(-3, 3, loc=cfg.training.online.reverse_rate, scale=cfg.training.online.reverse_rate/3)
                noise_ratio = X.rvs(all_votes_1.shape[1])
                noise_ratio = noise_ratio.reshape(-1, 1)

                # condiction = (all_votes_1[local_epoch_idx] > all_votes_2[local_epoch_idx])
                all_votes_1[local_epoch_idx][indices], all_votes_2[local_epoch_idx][indices] = all_votes_1[local_epoch_idx][indices] + np.round((all_votes_2[local_epoch_idx][indices] - all_votes_1[local_epoch_idx][indices]) * noise_ratio[indices]), \
                                                                            all_votes_2[local_epoch_idx][indices] + np.round((all_votes_1[local_epoch_idx][indices] - all_votes_2[local_epoch_idx][indices]) * noise_ratio[indices])
                

                all_votes_1[local_epoch_idx] = np.maximum(all_votes_1[local_epoch_idx], 0)
                all_votes_2[local_epoch_idx] = np.maximum(all_votes_2[local_epoch_idx], 0)
    
        init_votes_1 = np.sum(all_votes_1, axis=0, keepdims=True).T / (cfg.training.online.all_votes / 5)
        init_votes_2 = np.sum(all_votes_2, axis=0, keepdims=True).T / (cfg.training.online.all_votes / 5)

        pref_dataset.pref_replay_buffer.meta['votes'] = init_votes_1.reshape(-1, 1)
        pref_dataset.pref_replay_buffer.meta['votes_2'] = init_votes_2.reshape(-1, 1)
        pref_dataset.pref_replay_buffer.root['meta']['votes'] = init_votes_1.reshape(-1, 1)
        pref_dataset.pref_replay_buffer.root['meta']['votes_2'] = init_votes_2.reshape(-1, 1)

        pref_dataset.set_beta_priori(data_size=150)
        pref_dataset.beta_model.online_update(dataset=pref_dataset.construct_pref_data(), num_epochs=40, warm_up_epochs=5, batch_size=5, lr=1.0e-5)
        pref_dataset.update_beta_priori()

        train_dataloader = DataLoader(pref_dataset, **cfg.dataloader)
        del dataset, dataset_1, dataset_2

        val_dataset = pref_dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

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
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device_gpu)
        self.policy.to(device)
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
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for online_epoch_idx in range(cfg.training.online.num_groups):
                print(f"Round {online_epoch_idx + 1} of {cfg.training.online.num_groups} for online training")

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

                self.optimizer = self.policy.get_optimizer(**cfg.optimizer)

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
                    train_losses = list()
                    with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            # compute loss
                            # torch.autograd.set_detect_anomaly(True)
                            raw_loss = self.policy.compute_loss(batch, ref_policy=ref_policy)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            # clip grad norm
                            torch.nn.utils.clip_grad_norm_(
                                self.policy.state_prior.parameters(), cfg.training.grad_norm_clip
                            )

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                                lr_scheduler.step()

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
                    self.policy.eval()

                    # run rollout
                    if (self.epoch % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(self.policy)
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
                                    raw_loss = self.policy.compute_loss(batch)
                                    val_losses.append(raw_loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss

                    # run sample on a training batch
                    if (self.epoch % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = train_sampling_batch
                            obs_dict = {'obs': batch['obs']}
                            gt_action = batch['action']
                            
                            result = self.policy.predict_action(obs_dict)
                            if cfg.pred_action_steps_only:
                                pred_action = result['action']
                                start = cfg.n_obs_steps - 1
                                end = start + cfg.n_action_steps
                                gt_action = gt_action[:,start:end]
                            else:
                                pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action[:, :pred_action.shape[1], ...])
                            # log
                            step_log['train_action_mse_error'] = mse.item()
                            # release RAM
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
                    self.policy.train()

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
    workspace = PbrlBETLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
