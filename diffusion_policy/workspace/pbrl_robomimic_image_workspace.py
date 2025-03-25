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
import robomimic.utils.torch_utils as TorchUtils

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_image_bc_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler


OmegaConf.register_new_resolver("eval", eval, replace=True)

class PbrlRobomimicImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: RobomimicImagePolicy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = self.model.get_optimizer()

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        OmegaConf.resolve(cfg) #add

        # resume training
        if cfg.training.resume:
            ckpt_path = pathlib.Path(cfg.checkpoint_dir)
            if ckpt_path.is_file():
                print(f"Resuming from checkpoint {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)
            self.optimizer = self.model.get_optimizer()
            self.global_step = 0
            self.epoch = 0

        device = torch.device(cfg.training.device_gpu)
        ref_model = copy.deepcopy(self.model)
        # ref_model.double()
        ref_model.train() #eval() 
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.to(device)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.origin_dataset)
        assert isinstance(dataset, BaseImageDataset)
        # train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        self.model.set_normalizer(normalizer)

        # # configure dataset
        # dataset_1: BaseImageDataset
        # dataset_1 = hydra.utils.instantiate(cfg.task.dataset_1)
        # #device = torch.device(cfg.training.device_cpu)
        # assert isinstance(dataset_1, BaseImageDataset)

        # # configure dataset
        # dataset_2: BaseImageDataset
        # dataset_2 = hydra.utils.instantiate(cfg.task.dataset_2)
        # # expert_normalizer = normal_dataset.get_normalizer()
        # assert isinstance(dataset_2, BaseImageDataset)

        pref_dataset: BaseImageDataset
        pref_dataset = hydra.utils.instantiate(cfg.task.pref_dataset, dataset_1=dataset, dataset_2=dataset) #cfg.task.perf_dataset

        # cut online groups
        votes_1, votes_2 = pref_dataset.pref_replay_buffer.meta['votes'], pref_dataset.pref_replay_buffer.meta['votes_2']

        # normalization votes
        votes_concat = np.concatenate((votes_1, votes_2), axis=1)
        votes_mean = votes_concat.mean()
        votes_std = votes_concat.std()

        votes_1 = np.clip(votes_1, votes_mean - 3 * votes_std, votes_mean + 3 * votes_std)
        votes_2 = np.clip(votes_2, votes_mean - 3 * votes_std, votes_mean + 3 * votes_std)

        votes_concat = np.concatenate((votes_1, votes_2), axis=1)
        votes_mean = votes_concat.mean()
        votes_std = votes_concat.std()

        votes_1 = (votes_1 - votes_mean) / (votes_std + 1e-8)
        votes_2 = (votes_2 - votes_mean) / (votes_std + 1e-8)

        votes_concat = np.concatenate((votes_1, votes_2), axis=1)
        votes_min = votes_concat.min()
        votes_max = votes_concat.max()

        votes_1_norm = (votes_1 - votes_min) / (votes_max - votes_min + 1e-8)
        votes_2_norm = (votes_2 - votes_min) / (votes_max - votes_min + 1e-8)

        scale_factor = 10
        votes_1 = votes_1_norm * scale_factor
        votes_2 = votes_2_norm * scale_factor

        #select uncertain samples
        var = (votes_1 * votes_2) / (((votes_1 + votes_2 + 1e-6) ** 2) * (votes_1 + votes_2 + 1))
        mask = (votes_1 + votes_2) != 0
        var_masked = var[mask]
        var_flat = var_masked.flatten()
        count = int(len(var_flat) * cfg.training.online.reverse_ratio)
        threshold = np.partition(var_flat, -count)[-count]
        masked_indices = np.where(var_flat >= threshold)[0]
        original_indices = np.where(mask.flatten())[0]
        indices = original_indices[masked_indices]
        ratio_1, ratio_2 = votes_1 / (votes_1 + votes_2), votes_2 / (votes_1 + votes_2)

        all_votes_1 = np.array([np.round(ratio_1 * (cfg.training.online.all_votes / cfg.training.online.num_groups)) for _ in range(cfg.training.online.num_groups)])
        all_votes_2 = np.array([np.round(ratio_2 * (cfg.training.online.all_votes / cfg.training.online.num_groups)) for _ in range(cfg.training.online.num_groups)])


        # add noise to votes
        for local_epoch_idx in range(cfg.training.online.num_groups):
            if local_epoch_idx % cfg.training.online.reverse_freq == 0:
                X = stats.truncnorm(-3, 3, loc=cfg.training.online.reverse_rate, scale=cfg.training.online.reverse_rate/3)
                noise_ratio = X.rvs(all_votes_1.shape[1])
                noise_ratio = noise_ratio.reshape(-1, 1)

                # condiction = (all_votes_1[local_epoch_idx] > all_votes_2[local_epoch_idx])
                all_votes_1[local_epoch_idx][indices], all_votes_2[local_epoch_idx][indices] = \
                    all_votes_1[local_epoch_idx][indices] + np.round((all_votes_2[local_epoch_idx][indices] - all_votes_1[local_epoch_idx][indices]) * noise_ratio[indices]), \
                    all_votes_2[local_epoch_idx][indices] + np.round((all_votes_1[local_epoch_idx][indices] - all_votes_2[local_epoch_idx][indices]) * noise_ratio[indices])
                

                all_votes_1[local_epoch_idx] = np.maximum(all_votes_1[local_epoch_idx], 0)
                all_votes_2[local_epoch_idx] = np.maximum(all_votes_2[local_epoch_idx], 0)

        train_dataloader = DataLoader(pref_dataset, **cfg.dataloader)

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

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

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
        self.model.to(device)
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

                local_votes_1 = np.array(all_votes_1[online_epoch_idx].T / (all_votes_1[online_epoch_idx].T + \
                                        (all_votes_2[online_epoch_idx].T)), dtype=np.float32).reshape(-1, 1)
                
                local_votes_2 = np.array(all_votes_2[online_epoch_idx].T / (all_votes_1[online_epoch_idx].T + \
                                        (all_votes_2[online_epoch_idx].T)), dtype=np.float32).reshape(-1, 1)

                pref_dataset.pref_replay_buffer.meta['votes'] = local_votes_1
                pref_dataset.pref_replay_buffer.meta['votes_2'] = local_votes_2
                pref_dataset.pref_replay_buffer.root['meta']['votes'] = local_votes_1
                pref_dataset.pref_replay_buffer.root['meta']['votes_2'] = local_votes_2

                train_dataloader = DataLoader(pref_dataset, **cfg.dataloader)

                self.optimizer = self.model.get_optimizer()

                obs_keys = dataset.lowdim_keys + dataset.rgb_keys
                self.model.train()
                for param in self.model.parameters():
                    param.requires_grad = True

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

                            loss = self.model.compute_loss(batch, ref_policy=ref_model, obs_keys=obs_keys)
                            loss = loss / cfg.training.gradient_accumulate_every
                            policy_grad_norms = TorchUtils.backprop_for_loss(
                                net=self.model.model.nets["policy"],
                                optim=self.model.model.optimizers["policy"],
                                loss=loss,
                            )

                            # # clip grad norm
                            # torch.nn.utils.clip_grad_norm_(
                            #     self.model.parameters(), cfg.training.grad_norm_clip
                            # )

                            # # step optimizer
                            # if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            #     self.optimizer.step()
                            #     self.optimizer.zero_grad(set_to_none=True)
                            #     lr_scheduler.step()

                            step_log = {
                                'train_loss': loss,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
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
                    self.model.eval()
                    for param in self.model.parameters():
                        param.requires_grad = False

                    # run rollout
                    if (self.epoch % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(self.model)
                        # log all
                        step_log.update(runner_log)

                    # run diffusion sampling on a training batch
                    if (self.epoch % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = {key:batch[key] for key in obs_keys}
                            gt_action = batch["action"]
                            obs_dict_2 = {key:batch[f'{key}_2'] for key in obs_keys}
                            get_action_2 = batch["action_2"]
                            T = gt_action.shape[1]
                            T_2 = get_action_2.shape[1]

                            pred_actions = list()
                            pred_actions_2 = list()
                            self.model.reset()
                            for i in range(T):
                                result = self.model.predict_action(
                                    dict_apply(obs_dict, lambda x: x[:,[i]])
                                )
                                pred_actions.append(result['action'])
                            pred_actions = torch.cat(pred_actions, dim=1)
                            mse = torch.nn.functional.mse_loss(pred_actions, gt_action)

                            for i in range(T_2):
                                result = self.model.predict_action(
                                    dict_apply(obs_dict_2, lambda x: x[:,[i]])
                                )
                                pred_actions_2.append(result['action'])
                            pred_actions_2 = torch.cat(pred_actions_2, dim=1)
                            mse_2 = torch.nn.functional.mse_loss(pred_actions_2, get_action_2)
                            step_log['train_action_mse_error'] = (mse.item() + mse_2.item())*0.5
                            del batch, obs_dict, gt_action, result, pred_actions, mse
                            del obs_dict_2, get_action_2, pred_actions_2, mse_2


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
                    self.model.train()
                    self.model.model.nets.training = False
                    for param in self.model.parameters():
                        param.requires_grad = True

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
    workspace = PbrlRobomimicImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
