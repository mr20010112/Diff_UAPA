from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mpmath import mp
from einops import rearrange, reduce
from torch.distributions import Beta
from scipy.stats import beta
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp.autocast_mode import autocast
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

from diffusion_policy.model.common.slice import slice_episode

class DiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            gamma,
            train_time_samples,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            use_map=False,
            beta=1.0,
            map_ratio=1.0,
            lambda_reg=0.8,
            bias_reg=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model

        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.train_time_samples = train_time_samples,
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.gamma = gamma
        self.beta = beta
        self.bias_reg = bias_reg
        self.kwargs = kwargs
        self.lambda_reg = lambda_reg

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch, ref_model: TransformerForDiffusion, avg_traj_loss=0.0, stride=1):

        for param in ref_model.parameters():
            param.requires_grad = False


        observations_1 = batch["obs"].to(self.device)
        actions_1 = batch["action"].to(self.device)
        votes_1 = batch["votes"].to(self.device)
        length_1 = batch["length"].to(self.device).detach()
        observations_2 = batch["obs_2"].to(self.device)
        actions_2 = batch["action_2"].to(self.device)
        votes_2 = batch["votes_2"].to(self.device)
        length_2 = batch["length_2"].to(self.device).detach()
        save_avg_traj_loss = torch.tensor(avg_traj_loss, device=self.device).detach()

        threshold = 1e-2
        diff = torch.abs(votes_1 - votes_2)
        condition_1 = (votes_1 > votes_2) & (diff >= threshold)
        condition_2 = (votes_1 < votes_2) & (diff >= threshold)

        votes_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_1 = torch.squeeze(votes_1, dim=-1).detach()
        votes_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_2 = torch.squeeze(votes_2, dim=-1).detach()

        mask = condition_2.squeeze(-1)
        actions_1[mask], actions_2[mask] = actions_2[mask], actions_1[mask]
        observations_1[mask], observations_2[mask] = observations_2[mask], observations_1[mask]
        length_1[mask], length_2[mask] = length_2[mask], length_1[mask]

        batch_1 = {'obs': observations_1, 'action': actions_1}
        batch_2 = {'obs': observations_2, 'action': actions_2}

        nbatch_1 = self.normalizer.normalize(batch_1)
        nbatch_2 = self.normalizer.normalize(batch_2)

        obs_1 = nbatch_1['obs']
        action_1 = nbatch_1['action']
        obs_2 = nbatch_2['obs']
        action_2 = nbatch_2['action']

        obs_1 = slice_episode(obs_1, horizon=self.horizon, stride=stride)
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=stride)
        obs_2 = slice_episode(obs_2, horizon=self.horizon, stride=stride)
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=stride)

        bsz = obs_1[0].shape[0]
        loss = 0

        delta = torch.zeros(bsz, device=self.device, requires_grad=False)

        for _ in range(self.train_time_samples[0]):
            timesteps_1 = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
            timesteps_2 = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()

            traj_loss_1, traj_loss_2, imitation_loss, avg_traj_loss = 0, 0, 0, save_avg_traj_loss

            # 计算轨迹损失 traj_loss_1
            for i in range(len(obs_1)):
                obs_1_slide = obs_1[i]
                action_1_slide = action_1[i]
                trajectory_1 = action_1_slide
                cond_1 = None
                if self.obs_as_cond:
                    cond_1 = obs_1_slide[:, :self.n_obs_steps, :].to(self.device)
                    if self.pred_action_steps_only:
                        trajectory_1 = action_1_slide[:, -self.n_action_steps:]
                else:
                    trajectory_1 = torch.cat([action_1_slide, obs_1_slide], dim=-1)
                trajectory_1 = trajectory_1.to(self.device)
                condition_mask_1 = self.mask_generator(trajectory_1.shape).to(self.device)
                noise_1 = torch.randn(trajectory_1.shape, device=self.device)
                noisy_trajectory_1 = self.noise_scheduler.add_noise(trajectory_1, noise_1, timesteps_1)
                noisy_trajectory_1[condition_mask_1] = trajectory_1[condition_mask_1]

                pred_1 = self.model(noisy_trajectory_1, timesteps_1, cond_1)
                target = noise_1 if self.noise_scheduler.config.prediction_type == 'epsilon' else trajectory_1

                mask_1 = (self.horizon + (i-1)*stride) <= length_1
                mask_1 = mask_1.int()
                slice_loss_1 = torch.norm((pred_1 - noise_1) * (~condition_mask_1).type(pred_1.dtype), dim=-1) ** 2
                traj_loss_1 += slice_loss_1 * mask_1

            for i in range(len(obs_2)):
                obs_2_slide = obs_2[i]
                action_2_slide = action_2[i]
                trajectory_2 = action_2_slide
                cond_2 = None
                if self.obs_as_cond:
                    cond_2 = obs_2_slide[:, :self.n_obs_steps, :].to(self.device)
                    if self.pred_action_steps_only:
                        trajectory_2 = action_2_slide[:, -self.n_action_steps:]
                else:
                    trajectory_2 = torch.cat([action_2_slide, obs_2_slide], dim=-1)
                trajectory_2 = trajectory_2.to(self.device)
                condition_mask_2 = self.mask_generator(trajectory_2.shape).to(self.device)
                noise_2 = torch.randn(trajectory_2.shape, device=self.device)
                noisy_trajectory_2 = self.noise_scheduler.add_noise(trajectory_2, noise_2, timesteps_2)
                noisy_trajectory_2[condition_mask_2] = trajectory_2[condition_mask_2]

                pred_2 = self.model(noisy_trajectory_2, timesteps_2, cond_2)
                target = noise_2 if self.noise_scheduler.config.prediction_type == 'epsilon' else trajectory_2

                mask_2 = (self.horizon + (i-1)*stride) <= length_2
                mask_2 = mask_2.int()
                slice_loss_2 = torch.norm((pred_2 - noise_2) * (~condition_mask_2).type(pred_2.dtype), dim=-1) ** 2
                traj_loss_2 += slice_loss_2 * mask_2

            traj_loss_1 = torch.sum(traj_loss_1, dim=-1)
            traj_loss_2 = torch.sum(traj_loss_2, dim=-1)
            imitation_loss = (traj_loss_1 + traj_loss_2)

            traj_loss_1 = -self.beta * self.noise_scheduler.config.num_train_timesteps * traj_loss_1
            traj_loss_2 = -self.beta * self.noise_scheduler.config.num_train_timesteps * traj_loss_2
            avg_traj_loss = -self.beta * self.noise_scheduler.config.num_train_timesteps * avg_traj_loss
            imitation_loss = -torch.mean(imitation_loss) / (self.horizon * (len(obs_1) + len(obs_2)) * 4)

            diff_loss = traj_loss_1 - traj_loss_2 
            delta = torch.max(
                torch.log(torch.tensor(1 / self.lambda_reg - 1, device=self.device)) - diff_loss,
                torch.zeros_like(diff_loss)
            ).detach()

            preference_term = diff_loss + delta 
            mle_loss = -F.logsigmoid(preference_term) 

            l1_reg = self.lambda_reg * torch.sum(torch.abs(delta))

            total_loss = torch.mean(mle_loss) + l1_reg
            loss += total_loss / self.train_time_samples[0]

        return torch.mean(loss)