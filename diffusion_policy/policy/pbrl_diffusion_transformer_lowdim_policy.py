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

class EfficientDiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
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
            map_ratio=1.0,
            bias_reg=1.0,
            guidance_scale=1.0,
            early_stop_threshold=0.01,
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
        self.train_time_samples = train_time_samples
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.gamma = gamma
        self.bias_reg = bias_reg
        self.guidance_scale = guidance_scale
        self.early_stop_threshold = early_stop_threshold
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            reward_model=None,
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
        
        # For early stopping
        prev_trajectory = None
        
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)
            
            # Apply reward guidance if reward model is provided
            if reward_model is not None and self.guidance_scale > 0:
                # Extract observation and action for reward computation
                if self.obs_as_cond:
                    nobs = cond
                    naction = trajectory
                else:
                    naction = trajectory[..., :self.action_dim]
                    nobs = trajectory[..., self.action_dim:]
                
                # Compute reward gradient
                with torch.enable_grad():
                    naction_grad = naction.detach().clone().requires_grad_(True)
                    reward = reward_model(nobs, naction_grad)
                    reward_grad = torch.autograd.grad(reward.sum(), naction_grad)[0]
                
                # Apply guidance
                guidance = self.guidance_scale * reward_grad
                model_output = model_output + guidance
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            
            # Check for early stopping
            if prev_trajectory is not None:
                diff = torch.norm(trajectory - prev_trajectory)
                if diff < self.early_stop_threshold:
                    break
            
            prev_trajectory = trajectory.clone()
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], reward_model=None) -> Dict[str, torch.Tensor]:
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
            reward_model=reward_model,
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

    def compute_loss(self, batch, reward_model: nn.Module, stride=1):
        observations_1 = batch["obs"].to(self.device)
        actions_1 = batch["action"].to(self.device)
        length_1 = batch["length"].to(self.device).detach()
        observations_2 = batch["obs_2"].to(self.device)
        actions_2 = batch["action_2"].to(self.device)
        length_2 = batch["length_2"].to(self.device).detach()
        bsz = observations_1.shape[0]
        
        batch_data = {
            'obs': torch.tensor(observations_1, device=self.device),
            'action': torch.tensor(actions_1, device=self.device),
        }

        batch_data_2 = {
            'obs': torch.tensor(observations_2, device=self.device),
            'action': torch.tensor(actions_2, device=self.device),
        }

        nbatch_1 = self.normalizer.normalize(batch_data)
        nbatch_2 = self.normalizer.normalize(batch_data_2)

        obs_1 = nbatch_1['obs']
        action_1 = nbatch_1['action']
        obs_2 = nbatch_2['obs']
        action_2 = nbatch_2['action']
        
        stride = stride

        obs_1 = slice_episode(obs_1, horizon=self.horizon, stride=stride)
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=stride)
        obs_2 = slice_episode(obs_2, horizon=self.horizon, stride=stride)
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=stride)

        
        loss = 0

        for _ in range(self.train_time_samples):
            timesteps_1 = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
            timesteps_2 = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()

            diffusion_loss = 0
            reward_loss = 0

            for i in range(len(obs_1)):
                obs_slide = obs_1[i]
                action_slide = action_1[i]
                trajectory = action_slide
                cond = None
                if self.obs_as_cond:
                    cond = obs_slide[:, :self.n_obs_steps, :]
                    cond = cond.to(self.device)
                    if self.pred_action_steps_only:
                        trajectory = action_slide[:, -self.n_action_steps:]
                else:
                    trajectory = torch.cat([action_slide, obs_slide], dim=-1)
                trajectory = trajectory.to(self.device)
                condition_mask = self.mask_generator(trajectory.shape).to(self.device)
                noise = torch.randn(trajectory.shape, device=self.device)
                noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps_1)

                loss_mask = ~condition_mask
                noisy_trajectory[condition_mask] = trajectory[condition_mask]

                pred = self.model(noisy_trajectory, timesteps_1, cond)

                pred_type = self.noise_scheduler.config.prediction_type
                target = noise if pred_type == 'epsilon' else trajectory

                mask = (self.horizon + (i-1)*stride) <= length_1
                mask = mask.int()

                # Diffusion loss
                slice_loss = torch.norm((pred - noise) * loss_mask.type(pred.dtype), dim=-1) ** 2
                
                # Apply discount factor
                discount_factors = (self.gamma ** (i*self.horizon + torch.arange(0, self.horizon, device=self.device))).reshape(1, -1)
                diffusion_loss += (slice_loss * mask) * discount_factors

                # Reward-guided loss
                if reward_model is not None:
                    # Extract observation and action for reward computation
                    if self.obs_as_cond:
                        nobs_reward = cond
                        naction_reward = trajectory
                    else:
                        naction_reward = trajectory[..., :self.action_dim]
                        nobs_reward = trajectory[..., self.action_dim:]
                    
                    # Compute reward
                    rewards = reward_model.forward(nobs_reward, naction_reward)
                    reward_loss -= torch.mean(rewards * mask)  # Negative because we want to maximize reward

            for i in range(len(obs_2)):
                obs_slide = obs_2[i]
                action_slide = action_2[i]
                trajectory = action_slide
                cond = None
                if self.obs_as_cond:
                    cond = obs_slide[:, :self.n_obs_steps, :]
                    cond = cond.to(self.device)
                    if self.pred_action_steps_only:
                        trajectory = action_slide[:, -self.n_action_steps:]
                else:
                    trajectory = torch.cat([action_slide, obs_slide], dim=-1)
                trajectory = trajectory.to(self.device)
                condition_mask = self.mask_generator(trajectory.shape).to(self.device)
                noise = torch.randn(trajectory.shape, device=self.device)
                noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps_2)

                loss_mask = ~condition_mask
                noisy_trajectory[condition_mask] = trajectory[condition_mask]

                pred = self.model(noisy_trajectory, timesteps_2, cond)

                pred_type = self.noise_scheduler.config.prediction_type
                target = noise if pred_type == 'epsilon' else trajectory

                mask = (self.horizon + (i-1)*stride) <= length_2
                mask = mask.int()

                # Diffusion loss
                slice_loss = torch.norm((pred - noise) * loss_mask.type(pred.dtype), dim=-1) ** 2
                
                # Apply discount factor
                discount_factors = (self.gamma ** (i*self.horizon + torch.arange(0, self.horizon, device=self.device))).reshape(1, -1)
                diffusion_loss += (slice_loss * mask) * discount_factors
                
                # Reward-guided loss
                if reward_model is not None:
                    # Extract observation and action for reward computation
                    if self.obs_as_cond:
                        nobs_reward = cond
                        naction_reward = trajectory
                    else:
                        naction_reward = trajectory[..., :self.action_dim]
                        nobs_reward = trajectory[..., self.action_dim:]
                    
                    # Compute reward
                    rewards = reward_model.forward(nobs_reward, naction_reward)
                    reward_loss -= torch.mean(rewards * mask)  # Negative because we want to maximize reward

            diffusion_loss = torch.sum(diffusion_loss, dim=-1)
            
            # Combine losses
            combined_loss = diffusion_loss + self.bias_reg * reward_loss
            loss += combined_loss
            
        return torch.mean(loss)
    
    def efficient_sample_with_reward_guidance(self, obs_dict, reward_model, num_samples=5):
        """
        Sample multiple trajectories and select the one with highest reward
        """
        best_action = None
        best_reward = float('-inf')
        
        for _ in range(num_samples):
            # Sample a trajectory
            result = self.predict_action(obs_dict, reward_model)
            action = result['action']
            
            # Compute reward for this trajectory
            nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
            naction = self.normalizer['action'].normalize(action)
            
            with torch.no_grad():
                reward = reward_model(nobs, naction).mean().item()
            
            # Update best action if this one has higher reward
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        # Return the best action and its predicted reward
        return {
            'action': best_action,
            'predicted_reward': best_reward
        }