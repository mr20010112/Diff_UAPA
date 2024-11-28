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
from torch.cuda.amp.grad_scaler import GradScaler
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
            beta,
            train_time_samples,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            map_ratio=0.1,
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
        self.map_ratio = map_ratio
        self.kwargs = kwargs

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

    def compute_loss(self, batch, ref_model: TransformerForDiffusion, avg_traj_loss=0):
        ref_model = ref_model.to(self.device)

        observations_1 = batch["obs"]
        actions_1 = batch["action"]
        votes_1 = batch["votes"]
        observations_2 = batch["obs_2"]
        actions_2 = batch["action_2"]
        votes_2 = batch["votes_2"]
        beta_priori = batch["beta_priori"]
        beta_priori_2 = batch["beta_priori_2"]
        avg_traj_loss = torch.tensor(avg_traj_loss, device=self.device, dtype=torch.float64)
        # length = batch["length"]
        # length_2 = batch["length_2"]

        batch = {
            'obs': torch.stack([observations_1, observations_2], dim=0).to(self.device),
            'action': torch.stack([actions_1, actions_2], dim=0).to(self.device),
        }

        nbatch = self.normalizer.normalize(batch)
        # vote_max = torch.max(torch.stack([votes_1, votes_2], dim=0))
        # votes_1, votes_2 = votes / vote_max, votes_2 / vote_max
        # votes_1 = torch.zeros(votes_1.shape, device=self.device, dtype=torch.float32)
        # votes_2 = torch.zeros(votes_2.shape, device=self.device, dtype=torch.float32)

        obs_1, obs_2 = nbatch['obs'][0], nbatch['obs'][1]
        action_1, action_2 = nbatch['action'][0], nbatch['action'][1]

        obs_1 = slice_episode(obs_1, horizon=self.horizon, stride=self.horizon)
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=self.horizon)
        obs_2 = slice_episode(obs_2, horizon=self.horizon, stride=self.horizon)
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=self.horizon)

        bsz = obs_1[0].shape[0]
        loss = 0

        with autocast():
            for _ in range(self.train_time_samples[0]):
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()

                traj_loss_1, traj_loss_2 = 0, 0
                condition_mask = []

                for i in range(len(obs_1)):
                    obs_1_slide = obs_1[i]
                    action_1_slide = action_1[i]

                    trajectory_1 = action_1_slide

                    if self.obs_as_cond:
                        cond_1 = obs_1_slide[:, :self.n_obs_steps, :]
                        cond_1 = cond_1.to(self.device)
                        if self.pred_action_steps_only:
                            trajectory_1 = action_1_slide[:, -self.n_action_steps:]
                    else:
                        trajectory_1 = torch.cat([action_1_slide, obs_1_slide], dim=-1)
                    trajectory_1 = trajectory_1.to(self.device)
                    condition_mask_1 = self.mask_generator(trajectory_1.shape).to(self.device)
                    condition_mask.append(condition_mask_1)
                    noise_1 = torch.randn(trajectory_1.shape, device=self.device)
                    noisy_trajectory_1 = self.noise_scheduler.add_noise(trajectory_1, noise_1, timesteps)

                    loss_mask_1 = ~condition_mask_1
                    noisy_trajectory_1[condition_mask_1] = trajectory_1[condition_mask_1]

                    ref_pred_1 = ref_model(noisy_trajectory_1, timesteps, cond_1).detach()
                    pred_1 = self.model(noisy_trajectory_1, timesteps, cond_1)

                    pred_type_1 = self.noise_scheduler.config.prediction_type
                    target = noise_1 if pred_type_1 == 'epsilon' else trajectory_1

                    slice_loss_1 = torch.norm((noise_1 - pred_1) * loss_mask_1.type(pred_1.dtype), dim=-1) ** 2 \
                                - torch.norm((noise_1 - ref_pred_1) * loss_mask_1.type(ref_pred_1.dtype), dim=-1) ** 2
                    # slice_loss_1 = F.mse_loss(pred_1 * loss_mask_1.type(pred_1.dtype), target * loss_mask_1.type(target.dtype), reduction='none')

                    traj_loss_1 += (slice_loss_1) * (self.gamma ** (i*self.horizon + torch.arange(1, self.horizon + 1, device=self.device)))

                for i in range(len(obs_2)):
                    obs_2_slide = obs_2[i]
                    action_2_slide = action_2[i]

                    trajectory_2 = action_2_slide
                    if self.obs_as_cond:
                        cond_2 = obs_2_slide[:, :self.n_obs_steps, :]
                        cond_2 = cond_2.to(self.device)
                        if self.pred_action_steps_only:
                            trajectory_2 = action_2_slide[:, -self.n_action_steps:]
                    else:
                        trajectory_2 = torch.cat([action_2_slide, obs_2_slide], dim=-1)
                    trajectory_2 = trajectory_2.to(self.device)
                    condition_mask_2 = condition_mask[i]
                    noise_2 = torch.randn(trajectory_2.shape, device=self.device)
                    noisy_trajectory_2 = self.noise_scheduler.add_noise(trajectory_2, noise_2, timesteps)

                    loss_mask_2 = ~condition_mask_2
                    noisy_trajectory_2[condition_mask_2] = trajectory_2[condition_mask_2]

                    ref_pred_2 = ref_model(noisy_trajectory_2, timesteps, cond_2).detach()
                    pred_2 = self.model(noisy_trajectory_2, timesteps, cond_2)

                    pred_type_2 = self.noise_scheduler.config.prediction_type
                    target = noise_2 if pred_type_2 == 'epsilon' else trajectory_2

                    slice_loss_2 = torch.norm((noise_2 - pred_2) * loss_mask_2.type(pred_2.dtype), dim=-1) ** 2\
                                - torch.norm((noise_2 - ref_pred_2) * loss_mask_2.type(ref_pred_2.dtype), dim=-1) ** 2
                    # slice_loss_2 = F.mse_loss(pred_2 * loss_mask_2.type(pred_2.dtype), target * loss_mask_2.type(target.dtype), reduction='none')

                    traj_loss_2 += (slice_loss_2) * (self.gamma ** (i*self.horizon + torch.arange(1, self.horizon + 1, device=self.device)))

                traj_loss_1 = torch.sum(traj_loss_1, dim=-1)
                traj_loss_2 = torch.sum(traj_loss_2, dim=-1)

                traj_loss_1 = traj_loss_1.to(torch.float64)
                traj_loss_2 = traj_loss_2.to(torch.float64)

                term = torch.ones(timesteps.shape, device=self.device, dtype=torch.float64)
                # beta_dist = Beta(beta_priori[:, 0], beta_priori[:, 1])
                # beta_dist_2 = Beta(beta_priori_2[:, 0], beta_priori_2[:, 1])

                # max_idx_1 = (beta_priori[:, 0] - 1) / (beta_priori[:, 0] + beta_priori[:, 1] - 2)
                # max_idx_2 = (beta_priori_2[:, 0] - 1) / (beta_priori_2[:, 0] + beta_priori_2[:, 1] - 2)

                mle_loss_1 = -F.logsigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_1 - traj_loss_2)))
                mle_loss_2 = -F.logsigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_2 - traj_loss_1)))

                # map_loss_1 = -F.logsigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_1 - avg_traj_loss))) \
                #             + F.softplus(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_1 - avg_traj_loss))) \
                #             - beta_dist.log_prob(torch.clamp(torch.sigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * \
                #                 (term * (traj_loss_1 - avg_traj_loss))), min=1e-4, max=1-1e-4)) + beta_dist.log_prob(torch.clamp(max_idx_1, min=1e-4, max=1-1e-4))
                            
                # map_loss_2 = -F.logsigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_2 - avg_traj_loss))) \
                #             + F.softplus(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_2 - avg_traj_loss))) \
                #             - beta_dist_2.log_prob(torch.clamp(torch.sigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * \
                #                 (term * (traj_loss_2 - avg_traj_loss))), min=1e-4, max=1-1e-4)) + beta_dist_2.log_prob(torch.clamp(max_idx_2, min=1e-4, max=1-1e-4))


                # loss_1 = -F.logsigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_1 - avg_traj_loss)))
                # loss_2 = F.softplus(-self.beta * self.noise_scheduler.config.num_train_timesteps * (term * (traj_loss_1 - avg_traj_loss)))
                # loss_3 = - beta_dist.log_prob(torch.clamp(torch.sigmoid(-self.beta * self.noise_scheduler.config.num_train_timesteps * \
                #                 (term * (traj_loss_1 - avg_traj_loss))), min=1e-4, max=1-1e-4)) + beta_dist.log_prob(torch.clamp(max_idx_1, min=1e-4, max=1-1e-4))
                

                # map_loss = [torch.mean(loss_1), torch.mean(loss_2), torch.mean(loss_3)]

                loss += (votes_1.to(self.device) * mle_loss_1 + votes_2.to(self.device) * mle_loss_2) / (2 * self.train_time_samples[0]) #+ self.map_ratio * (map_loss_1 + map_loss_2)

        return torch.mean(loss)