from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.realrobot_image_obs_encoder import RealRobotImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.slice import slice_episode


class DiffusionRealRobotPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: RealRobotImageObsEncoder,
            gamma,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            use_map=False,
            beta=1.0,
            map_ratio=1.0,
            bias_reg=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.gamma = gamma
        self.beta = beta
        self.bias_reg = bias_reg
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
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
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

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
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, ref_model: ConditionalUnet1D=None, avg_traj_loss=0.0, stride=1):

        for param in ref_model.parameters():
            param.requires_grad = False

        batch_size = batch['action'].shape[0]
        horizon = self.horizon
        
        def process_batch_data(obs, action):
            nobs = self.normalizer.normalize(obs)
            nactions = self.normalizer['action'].normalize(action)
            
            local_cond = None
            global_cond = None
            
            if self.obs_as_global_cond:
                # reshape B, T, ... to B*T
                this_nobs = dict_apply(nobs, 
                    lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
                trajectory = nactions
                cond_data = trajectory
            else:
                # reshape B, T, ... to B*T
                this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)
                # reshape back to B, T, Do
                nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                cond_data = torch.cat([nactions, nobs_features], dim=-1)
                trajectory = cond_data.detach()
                
            condition_mask = self.mask_generator(trajectory.shape)
            
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (batch_size,), device=trajectory.device
            ).long()
            
            noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
            
            loss_mask = ~condition_mask
            
            noisy_trajectory[condition_mask] = cond_data[condition_mask]
            
            pred = self.model(noisy_trajectory, timesteps, 
                local_cond=local_cond, global_cond=global_cond)
            
            pred_type = self.noise_scheduler.config.prediction_type 
            if pred_type == 'epsilon':
                target = noise
            elif pred_type == 'sample':
                target = trajectory
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")
            
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
            loss = reduce(loss, 'b t ... -> b t (...)', 'mean')
            
            traj_loss = torch.sum(loss, dim=-1)
            
            del noise, noisy_trajectory, pred, target, loss, loss_mask
            torch.cuda.empty_cache()
            
            return traj_loss
        
        traj_loss = process_batch_data(batch['obs'], batch['action'])
        
        traj_loss_2 = process_batch_data(batch['obs_2'], batch['action_2'])
        
        total_actions = batch['action'].size(0) + batch['action_2'].size(0)
        scale_factor = self.beta * self.noise_scheduler.config.num_train_timesteps
        
        immitation_loss = torch.mean(traj_loss + traj_loss_2) / (self.horizon * total_actions)
        
        traj_loss = -scale_factor * traj_loss
        traj_loss_2 = -scale_factor * traj_loss_2
        
        diff_term = traj_loss - traj_loss_2 + immitation_loss
        mle_loss = -F.logsigmoid(diff_term)
        
        return torch.mean(mle_loss)
