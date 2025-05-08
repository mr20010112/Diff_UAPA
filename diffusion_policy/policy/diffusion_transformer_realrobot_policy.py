import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from typing import Dict, Tuple
# from .detr.main import build_ACT_model_and_optimizer, build_ACT_model_and_optimizer_no_extra_args, build_CNNMLP_model_and_optimizer
# from detr.main import build_ACT_model_and_optimizer, build_ACT_model_and_optimizer_no_extra_args, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

from collections import OrderedDict
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
# from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

class DiffusionTransformerRealRobotPolicy(BaseImagePolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_kp,
            feature_dimension,
            camera_names=None,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            **kwargs):
        super().__init__()

        self.normalizer = LinearNormalizer()
        self.camera_names = camera_names
        self.observation_horizon = n_obs_steps ### TODO TODO TODO DO THIS
        self.action_horizon = n_action_steps # apply chunk size
        self.prediction_horizon = horizon # chunk size
        self.num_inference_timesteps = num_inference_steps


        self.num_kp = num_kp
        self.feature_dimension = feature_dimension
        self.ac_dim = action_dim # 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14 # camera features and proprio

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

        # setup noise scheduler
        self.noise_scheduler = noise_scheduler

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.nets['policy']['noise_pred_net'].configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas)) #to do


    def predict_action(self, qpos, image) -> Dict[str, torch.Tensor]:
        To = self.observation_horizon
        Ta = self.action_horizon
        Tp = self.prediction_horizon
        action_dim = self.ac_dim
        B = qpos.shape[0]

        nets = self.nets

        all_features = []
        for cam_id in range(len(self.camera_names)):
            cam_image = image[:, cam_id]
            cam_features = nets['policy']['backbones'][cam_id](cam_image)
            pool_features = nets['policy']['pools'][cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets['policy']['linears'][cam_id](pool_features)
            all_features.append(out_features)

        obs_cond = torch.cat(all_features + [qpos], dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=obs_cond.device)
        naction = noisy_action

        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets['policy']['noise_pred_net'](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction
    
    def compute_loss(self, batch):
        qpos = batch['qpos']
        image = batch['image']
        actions = batch['actions']

        B = qpos.shape[0]
        if actions is not None: # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)

            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            return loss_dict
        
    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        return status
