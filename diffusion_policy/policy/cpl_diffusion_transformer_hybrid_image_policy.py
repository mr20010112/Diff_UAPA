from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.slice import slice_episode



class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            #policy
            gamma,
            train_time_samples,            
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            #policy
            use_map=False,
            beta=1.0,
            map_ratio=1.0,
            bias_reg=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.train_time_samples = train_time_samples
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
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
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
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
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch, obs_keys, ref_model: TransformerForDiffusion=None, avg_traj_loss=0.0, stride=1):
        
        observations_1 = {key:batch[key].to(self.device) for key in obs_keys}
        actions_1 = batch["action"].to(self.device)
        votes_1 = batch["votes"].to(self.device)
        length_1 = batch["length"].to(self.device).detach()
        observations_2 = {key:batch[f'{key}_2'].to(self.device) for key in obs_keys}
        actions_2 = batch["action_2"].to(self.device)
        votes_2 = batch["votes_2"].to(self.device)
        length_2 = batch["length_2"].to(self.device).detach()

        threshold = 1e-2
        diff = torch.abs(votes_1 - votes_2)
        condition_1 = (votes_1 > votes_2) & (diff >= threshold)  # votes_1 > votes_2 and diff >= threshold
        condition_2 = (votes_1 < votes_2) & (diff >= threshold)  # votes_1 < votes_2 and diff >= threshold

        votes_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_1 = torch.squeeze(votes_1, dim=-1).detach()
        votes_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_2 = torch.squeeze(votes_2, dim=-1).detach()

        #exchange data
        mask = condition_2.squeeze(-1)

        actions_1[mask], actions_2[mask] = actions_2[mask], actions_1[mask]

        for key in obs_keys:
            observations_1[key][mask], observations_2[key][mask] = (
                observations_2[key][mask],
                observations_1[key][mask]
            )

        length_1[mask], length_2[mask] = length_2[mask], length_1[mask]

        # normalize input
        obs_1 = self.normalizer.normalize(observations_1)
        action_1 = self.normalizer['action'].normalize(actions_1)
        obs_2 = self.normalizer.normalize(observations_2)
        action_2 = self.normalizer['action'].normalize(actions_2)

        # slice episode
        stride = stride

        obs_1 = {key:slice_episode(obs_1[key], horizon=self.horizon, stride=stride) for key in obs_1.keys()}
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=stride)
        obs_2 = {key:slice_episode(obs_2[key], horizon=self.horizon, stride=stride) for key in obs_2.keys()}
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=stride)        

        batch_size = action_1.shape[1]
        horizon = action_1.shape[2]
        To = self.n_obs_steps
        loss = 0

        # handle different ways of passing observation
        for _ in range(self.train_time_samples):
            timesteps_1 = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()
            timesteps_2 = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device).long()

            traj_loss_1, traj_loss_2 = 0, 0
            # mseloss_1, mseloss_2 = 0, 0

            for i in range(len(action_1)):
                obs_1_slide = {key:obs_1[key][i] for key in obs_1.keys()}
                action_1_slide = action_1[i]

                cond_1 = None
                trajectory_1 = action_1_slide
                if self.obs_as_cond:
                    # reshape B, T, ... to B*T
                    this_nobs = dict_apply(obs_1_slide, 
                        lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                    nobs_features = self.obs_encoder(this_nobs)
                    # reshape back to B, T, Do
                    cond_1 = nobs_features.reshape(batch_size, To, -1)
                    if self.pred_action_steps_only:
                        start = To - 1
                        end = start + self.n_action_steps
                        trajectory_1 = action_1_slide[:,start:end]
                else:
                    # reshape B, T, ... to B*T
                    this_nobs = dict_apply(obs_1_slide, lambda x: x.reshape(-1, *x.shape[2:]))
                    nobs_features = self.obs_encoder(this_nobs)
                    # reshape back to B, T, Do
                    nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                    trajectory_1 = torch.cat([action_1_slide, nobs_features], dim=-1).detach()

                # generate impainting mask
                if self.pred_action_steps_only:
                    condition_mask_1 = torch.zeros_like(trajectory_1, dtype=torch.bool)
                else:
                    condition_mask_1 = self.mask_generator(trajectory_1.shape)

                # Sample noise that we'll add to the images
                noise_1 = torch.randn(trajectory_1.shape, device=trajectory_1.device)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_trajectory_1 = self.noise_scheduler.add_noise(
                    trajectory_1, noise_1, timesteps_1)

                # compute loss mask
                loss_mask_1 = ~condition_mask_1

                # apply conditioning
                noisy_trajectory_1[condition_mask_1] = trajectory_1[condition_mask_1]
        
                # Predict the noise residual
                pred_1 = self.model(noisy_trajectory_1, timesteps_1, cond_1)

                pred_type_1 = self.noise_scheduler.config.prediction_type
                target = noise_1 if pred_type_1 == 'epsilon' else trajectory_1

                mask_1 = (self.horizon + (i-1)*stride) <= length_1
                mask_1 = mask_1.int()

                slice_loss_1 = torch.norm((pred_1 - noise_1) * loss_mask_1.type(pred_1.dtype), dim=-1) ** 2

                traj_loss_1 += (slice_loss_1*mask_1) * (self.gamma ** (i*self.horizon + torch.arange(0, self.horizon, device=self.device))).reshape(1, -1)
               
            for i in range(len(obs_2)):
                obs_2_slide = {key:obs_2[key][i] for key in obs_2.keys()}
                action_2_slide = action_2[i]

                cond_2 = None
                trajectory_2 = action_2_slide
                if self.obs_as_cond:
                    # reshape B, T, ... to B*T
                    this_nobs = dict_apply(obs_2_slide, 
                        lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                    nobs_features = self.obs_encoder(this_nobs)
                    # reshape back to B, T, Do
                    cond_2 = nobs_features.reshape(batch_size, To, -1)
                    if self.pred_action_steps_only:
                        start = To - 1
                        end = start + self.n_action_steps
                        trajectory_2 = action_2_slide[:,start:end]
                else:
                    # reshape B, T, ... to B*T
                    this_nobs = dict_apply(obs_2_slide, lambda x: x.reshape(-1, *x.shape[2:]))
                    nobs_features = self.obs_encoder(this_nobs)
                    # reshape back to B, T, Do
                    nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                    trajectory_2 = torch.cat([action_2_slide, nobs_features], dim=-1).detach()

                # generate impainting mask
                if self.pred_action_steps_only:
                    condition_mask_2 = torch.zeros_like(trajectory_2, dtype=torch.bool)
                else:
                    condition_mask_2 = self.mask_generator(trajectory_2.shape)

                # Sample noise that we'll add to the images
                noise_2 = torch.randn(trajectory_2.shape, device=trajectory_2.device)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_trajectory_2 = self.noise_scheduler.add_noise(
                    trajectory_2, noise_2, timesteps_2)

                # compute loss mask
                loss_mask_2 = ~condition_mask_2

                # apply conditioning
                noisy_trajectory_2[condition_mask_2] = trajectory_2[condition_mask_2]

                # Predict the noise residual
                pred_2 = self.model(noisy_trajectory_2, timesteps_2, cond_2)

                pred_type_2 = self.noise_scheduler.config.prediction_type
                target = noise_2 if pred_type_2 == 'epsilon' else trajectory_2

                mask_2 = (self.horizon + (i-1)*stride) <= length_2
                mask_2 = mask_2.int()

                slice_loss_2 = torch.norm((pred_2 - noise_2) * loss_mask_2.type(pred_2.dtype), dim=-1) ** 2

                traj_loss_2 += (slice_loss_2*mask_2) * (self.gamma ** (i*self.horizon + torch.arange(0, self.horizon, device=self.device))).reshape(1, -1)

            traj_loss_1 = torch.sum(traj_loss_1, dim=-1)
            traj_loss_2 = torch.sum(traj_loss_2, dim=-1)
            immitation_loss = (traj_loss_1 + traj_loss_2)

            traj_loss_1 = -self.beta * self.noise_scheduler.config.num_train_timesteps * traj_loss_1
            traj_loss_2 = -self.beta * self.noise_scheduler.config.num_train_timesteps * traj_loss_2

            diff_loss_1 = torch.mean(torch.abs(traj_loss_1 - self.bias_reg*traj_loss_2))
            # diff_loss_2 = torch.mean(torch.abs(traj_loss_2 - self.bias_reg*traj_loss_1))

            mle_loss_1 = -F.logsigmoid(traj_loss_1 - self.bias_reg*traj_loss_2) + immitation_loss/((len(action_1) + len(action_2))*self.horizon)
            mle_loss_2 = -F.logsigmoid(traj_loss_2 - self.bias_reg*traj_loss_1) + immitation_loss/((len(action_1) + len(action_2))*self.horizon)

            loss += mle_loss_1 / (2 * self.train_time_samples) 

        return torch.mean(loss)
