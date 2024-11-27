from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
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
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict  # not implemented yet
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
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition through global feature
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through inpainting
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        # Debug: print shapes before calling conditional_sample
        print(f"Shape of cond_data before sampling: {cond_data.shape}")
        print(f"Shape of cond_mask before sampling: {cond_mask.shape}")

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred

        # Debug: print shapes after sampling
        print(f"Shape of action_pred: {action_pred.shape}")
        print(f"Shape of action: {action.shape}")

        return result


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # Hyperparameters
        beta = 1000  # Adjust as needed
        gamma = 0.99  # Adjust as needed
        N = self.noise_scheduler.config.num_train_timesteps  # Total diffusion steps
        num_n_samples = 10  # Number of times to sample n per sample

        # Get alphas and sigmas from the noise scheduler
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        betas = self.noise_scheduler.betas.to(device)

        # Batch size
        bsz = len(batch)  # Assuming batch is a list of samples
        device = self.device  # Assuming self.device is set appropriately

        total_loss = 0.0

        for sample in batch:
            # Extract preferred and disliked trajectories from the sample
            obs_w = sample['w']['obs']  # Preferred trajectory observations
            action_w = sample['w']['action']  # Preferred trajectory actions
            obs_l = sample['l']['obs']  # Disliked trajectory observations
            action_l = sample['l']['action']  # Disliked trajectory actions

            # Normalize the data
            obs_w = self.normalizer['obs'].normalize(obs_w)
            action_w = self.normalizer['action'].normalize(action_w)
            obs_l = self.normalizer['obs'].normalize(obs_l)
            action_l = self.normalizer['action'].normalize(action_l)

            # Get trajectory lengths
            T_w = action_w.shape[0]  # Length of preferred trajectory
            T_l = action_l.shape[0]  # Length of disliked trajectory

            # Repeat for num_n_samples
            sample_loss = 0.0
            for _ in range(num_n_samples):
                # Sample random timestep n
                n = torch.randint(0, N, (1,), device=device).long().item()

                # Sample noise for both trajectories
                noise_w = torch.randn_like(action_w)
                noise_l = torch.randn_like(action_l)

                bar_alpha_n = alphas_cumprod[n]  # Scalar
                beta_n = betas[n]  # Scalar
                alpha_n = torch.sqrt(bar_alpha_n)  # Scalar
                sigma_n_squared = 1 - bar_alpha_n  # Scalar
                sigma_n = torch.sqrt(sigma_n_squared)  # Scalar

                # Compute lambda_n
                lambda_n = (alpha_n ** 2) / (sigma_n ** 2)  # Scalar

                # Compute w(lambda_n)
                omega_lambda_n = (beta_n ** 2) / (2 * sigma_n_squared * alpha_n * (1 - bar_alpha_n))  # Scalar

                # Generate noisy actions using the noise scheduler
                trajectory_w = action_w
                noisy_action_w = self.noise_scheduler.add_noise(trajectory_w.unsqueeze(0), noise_w.unsqueeze(0), torch.tensor([n], device=device))
                noisy_action_w = noisy_action_w.squeeze(0)

                trajectory_l = action_l
                noisy_action_l = self.noise_scheduler.add_noise(trajectory_l.unsqueeze(0), noise_l.unsqueeze(0), torch.tensor([n], device=device))
                noisy_action_l = noisy_action_l.squeeze(0)

                # Prepare observations (if needed for conditioning)
                obs_w_input = obs_w
                obs_l_input = obs_l

                # Prepare model inputs based on conditioning strategy
                if self.obs_as_local_cond:
                    # Local conditioning
                    local_cond_w = obs_w_input.clone()
                    local_cond_l = obs_l_input.clone()
                    if self.n_obs_steps < local_cond_w.shape[0]:
                        local_cond_w[self.n_obs_steps:, :] = 0
                    if self.n_obs_steps < local_cond_l.shape[0]:
                        local_cond_l[self.n_obs_steps:, :] = 0
                    model_input_w = noisy_action_w
                    model_input_l = noisy_action_l
                elif self.obs_as_global_cond:
                    # Global conditioning
                    global_cond_w = obs_w_input[:self.n_obs_steps].reshape(-1)
                    global_cond_l = obs_l_input[:self.n_obs_steps].reshape(-1)
                    model_input_w = noisy_action_w
                    model_input_l = noisy_action_l
                else:
                    # No conditioning
                    model_input_w = torch.cat([noisy_action_w, obs_w_input], dim=-1)
                    model_input_l = torch.cat([noisy_action_l, obs_l_input], dim=-1)
                    local_cond_w = None
                    local_cond_l = None
                    global_cond_w = None
                    global_cond_l = None

                # Get model predictions for both trajectories
                pred_w = self.model(model_input_w.unsqueeze(0), torch.tensor([n], device=device),
                                    local_cond=local_cond_w.unsqueeze(0) if local_cond_w is not None else None,
                                    global_cond=global_cond_w.unsqueeze(0) if global_cond_w is not None else None)
                pred_w = pred_w.squeeze(0)

                pred_l = self.model(model_input_l.unsqueeze(0), torch.tensor([n], device=device),
                                    local_cond=local_cond_l.unsqueeze(0) if local_cond_l is not None else None,
                                    global_cond=global_cond_l.unsqueeze(0) if global_cond_l is not None else None)
                pred_l = pred_l.squeeze(0)

                # Reference model's predictions for both trajectories
                with torch.no_grad():
                    pred_ref_w = self.reference_model(model_input_w.unsqueeze(0), torch.tensor([n], device=device),
                                                    local_cond=local_cond_w.unsqueeze(0) if local_cond_w is not None else None,
                                                    global_cond=global_cond_w.unsqueeze(0) if global_cond_w is not None else None)
                    pred_ref_w = pred_ref_w.squeeze(0)

                    pred_ref_l = self.reference_model(model_input_l.unsqueeze(0), torch.tensor([n], device=device),
                                                    local_cond=local_cond_l.unsqueeze(0) if local_cond_l is not None else None,
                                                    global_cond=global_cond_l.unsqueeze(0) if global_cond_l is not None else None)
                    pred_ref_l = pred_ref_l.squeeze(0)

                # Compute differences between model's predictions and reference model's predictions
                diff_w = pred_w - pred_ref_w  # Shape: [T_w, D]
                diff_l = pred_l - pred_ref_l  # Shape: [T_l, D]

                # Compute norms of the differences
                norm_w = diff_w.norm(p=2, dim=-1)  # Shape: [T_w]
                norm_l = diff_l.norm(p=2, dim=-1)  # Shape: [T_l]

                # Apply discount factor gamma^t
                discounts_w = gamma ** torch.arange(T_w, device=device).float()
                discounts_l = gamma ** torch.arange(T_l, device=device).float()

                # Apply discounts
                discounted_norm_w = norm_w * discounts_w  # Shape: [T_w]
                discounted_norm_l = norm_l * discounts_l  # Shape: [T_l]

                # Sum over time steps
                sum_w = discounted_norm_w.sum()  # Scalar
                sum_l = discounted_norm_l.sum()  # Scalar

                # Compute the argument inside the sigmoid
                argument = -beta * N * omega_lambda_n * (sum_w - sum_l)  # Scalar

                # Compute the loss
                loss = -torch.log(torch.sigmoid(argument))

                # Accumulate sample loss
                sample_loss += loss

            # Average sample loss over num_n_samples
            sample_loss = sample_loss / num_n_samples

            # Accumulate total loss over batch
            total_loss += sample_loss

        # Average total loss over batch size
        total_loss = total_loss / bsz
        total_loss = total_loss * loss_mask.type(total_loss.dtype)
        total_loss = reduce(total_loss, 'b ... -> b (...)', 'mean')

        return total_loss


