import torch
import numpy as np
from diffusion_policy.model.common.slice import slice_episode

def compute_all_traj_loss(replay_buffer=None, model=None, ref_model=None):
    if replay_buffer is None:
        return np.zeros([1])
    else:
        data = replay_buffer.data
        observations_1 = np.array(data['obs'], dtype=np.float32)
        actions_1 = np.array(data['action'], dtype=np.float32)
        observations_2 = np.array(data['obs_2'], dtype=np.float32)
        actions_2 = np.array(data['action_2'], dtype=np.float32)

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        ref_model = ref_model.to(model.device)

        # Normalize data
        batch_1 = {
            'obs': observations_1,
            'action': actions_1,
        }

        batch_2 = {
            'obs': observations_2,
            'action': actions_2,
        }
        nbatch_1 = model.normalizer.normalize(batch_1)
        nbatch_2 = model.normalizer.normalize(batch_2)
        obs_1, obs_2 = nbatch_1['obs'], nbatch_2['obs']
        actions_1, actions_2 = nbatch_1['action'], nbatch_2['action']

        # Slice trajectories
        obs_1 = slice_episode(obs_1, horizon=model.horizon, stride=model.horizon)
        action_1 = slice_episode(actions_1, horizon=model.horizon, stride=model.horizon)
        obs_2 = slice_episode(obs_2, horizon=model.horizon, stride=model.horizon)
        action_2 = slice_episode(actions_2, horizon=model.horizon, stride=model.horizon)


        bsz = obs_1[0].shape[0]
        timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=model.device).long()

        # Pre-allocate loss
        traj_loss_1, traj_loss_2 = 0, 0

        # Helper function to compute loss for a single trajectory
        def compute_traj_loss(obs_slices, action_slices, timestep, model, ref_model):
            total_loss = 0
            
            for idx, (obs_slide, action_slide) in enumerate(zip(obs_slices, action_slices)):
                gamma_factors = model.gamma ** (idx * model.horizon + np.arange(model.horizon))
                if model.obs_as_cond:
                    cond = obs_slide[:, :model.n_obs_steps, :]
                    cond = cond.detach().to(model.device)
                    # cond.detach().to(model.device)
                    trajectory = action_slide[:, -model.n_action_steps:] if model.pred_action_steps_only else action_slide
                else:
                    cond = None
                    trajectory = np.concatenate([action_slide, obs_slide], axis=-1)

                condition_mask = model.mask_generator(trajectory.shape).to(model.device)
                loss_mask = (~condition_mask).float()

                trajectory = torch.tensor(trajectory, device=model.device, dtype=torch.float32)
                noise = torch.randn(trajectory.shape, device=model.device)

                # Disable gradient computation
                with torch.no_grad():
                    noisy_trajectory = model.noise_scheduler.add_noise(trajectory, noise, timestep)
                    noisy_trajectory[condition_mask] = trajectory[condition_mask]

                    ref_pred = ref_model(noisy_trajectory, timestep, cond)
                    pred = model.model(noisy_trajectory, timestep, cond)

                    slice_loss = torch.norm((pred - noise) * loss_mask.type(pred.dtype), dim=-1) ** 2\
                                - torch.norm((ref_pred - noise) * loss_mask.type(ref_pred.dtype), dim=-1) ** 2
                    total_loss += torch.sum(slice_loss * torch.tensor(gamma_factors, device=model.device, dtype=torch.float32), dim=-1)

                # Explicitly delete unused variables to release GPU memory
                del trajectory, noise, noisy_trajectory, ref_pred, pred, loss_mask, condition_mask
                torch.cuda.empty_cache()

            return total_loss.detach()

        # Compute loss for trajectory 1
        traj_loss_1 = compute_traj_loss(obs_1, action_1, timesteps, model, ref_model)
        # Compute loss for trajectory 2
        traj_loss_2 = compute_traj_loss(obs_2, action_2, timesteps, model, ref_model)

        # Average the losses
        loss = (traj_loss_1 + traj_loss_2) / 2

        return torch.mean(loss)