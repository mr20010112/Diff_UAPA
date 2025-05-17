import torch
import numpy as np
import torch.nn.functional as F
import random
from einops import rearrange, reduce
import cv2
import concurrent.futures
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.slice import slice_episode

def unflatten_dataset_dict(flat_dict, delimiter='/'):
    result = {}
    for compound_key, value in flat_dict.items():
        keys = compound_key.split(delimiter)
        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    return result

def decode_image(data):
    return cv2.imdecode(data, 1)

def compute_all_traj_loss(replay_buffer=None, model:BaseImagePolicy=None, ref_model:BaseImagePolicy=None, stride=1):
    if replay_buffer is None:
        return np.zeros([1])
    else:
        data = replay_buffer.data
        meta_data = replay_buffer.meta
        observations_1 = np.array(data['obs'], dtype=np.float32)
        actions_1 = np.array(data['action'], dtype=np.float32)
        observations_2 = np.array(data['obs_2'], dtype=np.float32)
        actions_2 = np.array(data['action_2'], dtype=np.float32)

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
        obs_1 = slice_episode(obs_1, horizon=model.horizon, stride=stride)
        action_1 = slice_episode(actions_1, horizon=model.horizon, stride=stride)
        obs_2 = slice_episode(obs_2, horizon=model.horizon, stride=stride)
        action_2 = slice_episode(actions_2, horizon=model.horizon, stride=stride)


        bsz = obs_1[0].shape[0]
        timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=model.device).long()

        # Pre-allocate loss
        traj_loss_1, traj_loss_2 = 0, 0

        # Helper function to compute loss for a single trajectory
        def compute_traj_loss(obs_slices, action_slices, timestep, model, ref_policy):
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

                    pred_ref = ref_policy.model(noisy_trajectory, timestep, cond)
                    pred = model.model(noisy_trajectory, timestep, cond)

                    pred_type = self.noise_scheduler.config.prediction_type 
                    if pred_type == 'epsilon':
                        target = noise
                    elif pred_type == 'sample':
                        target = trajectory
                    else:
                        raise ValueError(f"Unsupported prediction type {pred_type}")
                    
                    loss = F.mse_loss(pred, target, reduction='none')
                    loss_ref = F.mse_loss(pred_ref, target, reduction='none')
                    loss = loss * loss_mask.type(loss.dtype)
                    loss_ref = loss_ref * loss_mask.type(loss.dtype)
                    loss = reduce(loss, 'b t ... -> b t (...)', 'mean')
                    loss_ref = reduce(loss_ref, 'b t ... -> b t (...)', 'mean')
                    
                    slice_loss = torch.sum((loss - loss_ref), dim=-1)
                    total_loss += torch.sum(slice_loss * gamma_factors)
                # Explicitly delete unused variables to release GPU memory
                del trajectory, noise, noisy_trajectory, pred_ref, pred, loss_mask, condition_mask
                torch.cuda.empty_cache()

            return total_loss.detach()

        # Compute loss for trajectory 1
        traj_loss_1 = compute_traj_loss(obs_1, action_1, timesteps, model, ref_model)
        # Compute loss for trajectory 2
        traj_loss_2 = compute_traj_loss(obs_2, action_2, timesteps, model, ref_model)

        # Average the losses
        loss = (traj_loss_1 + traj_loss_2) / 2

        return torch.mean(loss)
    


def compute_all_traj_loss_realrobot(replay_buffer=None, model=None, ref_model=None, stride=1):
    if replay_buffer is None:
        return np.zeros([1])
    else:
        data = replay_buffer.data
        data = unflatten_dataset_dict(flat_dict=data)
        observations_1 = data['obs']
        actions_1 = np.array(data['action'], dtype=np.float32)
        observations_2 = data['obs_2']
        actions_2 = np.array(data['action_2'], dtype=np.float32)
        compress_len_1 = data['compress_len']
        compress_len_2 = data['compress_len_2']
        camera_keys = observations_1['images'].keys()
        qpos_keys = [key for key in observations_1.keys() if key != 'images']

        for key in camera_keys:
            img_data_1 = observations_1['images'][key]
            img_data_2 = observations_2['images'][key]
            decompressed_images_1 = []
            decompressed_images_2 = []

            for k in range(img_data_1.shape[0]):
                image = img_data_1[k, :, :int(compress_len_1[k, 0])].copy()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(decode_image, image)
                    decompressed_images = list(results)
                decompressed_images_1.append(decompressed_images)

            decompressed_images_1 = np.array(decompressed_images_1)
            decompressed_images_1 = np.einsum('b k h w c -> b k c h w', decompressed_images_1)
            observations_1[key] = torch.from_numpy(decompressed_images_1 / 255.0).float()

            for k in range(img_data_2.shape[0]):
                image = img_data_2[k, :, :int(compress_len_2[k, 0])].copy()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(decode_image, image)
                    decompressed_images = list(results)
                decompressed_images_2.append(decompressed_images)

            decompressed_images_2 = np.array(decompressed_images_2)
            decompressed_images_2 = np.einsum('b k h w c -> b k c h w', decompressed_images_2)
            observations_2[key] = torch.from_numpy(decompressed_images_2 / 255.0).float()

            del img_data_1, img_data_2, decompressed_images_1, decompressed_images_2
            torch.cuda.empty_cache()

        for key in qpos_keys:
            observations_1[key] = torch.from_numpy(observations_1[key]).float()
            observations_2[key] = torch.from_numpy(observations_2[key]).float()

        del data
        torch.cuda.empty_cache()

        for param in ref_model.parameters():
            param.requires_grad = False

        ref_model = ref_model.to(model.device)

        obs_1 = model.normalizer.normalize(observations_1)
        action_1 = model.normalizer['action'].normalize(actions_1)
        obs_2 = model.normalizer.normalize(observations_2)
        action_2 = model.normalizer['action'].normalize(actions_2)

        start_1 = random.randint(0, model.n_obs_steps)
        start_2 = random.randint(0, model.n_obs_steps)
        
        obs_1 = {key: slice_episode(obs_1[key], horizon=model.horizon, stride=stride, start=start_1) for key in obs_1.keys()}
        action_1 = slice_episode(action_1, horizon=model.horizon, stride=stride, start=start_1)
        obs_2 = {key: slice_episode(obs_2[key], horizon=model.horizon, stride=stride, start=start_2) for key in obs_2.keys()}
        action_2 = slice_episode(action_2, horizon=model.horizon, stride=stride, start=start_2)

        bsz = action_1.shape[0]
        timesteps_1 = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=model.device).long()
        timesteps_2 = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=model.device).long()

        def compute_traj_image_loss(obs_slices, action_slices, timesteps, model, ref_model):
            with torch.no_grad(): 
                total_loss = torch.zeros(action_slices.shape[0], device=model.device)
                To = model.n_obs_steps
                batch_size = action_slices.shape[0]
                horizon = model.horizon
                cond = None

                for idx in range(len(action_slices)):
                    action_slide = action_slices[idx]
                    obs_slide = {key: obs_slices[key][idx] for key in obs_slices.keys()}
                    
                    local_cond = None
                    global_cond = None
                    global_cond_ref = None

                    if model.obs_as_global_cond:
                        this_nobs = dict_apply(obs_slide, 
                            lambda x: x[:,:To,...].reshape(-1, *x.shape[2:]))
                        nobs_features = model.obs_encoder(this_nobs)
                        nobs_features_ref = ref_model.obs_encoder(this_nobs)
                        
                        global_cond = nobs_features.reshape(batch_size, -1)
                        global_cond_ref = nobs_features_ref.reshape(batch_size, -1)
                        trajectory = action_slide
                    else:
                        this_nobs = dict_apply(obs_slide, 
                            lambda x: x.reshape(-1, *x.shape[2:]))
                        nobs_features = model.obs_encoder(this_nobs)
                        nobs_features_ref = ref_model.obs_encoder(this_nobs)
                        
                        nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                        nobs_features_ref = nobs_features_ref.reshape(batch_size, horizon, -1)
                        
                        trajectory = torch.cat([action_slide, nobs_features], dim=-1)
                        trajectory_ref = torch.cat([action_slide, nobs_features_ref], dim=-1)

                    condition_mask = model.mask_generator(trajectory.shape).to(model.device)
                    loss_mask = (~condition_mask).float()

                    noise = torch.randn(trajectory.shape, device=model.device)

                    noisy_trajectory = model.noise_scheduler.add_noise(trajectory, noise, timesteps)
                    noisy_trajectory[condition_mask] = trajectory[condition_mask]
                    
                    if not model.obs_as_global_cond:
                        noisy_trajectory_ref = model.noise_scheduler.add_noise(trajectory_ref, noise, timesteps)
                        noisy_trajectory_ref[condition_mask] = trajectory_ref[condition_mask]
                    else:
                        noisy_trajectory_ref = noisy_trajectory.clone()

                    pred = model.model(noisy_trajectory, timesteps, 
                                      local_cond=local_cond, global_cond=global_cond)
                    pred_ref = ref_model.model(noisy_trajectory_ref if not model.obs_as_global_cond else noisy_trajectory, 
                                             timesteps, local_cond=local_cond, global_cond=global_cond_ref)

                    pred_type = model.noise_scheduler.config.prediction_type
                    if pred_type == 'epsilon':
                        target = noise
                    elif pred_type == 'sample':
                        target = trajectory
                    else:
                        raise ValueError(f"Unsupported prediction type {pred_type}")

                    loss = F.mse_loss(pred, target, reduction='none')
                    loss_ref = F.mse_loss(pred_ref, target, reduction='none')
                    
                    loss = loss * loss_mask
                    loss_ref = loss_ref * loss_mask
                    loss = reduce(loss, 'b t ... -> b t (...)', 'mean')
                    loss_ref = reduce(loss_ref, 'b t ... -> b t (...)', 'mean')
                    
                    slice_loss = torch.sum(loss_ref - loss, dim=1)
                    total_loss += slice_loss
                    
                    del trajectory, noise, noisy_trajectory, pred, pred_ref
                    if not model.obs_as_global_cond:
                        del trajectory_ref, noisy_trajectory_ref
                    del nobs_features, nobs_features_ref, this_nobs
                    torch.cuda.empty_cache()

                return total_loss

        with torch.no_grad():

            traj_loss_1 = compute_traj_image_loss(obs_1, action_1, timesteps_1, model, ref_model)
            
            traj_loss_2 = compute_traj_image_loss(obs_2, action_2, timesteps_2, model, ref_model)

            loss = (traj_loss_1 + traj_loss_2) / 2
            final_loss = torch.mean(loss)

        del obs_1, obs_2, action_1, action_2, traj_loss_1, traj_loss_2, loss
        torch.cuda.empty_cache()

        return final_loss

    
def compute_all_bet_traj_loss(replay_buffer=None, model=None, stride=1):
    if replay_buffer is None:
        return np.zeros([1])
    else:
        data = replay_buffer.data
        meta_data = replay_buffer.meta
        observations_1 = np.array(data['obs'], dtype=np.float32)
        actions_1 = np.array(data['action'], dtype=np.float32)
        observations_2 = np.array(data['obs_2'], dtype=np.float32)
        actions_2 = np.array(data['action_2'], dtype=np.float32)
        length_1 = torch.tensor(meta_data['length'], device=model.device)
        length_2 = torch.tensor(meta_data['length_2'], device=model.device)

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
        obs_1 = slice_episode(obs_1, horizon=model.horizon, stride=stride)
        action_1 = slice_episode(actions_1, horizon=model.horizon, stride=stride)
        obs_2 = slice_episode(obs_2, horizon=model.horizon, stride=stride)
        action_2 = slice_episode(actions_2, horizon=model.horizon, stride=stride)

        # Pre-allocate loss
        traj_loss_1, traj_loss_2 = 0, 0

        # Helper function to compute loss for a single trajectory
        def compute_traj_loss(obs_slices, action_slices, model, length, stride):
            total_loss = 0
            
            for idx, (obs_slide, action_slide) in enumerate(zip(obs_slices, action_slices)):
                gamma_factors = model.gamma ** (idx * model.horizon)
                obs_slide[:, model.n_obs_steps:, :] = -2

                enc_obs = model.obs_encoding_net(obs_slide)
                latent = model.action_ae.encode_into_latent(action_slide, enc_obs)

                loss = model.get_pred_loss(
                    obs_rep=enc_obs.clone(),
                    target_latents=latent,
                )

                mask = (model.horizon + (idx - 1)*stride) <= length
                mask = mask.int()

                total_loss += (loss * mask) * gamma_factors

            total_loss = torch.sum(total_loss, dim=-1)

            return total_loss.detach()

        # Compute loss for trajectory 1
        traj_loss_1 = compute_traj_loss(obs_1, action_1, model, length_1, stride)
        # Compute loss for trajectory 2
        traj_loss_2 = compute_traj_loss(obs_2, action_2, model, length_2, stride)

        # Average the losses
        loss = (traj_loss_1 + traj_loss_2) / 2

        return torch.mean(loss)