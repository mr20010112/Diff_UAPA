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
    


def compute_all_traj_loss_realrobot(replay_buffer=None, model=None, ref_model=None, stride=1, sample_size = 20, batch_size=10, gc_every_n_batches=12):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    if replay_buffer is None:
        return np.zeros([1])
    else:
        assert sample_size >= batch_size, "data_size should be greater than or equal to batch_size"
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        data = replay_buffer.data
        data = unflatten_dataset_dict(flat_dict=data)
        data_size = len(data['action'])
        indicis = np.random.choice(data_size, size=sample_size, replace=False)

        observations_1 = data['obs']
        actions_1 = np.array(data['action'][indicis], dtype=np.float32)
        observations_2 = data['obs_2']
        actions_2 = np.array(data['action_2'][indicis], dtype=np.float32)
        compress_len_1 = data['compress_len'][indicis]
        compress_len_2 = data['compress_len_2'][indicis]
        camera_keys = observations_1['images'].keys()
        qpos_keys = [key for key in observations_1.keys() if key != 'images']
        del data

        for key in camera_keys:
            img_data_1 = observations_1['images'][key][indicis]
            img_data_2 = observations_2['images'][key][indicis]
            total_images = img_data_1.shape[0]
            
            img_batch_size = min(batch_size, total_images)
            decompressed_images_1 = []
            
            for batch_idx in range(0, total_images, img_batch_size):
                end_idx = min(batch_idx + img_batch_size, total_images)
                batch_decompressed = []
                
                for k in range(batch_idx, end_idx):
                    image = img_data_1[k, :, :int(compress_len_1[k, 0])].copy()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(decode_image, image)
                        decompressed_images = list(results)
                    batch_decompressed.append(decompressed_images)
                
                batch_decompressed = np.array(batch_decompressed)
                batch_decompressed = np.einsum('b k h w c -> b k c h w', batch_decompressed)
                decompressed_images_1.append(torch.from_numpy(batch_decompressed / 255.0).float())
                
                del batch_decompressed
                torch.cuda.empty_cache()
                
                if batch_idx % (img_batch_size * gc_every_n_batches) == 0:
                    gc.collect()
            
            observations_1[key] = torch.cat(decompressed_images_1, dim=0)
            del observations_1['images']
            del decompressed_images_1
            
            decompressed_images_2 = []
            
            for batch_idx in range(0, total_images, img_batch_size):
                end_idx = min(batch_idx + img_batch_size, total_images)
                batch_decompressed = []
                
                for k in range(batch_idx, end_idx):
                    image = img_data_2[k, :, :int(compress_len_2[k, 0])].copy()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(decode_image, image)
                        decompressed_images = list(results)
                    batch_decompressed.append(decompressed_images)
                
                batch_decompressed = np.array(batch_decompressed)
                batch_decompressed = np.einsum('b k h w c -> b k c h w', batch_decompressed)
                decompressed_images_2.append(torch.from_numpy(batch_decompressed / 255.0).float())
                
                del batch_decompressed
                torch.cuda.empty_cache()
                
                if batch_idx % (img_batch_size * gc_every_n_batches) == 0:
                    gc.collect()
            
            observations_2[key] = torch.cat(decompressed_images_2, dim=0)
            del observations_2['images']
            del decompressed_images_2
            
            del img_data_1, img_data_2
            gc.collect()
            torch.cuda.empty_cache()

        for key in qpos_keys:
            observations_1[key] = torch.from_numpy(observations_1[key]).float()
            observations_2[key] = torch.from_numpy(observations_2[key]).float()
            
            observations_1[key] = observations_1[key].cpu()
            observations_2[key] = observations_2[key].cpu()

        gc.collect()
        torch.cuda.empty_cache()

        for param in ref_model.parameters():
            param.requires_grad = False
        
        device = model.device
        ref_model = ref_model.to(device)
        
        with torch.no_grad():
            obs_1 = model.normalizer.normalize(observations_1)
            action_1 = model.normalizer['action'].normalize(actions_1)
            obs_2 = model.normalizer.normalize(observations_2)
            action_2 = model.normalizer['action'].normalize(actions_2)
        
        start_1 = random.randint(0, model.n_obs_steps)
        start_2 = random.randint(0, model.n_obs_steps)
        
        with torch.no_grad():
            obs_1 = {key: slice_episode(obs_1[key], horizon=model.horizon, stride=stride, start=start_1) for key in obs_1.keys()}
            action_1 = slice_episode(action_1, horizon=model.horizon, stride=stride, start=start_1)
            obs_2 = {key: slice_episode(obs_2[key], horizon=model.horizon, stride=stride, start=start_2) for key in obs_2.keys()}
            action_2 = slice_episode(action_2, horizon=model.horizon, stride=stride, start=start_2)
        
        del observations_1, observations_2, actions_1, actions_2
        gc.collect()
        torch.cuda.empty_cache()

        def compute_traj_image_loss_batched(obs_slices, action_slices, model, ref_model, batch_size):
            with torch.no_grad(): 
                To = model.n_obs_steps
                horizon = model.horizon
                total_samples = action_slices.shape[0]
                total_loss = torch.zeros(total_samples, device='cpu')
                
                num_batches = (total_samples + batch_size - 1) // batch_size
                device = model.device
                
                for batch_idx in range(num_batches):

                    if batch_idx > 0 and batch_idx % gc_every_n_batches == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total_samples)
                    current_batch_size = end_idx - start_idx
                    
                    batch_timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, 
                                                   (current_batch_size,), device=device).long()
                    
                    batch_action_slices = action_slices[start_idx:end_idx].to(device)
                    batch_obs_slices = {key: obs_slices[key][start_idx:end_idx].to(device) for key in obs_slices.keys()}
                    
                    batch_total_loss = torch.zeros(current_batch_size, device=device)
                    
                    for idx in range(current_batch_size):
                        action_slide = batch_action_slices[idx:idx+1]
                        obs_slide = {key: batch_obs_slices[key][idx:idx+1] for key in batch_obs_slices.keys()}
                        sample_timesteps = batch_timesteps[idx:idx+1]
                        
                        local_cond = None
                        global_cond = None
                        global_cond_ref = None

                        if model.obs_as_global_cond:
                            this_nobs = dict_apply(obs_slide, 
                                lambda x: x[:,:To,...].reshape(-1, *x.shape[2:]))
                            nobs_features = model.obs_encoder(this_nobs)
                            nobs_features_ref = ref_model.obs_encoder(this_nobs)
                            
                            global_cond = nobs_features.reshape(1, -1)
                            global_cond_ref = nobs_features_ref.reshape(1, -1)
                            trajectory = action_slide
                        else:
                            this_nobs = dict_apply(obs_slide, 
                                lambda x: x.reshape(-1, *x.shape[2:]))
                            nobs_features = model.obs_encoder(this_nobs)
                            nobs_features_ref = ref_model.obs_encoder(this_nobs)
                            
                            nobs_features = nobs_features.reshape(1, horizon, -1)
                            nobs_features_ref = nobs_features_ref.reshape(1, horizon, -1)
                            
                            trajectory = torch.cat([action_slide, nobs_features], dim=-1)
                            trajectory_ref = torch.cat([action_slide, nobs_features_ref], dim=-1)

                        condition_mask = model.mask_generator(trajectory.shape).to(device)
                        loss_mask = (~condition_mask).float()

                        noise = torch.randn(trajectory.shape, device=device)

                        noisy_trajectory = model.noise_scheduler.add_noise(trajectory, noise, sample_timesteps)
                        noisy_trajectory[condition_mask] = trajectory[condition_mask]
                        
                        if not model.obs_as_global_cond:
                            noisy_trajectory_ref = model.noise_scheduler.add_noise(trajectory_ref, noise, sample_timesteps)
                            noisy_trajectory_ref[condition_mask] = trajectory_ref[condition_mask]
                        else:
                            noisy_trajectory_ref = noisy_trajectory.clone()

                        pred = model.model(noisy_trajectory, sample_timesteps, 
                                          local_cond=local_cond, global_cond=global_cond)
                        pred_ref = ref_model.model(noisy_trajectory_ref if not model.obs_as_global_cond else noisy_trajectory, 
                                                 sample_timesteps, local_cond=local_cond, global_cond=global_cond_ref)

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
                        batch_total_loss[idx] = slice_loss.squeeze()
                        
                        del trajectory, noise, noisy_trajectory, pred, pred_ref
                        if not model.obs_as_global_cond:
                            del trajectory_ref, noisy_trajectory_ref
                        del nobs_features, nobs_features_ref, this_nobs
                    
                    total_loss[start_idx:end_idx] = batch_total_loss.cpu()
                    
                    del batch_action_slices, batch_obs_slices, batch_timesteps, batch_total_loss
                    torch.cuda.empty_cache()

                return total_loss

        with torch.no_grad():
            total_samples = action_1.shape[0]
            traj_loss_1 = compute_traj_image_loss_batched(obs_1, action_1, model, ref_model, batch_size)
            
            if traj_loss_1.device.type != 'cpu':
                traj_loss_1 = traj_loss_1.cpu()
            
            del obs_1, action_1
            gc.collect()
            torch.cuda.empty_cache()
            
            total_samples = action_2.shape[0]
            traj_loss_2 = compute_traj_image_loss_batched(obs_2, action_2, model, ref_model, batch_size)
            
            if traj_loss_2.device.type != 'cpu':
                traj_loss_2 = traj_loss_2.cpu()
            
            loss = (traj_loss_1 + traj_loss_2) / 2
            final_loss = torch.mean(loss)

        del obs_2, action_2, traj_loss_1, traj_loss_2, loss
        gc.collect()
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