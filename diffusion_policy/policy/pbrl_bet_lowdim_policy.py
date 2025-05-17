
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from omegaconf import DictConfig

from diffusion_policy.policy.bet_lowdim_policy import BETLowdimPolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.libraries.batch_loss_fn import BatchFocalLoss, soft_cross_entropy
from diffusion_policy.model.bet.utils import eval_mode
from diffusion_policy.model.common.slice import slice_episode

class TD3BCBETLowdimPolicy(BETLowdimPolicy):
    def __init__(self, 
                 action_ae: KMeansDiscretizer, 
                 obs_encoding_net: nn.Module, 
                 state_prior: MinGPT,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_freq: int = 2,
                 alpha: float = 0.02,
                 bc_alpha: float = 0.02):
        super().__init__(action_ae, obs_encoding_net, state_prior, horizon, 
                        n_action_steps, n_obs_steps)
        
        self.discount = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.bc_alpha = bc_alpha

        
        self.qf1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.qf2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.qf1_target = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.qf2_target = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        self.step = 0

    def get_optimizers(self, learning_rate: float, weight_decay: float, betas: Tuple[float, float] = (0.9, 0.999)) -> Dict[str, torch.optim.Optimizer]:
        return {
            'actor': self.state_prior.get_optimizer(weight_decay=0.1, learning_rate=learning_rate, betas=betas),
            'qf1': torch.optim.Adam(self.qf1.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas),
            'qf2': torch.optim.Adam(self.qf2.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas),
        }

    def get_pred_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.state_prior.predict_offsets:
            target_latents, target_offsets = target_latents
        is_soft_target = (target_latents.shape[-1] == self.state_prior.vocab_size) and (
            self.state_prior.vocab_size != 1
        )
        if is_soft_target:
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(target_latents.size(0),-1)
            if self.state_prior.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = BatchFocalLoss(gamma=self.state_prior.focal_loss_gamma)
        if self.state_prior.predict_offsets:
            # print(obs_rep._version)
            output, _ = self.state_prior.model(obs_rep)
            logits = output[:, :, : self.state_prior.vocab_size]
            offsets = output[:, :, self.state_prior.vocab_size :]
            batch = logits.shape[0]
            seq = logits.shape[1]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.state_prior.vocab_size,
                A=self.state_prior.action_dim,
            )
            # calculate (optionally soft) cross entropy and offset losses
            class_loss = criterion(logits, target_latents)
            # offset loss is only calculated on the target class
            # if soft targets, argmax is considered the target class
            selected_offsets = offsets[
                torch.arange(offsets.size(0)),
                target_latents.view(-1).argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1),
            ]
            offset_loss = self.state_prior.offset_loss_scale * F.mse_loss(
                selected_offsets.view(batch, -1, self.state_prior.action_dim), target_offsets, reduction='none'
            )

            offset_loss = offset_loss.mean(dim=(1, 2))
            loss = offset_loss + class_loss
        else:
            logits, _ = self.state_prior.model(obs_rep)
            loss = criterion(logits, target_latents)

        return loss

    def update_target_networks(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def compute_loss(self, batch: Dict[str, torch.Tensor], reward_model: nn.Module, stride: int = 1, avg_Traj_loss=0.0) -> Dict:
        To = self.n_obs_steps
        Ta = self.n_action_steps
        Th = self.horizon

        # Move data to device
        observations_1 = batch["obs"].to(self.device)
        actions_1 = batch["action"].to(self.device)
        votes_1 = batch["votes"].to(self.device)
        length_1 = batch["length"].to(self.device).detach()
        observations_2 = batch["obs_2"].to(self.device)
        actions_2 = batch["action_2"].to(self.device)
        votes_2 = batch["votes_2"].to(self.device)
        length_2 = batch["length_2"].to(self.device).detach()

        batch_1 = {
            'obs': observations_1,  # Already tensors, no need to recreate
            'action': actions_1,
        }

        batch_2 = {
            'obs': observations_2,
            'action': actions_2,
        }

        nbatch_1 = self.normalizer.normalize(batch_1)
        nbatch_2 = self.normalizer.normalize(batch_2)

        obs_1 = nbatch_1['obs']
        action_1 = nbatch_1['action']
        obs_2 = nbatch_2['obs']
        action_2 = nbatch_2['action']

        obs_1 = slice_episode(obs_1, horizon=2*To+Ta, stride=stride)
        action_1 = slice_episode(action_1, horizon=2*To+Ta, stride=stride)
        obs_2 = slice_episode(obs_2, horizon=2*To+Ta, stride=stride)
        action_2 = slice_episode(action_2, horizon=2*To+Ta, stride=stride)

        # Initialize as Python lists
        critic_loss_all = torch.tensor(0.0, device=self.device, requires_grad=True)
        actor_loss_all = torch.tensor(0.0, device=self.device, requires_grad=True)
        qf1_loss_all = torch.tensor(0.0, device=self.device, requires_grad=True)
        qf2_loss_all = torch.tensor(0.0, device=self.device, requires_grad=True)
        bc_loss_all = torch.tensor(0.0, device=self.device, requires_grad=True)
        q_loss_all = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Process first batch
        for i in range(len(obs_1)):
            obs_slide = obs_1[i]
            action_slide = action_1[i]
            nobs = obs_slide[:, :To, :]
            naction = action_slide[:, To-1:To+Ta-1, :]
            n_next_obs = obs_slide[:, To+Ta-1:2*To+Ta-1, :]

            q1 = self.qf1(torch.cat([nobs[:, -1, :], naction[:, 0, :]], dim=-1))
            q2 = self.qf2(torch.cat([nobs[:, -1, :], naction[:, 0, :]], dim=-1))

            reward = torch.mean(torch.stack([
                reward_model.ensemble[i](
                    nobs, 
                    naction,
                ) for i in range(len(reward_model.ensemble))
            ]), dim=0)

            with torch.no_grad():
                next_action_pred = self.predict_action({'obs': self.normalizer['obs'].unnormalize(n_next_obs)})['action']
                noise = torch.clamp(
                    self.policy_noise * torch.randn_like(next_action_pred),
                    -self.noise_clip, self.noise_clip
                )
                next_action = next_action_pred + noise

                next_q1 = self.qf1_target(torch.cat([n_next_obs[:, -1, :], 
                                                     self.normalizer['action'].normalize(next_action[:, 0, :])], dim=-1))
                next_q2 = self.qf2_target(torch.cat([n_next_obs[:, -1, :], 
                                                     self.normalizer['action'].normalize(next_action[:, 0, :])], dim=-1))
                next_q = torch.min(next_q1, next_q2)
                target_q = reward + self.discount * next_q

            qf1_loss = F.mse_loss(q1, target_q)
            qf2_loss = F.mse_loss(q2, target_q)
            critic_loss = qf1_loss + qf2_loss

            qf1_loss_all = qf1_loss_all + qf1_loss
            qf2_loss_all = qf2_loss_all + qf2_loss
            critic_loss_all = critic_loss_all + critic_loss

            if self.step % self.policy_freq == 0:
                pred_action_dict = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})
                pred_action = pred_action_dict['action'].clone()
                
                q_values = self.qf1(torch.cat([nobs[:, -1, :], 
                                               self.normalizer['action'].normalize(pred_action[:, -1, :])], dim=-1))
                q_loss = -q_values.mean()

                obs_slide[:,To:,:] = -2
                obs_input = obs_slide[:, :Th, :].view(obs_slide.size(0), Th, -1)
                enc_obs = self.obs_encoding_net(obs_input)
                action_input = action_slide[:, :Th, :].view(action_slide.size(0), Th, -1)
                latent = self.action_ae.encode_into_latent(action_input, enc_obs)

                _, bc_loss = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs.clone(),
                    target_latents=latent,
                )
                
                actor_loss = q_loss + self.alpha * bc_loss

                bc_loss_all = bc_loss_all + bc_loss
                q_loss_all = q_loss_all + q_loss
                actor_loss_all = actor_loss_all + actor_loss
            else:
                actor_loss = torch.tensor(0.0, device=self.device)
                actor_loss_all = actor_loss_all + actor_loss

            self.step += 1
            if self.step % self.policy_freq == 0:
                self.update_target_networks()

        # Process second batch
        for i in range(len(obs_2)):
            obs_slide = obs_2[i]
            action_slide = action_2[i]
            nobs = obs_slide[:, :To, :]
            naction = action_slide[:, To-1:To+Ta-1, :]
            n_next_obs = obs_slide[:, To+Ta-1:2*To+Ta-1, :]

            q1 = self.qf1(torch.cat([nobs[:, -1, :], naction[:, 0, :]], dim=-1))
            q2 = self.qf2(torch.cat([nobs[:, -1, :], naction[:, 0, :]], dim=-1))

            reward = torch.mean(torch.stack([
                reward_model.ensemble[i](
                    self.normalizer['obs'].unnormalize(nobs), 
                    self.normalizer['action'].unnormalize(naction),
                ) for i in range(len(reward_model.ensemble))
            ]), dim=0)

            with torch.no_grad():
                next_action_pred = self.predict_action({'obs': self.normalizer['obs'].unnormalize(n_next_obs)})['action']
                noise = torch.clamp(
                    self.policy_noise * torch.randn_like(next_action_pred),
                    -self.noise_clip, self.noise_clip
                )
                next_action = next_action_pred + noise

                next_q1 = self.qf1_target(torch.cat([n_next_obs[:, -1, :], 
                                                     self.normalizer['action'].normalize(next_action[:, 0, :])], dim=-1))
                next_q2 = self.qf2_target(torch.cat([n_next_obs[:, -1, :], 
                                                     self.normalizer['action'].normalize(next_action[:, 0, :])], dim=-1))
                next_q = torch.min(next_q1, next_q2)
                target_q = reward + self.discount * next_q

            qf1_loss = F.mse_loss(q1, target_q)
            qf2_loss = F.mse_loss(q2, target_q)
            critic_loss = qf1_loss + qf2_loss

            # Append to lists
            qf1_loss_all = qf1_loss_all + qf1_loss
            qf2_loss_all = qf2_loss_all + qf2_loss
            critic_loss_all = critic_loss_all + critic_loss

            if self.step % self.policy_freq == 0:
                pred_action_dict = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})
                pred_action = pred_action_dict['action'].clone()
                
                q_values = self.qf1(torch.cat([nobs[:, -1, :], 
                                               self.normalizer['action'].normalize(pred_action[:, -1, :])], dim=-1))
                q_loss = -q_values.mean()

                obs_slide[:,To:,:] = -2
                obs_input = obs_slide[:, :Th, :].view(obs_slide.size(0), Th, -1)
                enc_obs = self.obs_encoding_net(obs_input)
                action_input = action_slide[:, :Th, :].view(action_slide.size(0), Th, -1)
                latent = self.action_ae.encode_into_latent(action_input, enc_obs)

                _, bc_loss = self.state_prior.get_latent_and_loss(
                    obs_rep=enc_obs.clone(),
                    target_latents=latent,
                )
                
                actor_loss = q_loss + self.alpha * bc_loss

                bc_loss_all = bc_loss_all + bc_loss
                q_loss_all = q_loss_all + q_loss
                actor_loss_all = actor_loss_all + actor_loss
            else:
                actor_loss = torch.tensor(0.0, device=self.device)
                actor_loss_all = actor_loss_all + actor_loss

            self.step += 1

        if self.step % self.policy_freq == 0:
            self.update_target_networks()

        # Compute means of losses, ensuring they remain tensors with gradients
        losses = {
            'critic_loss': critic_loss_all / (len(obs_1) + len(obs_2)),
            'actor_loss': actor_loss_all / (len(obs_1) + len(obs_2)),
            'qf1_loss': qf1_loss_all / (len(obs_1) + len(obs_2)),
            'qf2_loss': qf2_loss_all / (len(obs_1) + len(obs_2)),
            'bc_loss': bc_loss_all / (len(obs_1) + len(obs_2)) if  (self.step % self.policy_freq == 0) 
                    else torch.tensor(0.0, device=self.device),
        }
        
        return losses

    def train_step(self, batch, optimizers, lr_schedulers, reward_model: nn.Module, stride: int = 1) -> Dict:
        losses = self.compute_loss(batch=batch, reward_model=reward_model, stride=stride)
        
        for opt in optimizers.values():
            opt.zero_grad()
        
        losses['critic_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm=1.0)
        
        if self.step % self.policy_freq == 0:
            losses['actor_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.state_prior.parameters(), max_norm=0.5)
        
        optimizers['qf1'].step()
        optimizers['qf2'].step()
        if self.step % self.policy_freq == 0:
            optimizers['actor'].step()
        
        lr_schedulers['qf1'].step()
        lr_schedulers['qf2'].step()
        if self.step % self.policy_freq == 0:
            for _ in range(self.policy_freq):
                lr_schedulers['actor'].step()

        return losses


class CQLBETLowdimPolicy(BETLowdimPolicy):
    def __init__(self, 
                 action_ae: KMeansDiscretizer, 
                 obs_encoding_net: nn.Module, 
                 state_prior: MinGPT,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 cql_alpha: float = 0.1,
                 num_samples: int = 10,
                 bc_alpha: float = 0.05):
        super().__init__(action_ae, obs_encoding_net, state_prior, horizon, 
                        n_action_steps, n_obs_steps)
        
        self.discount = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.num_samples = num_samples
        self.bc_alpha = bc_alpha
        
        self.qf1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.qf2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.qf1_target = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.qf2_target = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        self.step = 0

    def get_optimizers(self, cfg: DictConfig) -> Dict[str, torch.optim.Optimizer]:
        return {
            'actor': self.state_prior.get_optimizer(**cfg.optimizer.actor),
            'qf1': torch.optim.Adam(self.qf1.parameters(), **cfg.optimizer.qf1),
            'qf2': torch.optim.Adam(self.qf2.parameters(), **cfg.optimizer.qf2),
        }
    
    def update_target_networks(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def compute_cql_loss(self, n_obs: torch.Tensor, n_current_action: torch.Tensor, batch_actions: torch.Tensor) -> torch.Tensor:
        batch_size = n_obs.shape[0]
        sequence_length = batch_actions.shape[1]
        action_dim = n_current_action.shape[-1]

        # print(f"n_obs shape: {n_obs.shape}")
        # print(f"n_current_action shape: {n_current_action.shape}")
        # print(f"batch_actions shape: {batch_actions.shape}")

        idx = torch.randint(0, sequence_length, (batch_size, self.num_samples), device=self.device)
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(batch_size, self.num_samples)
        sampled_actions = batch_actions[batch_idx, idx]
        
        noise = torch.randn(batch_size, self.num_samples, action_dim, device=self.device) * 0.1
        sampled_actions = sampled_actions + noise
        sampled_actions = torch.clamp(sampled_actions, -1, 1)

        # print(f"sampled_actions shape: {sampled_actions.shape}")

        repeated_obs = n_obs.unsqueeze(1).repeat(1, self.num_samples, 1)
        
        q1_sampled = self.qf1(torch.cat([repeated_obs.reshape(-1, n_obs.shape[-1]), 
                                        sampled_actions.reshape(-1, action_dim)], dim=-1))
        q2_sampled = self.qf2(torch.cat([repeated_obs.reshape(-1, n_obs.shape[-1]), 
                                        sampled_actions.reshape(-1, action_dim)], dim=-1))
        
        q1_current = self.qf1(torch.cat([n_obs, n_current_action], dim=-1))
        q2_current = self.qf2(torch.cat([n_obs, n_current_action], dim=-1))
        
        # print(f"q1_sampled mean: {q1_sampled.mean().item()}, std: {q1_sampled.std().item()}")
        # print(f"q1_current mean: {q1_current.mean().item()}, std: {q1_current.std().item()}")

        temperature = 1.0
        q1_cql = (torch.logsumexp(q1_sampled.view(batch_size, self.num_samples) / temperature, dim=1) * temperature - q1_current.squeeze())
        q2_cql = (torch.logsumexp(q2_sampled.view(batch_size, self.num_samples) / temperature, dim=1) * temperature - q2_current.squeeze())
        
        # print(f"q1_cql mean: {q1_cql.mean().item()}, std: {q1_cql.std().item()}")

        cql_alpha = self.cql_alpha * max(0.1, 1.0 - self.step / 10000)
        return cql_alpha * (q1_cql + q2_cql).mean()
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], reward_model: nn.Module, stride: int = 1) -> Dict:
        To = self.n_obs_steps
        Ta = self.n_action_steps
        Th = self.horizon
        batch_size = batch["obs"].shape[0]

        observations_1 = batch["obs"].to(self.device)
        actions_1 = batch["action"].to(self.device)
        observations_2 = batch["obs_2"].to(self.device)
        actions_2 = batch["action_2"].to(self.device)

        batch_1 = {'obs': observations_1, 'action': actions_1}
        batch_2 = {'obs': observations_2, 'action': actions_2}

        nbatch_1 = self.normalizer.normalize(batch_1)
        nbatch_2 = self.normalizer.normalize(batch_2)

        whole_obs_1 = nbatch_1['obs']
        whole_action_1 = nbatch_1['action']
        whole_obs_2 = nbatch_2['obs']
        whole_action_2 = nbatch_2['action']


        obs_1 = slice_episode(whole_obs_1, horizon=max(Th, To+Ta), stride=stride)
        action_1 = slice_episode(whole_action_1, horizon=max(Th, To+Ta), stride=stride)
        obs_2 = slice_episode(whole_obs_2, horizon=max(Th, To+Ta), stride=stride)
        action_2 = slice_episode(whole_action_2, horizon=max(Th, To+Ta), stride=stride)

        critic_losses = []
        actor_losses = []
        qf1_losses = []
        qf2_losses = []
        cql_losses = []

        for i in range(len(obs_1)):
            obs_slide = obs_1[i]
            action_slide = action_1[i]
            nobs = obs_slide[:, :To, :]
            naction = action_slide[:, To-1:To+Ta-1, :]

            # if i == 0 and self.step % 100 == 0:
            #     print(f"Batch 1, iter {i}: nobs shape={nobs.shape}, naction shape={naction.shape}")

            if To == 1 and Ta == 1:
                q1 = self.qf1(torch.cat([obs_slide[:, 0, :], action_slide[:, 0, :]], dim=-1))
                q2 = self.qf2(torch.cat([obs_slide[:, 0, :], action_slide[:, 0, :]], dim=-1))
                cql_loss = self.compute_cql_loss(obs_slide[:, 0, :], 
                                                action_slide[:, 0, :], 
                                                whole_action_1)                
            else:
                q1 = self.qf1(torch.cat([obs_slide[:, To-1:To+Ta-2, :].view(-1, nobs.shape[-1]), 
                                        action_slide[:, To-1:To+Ta-2, :].view(-1, naction.shape[-1])], dim=-1))
                q1 = q1.view(batch_size, Ta, -1)
                q2 = self.qf2(torch.cat([obs_slide[:, To-1:To+Ta-2, :].view(-1, nobs.shape[-1]), 
                                        action_slide[:, To-1:To+Ta-2, :].view(-1, naction.shape[-1])], dim=-1))
                q2 = q2.view(batch_size, Ta, -1)
            
                cql_loss = self.compute_cql_loss(obs_slide[:, To-1:To+Ta-2, :], 
                                                action_slide[:, To-1:To+Ta-2, :], 
                                                whole_action_1)

            if To == 1 and Ta == 1:
                with torch.no_grad():
                    reward = reward_model.forward(nobs, naction)
                    next_action_pred = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})['action']
                    next_q1 = self.qf1_target(torch.cat([obs_slide[:, 0, :], 
                                                        self.normalizer['action'].normalize(next_action_pred[:, 0, :])], dim=-1))
                    next_q2 = self.qf2_target(torch.cat([obs_slide[:, 0, :], 
                                                        self.normalizer['action'].normalize(next_action_pred[:, 0, :])], dim=-1))
                    next_q = torch.min(next_q1, next_q2)
                    target_q = reward + self.discount * next_q
                    target_q = torch.clamp(target_q, -10, 10)
                    cql_loss = self.compute_cql_loss(obs_slide[:, 0, :],
                                                    action_slide[:, 0, :], 
                                                    whole_action_1)
            else:
                with torch.no_grad():
                    reward = reward_model.forward(nobs, naction)
                    reward = reward[:, To-1:To+Ta-2, :]
                    next_action_pred = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})['action']
                    next_q1 = self.qf1_target(torch.cat([obs_slide[:, To:To+Ta-1, :].view(-1, nobs.shape[-1]), 
                                                        (self.normalizer['action'].normalize(next_action_pred[:, 1:, :])).view(-1, naction.shape[-1])], dim=-1))
                    next_q1 = next_q1.view(batch_size, Ta, -1)
                    next_q2 = self.qf2_target(torch.cat([obs_slide[:, To:To+Ta-1, :].view(-1, nobs.shape[-1]), 
                                                        (self.normalizer['action'].normalize(next_action_pred[:, 1:, :])).view(-1, naction.shape[-1])], dim=-1))
                    next_q2 = next_q2.view(batch_size, Ta, -1)
                    next_q = torch.min(next_q1, next_q2)
                    target_q = reward + self.discount * next_q
                    target_q = torch.clamp(target_q, -10, 10)
                    cql_loss = self.compute_cql_loss(obs_slide[:, To-1:To+Ta-1, :], 
                                                    action_slide[:, To-1:To+Ta-1, :], 
                                                    whole_action_1)

            qf1_loss = F.mse_loss(torch.clamp(q1, -10, 10), target_q)
            qf2_loss = F.mse_loss(torch.clamp(q2, -10, 10), target_q)

            critic_loss = qf1_loss + qf2_loss + cql_loss.mean()

            qf1_losses.append(qf1_loss)
            qf2_losses.append(qf2_loss)
            cql_losses.append(torch.mean(cql_loss))
            critic_losses.append(critic_loss)

            if To == 1 and Ta == 1:
                pred_action_dict = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})
                pred_action = pred_action_dict['action'].clone()
                q_values = self.qf1(torch.cat([nobs[:, 0, :], self.normalizer['action'].normalize(pred_action[:, 0, :])], dim=-1))
                actor_loss = -q_values.mean()
            else:
                pred_action_dict = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})
                pred_action = pred_action_dict['action'].clone()
                q_values = self.qf1(torch.cat([obs_slide[:, To-1:To+Ta-2, :].view(-1, nobs.shape[-1]), 
                                            (self.normalizer['action'].normalize(pred_action[:, :-1, :])).view(-1, naction.shape[-1])], dim=-1))
                actor_loss = -q_values.mean()
            
            obs_slide[:, To:, :] = -2
            obs_input = obs_slide[:, :Th, :].view(obs_slide.size(0), Th, -1)
            enc_obs = self.obs_encoding_net(obs_input)
            action_input = action_slide[:, :Th, :].view(action_slide.size(0), Th, -1)
            latent = self.action_ae.encode_into_latent(action_input, enc_obs)

            _, bc_loss = self.state_prior.get_latent_and_loss(obs_rep=enc_obs.clone(), target_latents=latent)
            actor_loss = actor_loss + self.bc_alpha * max(0.1, 1.0 - self.step / 10000) * bc_loss

            actor_losses.append(actor_loss)
            self.step += 1

            if self.step % 2 == 0:
                self.update_target_networks()

        for i in range(len(obs_2)):
            obs_slide = obs_2[i]
            action_slide = action_2[i]
            nobs = obs_slide[:, :To, :]
            naction = action_slide[:, To-1:To+Ta-1, :]

            # if i == 0 and self.step % 100 == 0:
            #     print(f"Batch 1, iter {i}: nobs shape={nobs.shape}, naction shape={naction.shape}")
            if To == 1 and Ta == 1:
                with torch.no_grad():
                    q1 = self.qf1(torch.cat([obs_slide[:, 0, :].view(-1, nobs.shape[-1]),
                                            action_slide[:, 0, :].view(-1, naction.shape[-1])], dim=-1))
                    q1 = q1.view(batch_size, Ta, -1)
                    q2 = self.qf2(torch.cat([obs_slide[:, 0, :].view(-1, nobs.shape[-1]), 
                                                action_slide[:, 0, :].view(-1, naction.shape[-1])], dim=-1))
                    q2 = q2.view(batch_size, Ta, -1)
                    cql_loss = self.compute_cql_loss(obs_slide[:, 0, :], 
                                                    action_slide[:, 0, :], 
                                                    whole_action_2)

            else:
                with torch.no_grad():
                    q1 = self.qf1(torch.cat([obs_slide[:, To-1:To+Ta-2, :].view(-1, nobs.shape[-1]), 
                                            action_slide[:, To-1:To+Ta-2, :].view(-1, naction.shape[-1])], dim=-1))
                    q1 = q1.view(batch_size, Ta, -1)
                    q2 = self.qf2(torch.cat([obs_slide[:, To-1:To+Ta-2, :].view(-1, nobs.shape[-1]), 
                                            action_slide[:, To-1:To+Ta-2, :].view(-1, naction.shape[-1])], dim=-1))
                    q2 = q2.view(batch_size, Ta, -1)

                    cql_loss = self.compute_cql_loss(obs_slide[:, To-1:To+Ta-2, :], 
                                                    action_slide[:, To-1:To+Ta-2, :], 
                                                    whole_action_2)

            if To == 1 and Ta == 1:
                with torch.no_grad():
                    reward = reward_model.forward(nobs, naction)
                    next_action_pred = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})['action']
                    next_q1 = self.qf1_target(torch.cat([obs_slide[:, 0, :], 
                                                        self.normalizer['action'].normalize(next_action_pred[:, 0, :])], dim=-1))
                    next_q2 = self.qf2_target(torch.cat([obs_slide[:, 0, :], 
                                                        self.normalizer['action'].normalize(next_action_pred[:, 0, :])], dim=-1))
                    next_q = torch.min(next_q1, next_q2)
                    target_q = reward + self.discount * next_q
                    target_q = torch.clamp(target_q, -10, 10)
                    cql_loss = self.compute_cql_loss(obs_slide[:, 0, :],
                                                    action_slide[:, 0, :], 
                                                    whole_action_1)
            else:
                with torch.no_grad():
                    reward = reward_model.forward(nobs, naction)
                    reward = reward[:, To-1:To+Ta-2, :]
                    next_action_pred = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})['action']
                    next_q1 = self.qf1_target(torch.cat([obs_slide[:, To:To+Ta-1, :].view(-1, nobs.shape[-1]), 
                                            (self.normalizer['action'].normalize(next_action_pred[:, 1:, :])).view(-1, naction.shape[-1])], dim=-1))
                    next_q1 = next_q1.view(batch_size, Ta, -1)
                    next_q2 = self.qf2_target(torch.cat([obs_slide[:, To:To+Ta-1, :].view(-1, nobs.shape[-1]), 
                                            (self.normalizer['action'].normalize(next_action_pred[:, 1:, :])).view(-1, naction.shape[-1])], dim=-1))
                    next_q2 = next_q2.view(batch_size, Ta, -1)
                    next_q = torch.min(next_q1, next_q2)
                    target_q = reward + self.discount * next_q
                    target_q = torch.clamp(target_q, -10, 10)
                    cql_loss = self.compute_cql_loss(obs_slide[:, To-1:To+Ta-1, :], 
                                                action_slide[:, To-1:To+Ta-1, :], 
                                                whole_action_2)

            qf1_loss = F.mse_loss(torch.clamp(q1, -10, 10), target_q)
            qf2_loss = F.mse_loss(torch.clamp(q2, -10, 10), target_q)
            
            critic_loss = qf1_loss + qf2_loss + cql_loss.mean()

            qf1_losses.append(qf1_loss)
            qf2_losses.append(qf2_loss)
            cql_losses.append(torch.mean(cql_loss))
            critic_losses.append(critic_loss)

            if To == 1 and Ta == 1:
                pred_action_dict = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})
                pred_action = pred_action_dict['action'].clone()
                q_values = self.qf1(torch.cat([nobs[:, 0, :], self.normalizer['action'].normalize(pred_action[:, 0, :])], dim=-1))
                actor_loss = -q_values.mean()
            else:
                pred_action_dict = self.predict_action({'obs': self.normalizer['obs'].unnormalize(nobs.clone())})
                pred_action = pred_action_dict['action'].clone()
                q_values = self.qf1(torch.cat([obs_slide[:, To-1:To+Ta-2, :].view(-1, nobs.shape[-1]), 
                                            (self.normalizer['action'].normalize(pred_action[:, :-1, :])).view(-1, naction.shape[-1])], dim=-1))
                actor_loss = -q_values.mean()
                
            obs_slide[:, To:, :] = -2
            obs_input = obs_slide[:, :Th, :].view(obs_slide.size(0), Th, -1)
            enc_obs = self.obs_encoding_net(obs_input)
            action_input = action_slide[:, :Th, :].view(action_slide.size(0), Th, -1)
            latent = self.action_ae.encode_into_latent(action_input, enc_obs)

            _, bc_loss = self.state_prior.get_latent_and_loss(obs_rep=enc_obs.clone(), target_latents=latent)
            actor_loss = actor_loss + self.bc_alpha * max(0.1, 1.0 - self.step / 10000) * bc_loss

            actor_losses.append(actor_loss)
            self.step += 1

            if self.step % 2 == 0:
                self.update_target_networks()

        losses = {
            'critic_loss': torch.mean(torch.stack(critic_losses)),
            'actor_loss': torch.mean(torch.stack(actor_losses)),
            'qf1_loss': torch.mean(torch.stack(qf1_losses)),
            'qf2_loss': torch.mean(torch.stack(qf2_losses)),
            'cql_loss': torch.mean(torch.stack(cql_losses)),
        }
        
        return losses

    def train_step(self, batch, optimizers, lr_schedulers, reward_model: nn.Module, stride: int = 1):
        losses = self.compute_loss(batch=batch, reward_model=reward_model, stride=stride)
        
        for opt in optimizers.values():
            opt.zero_grad()
        
        losses['critic_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), max_norm=1.0)
        
        losses['actor_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.state_prior.parameters(), max_norm=0.5)
        
        optimizers['qf1'].step()
        optimizers['qf2'].step()
        optimizers['actor'].step()
        
        lr_schedulers['qf1'].step()
        lr_schedulers['qf2'].step()
        lr_schedulers['actor'].step()

        return losses