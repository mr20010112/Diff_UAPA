from typing import Dict, Tuple
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F
import einops
from typing import Optional, Tuple

from diffusion_policy.common.reward_model import RewardModel
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.libraries.batch_loss_fn import BatchFocalLoss, soft_cross_entropy
from diffusion_policy.model.bet.utils import eval_mode
from diffusion_policy.model.common.slice import slice_episode

class BETLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            action_ae: KMeansDiscretizer, 
            obs_encoding_net: nn.Module, 
            state_prior: MinGPT,
            reward_model: nn.Module,
            gamma,
            horizon,
            n_action_steps,
            n_obs_steps,
            map_ratio=0.1,
            bias_reg=0.0,
            beta=1.0):
        super().__init__()
    
        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        self.obs_encoding_net = obs_encoding_net
        self.state_prior = state_prior
        self.reward_model = reward_model
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.gamma = gamma
        self.beta = beta

    def train_reward_model(self, batch, optimizer):
        """
        训练奖励模型。
        
        参数:
            batch: 包含 obs, action, obs_2, action_2, votes, votes_2 的字典
            optimizer: 奖励模型的优化器
        """
        obs_1 = batch['obs'].to(self.device)
        action_1 = batch['action'].to(self.device)
        obs_2 = batch['obs_2'].to(self.device)
        action_2 = batch['action_2'].to(self.device)
        votes_1 = batch['votes'].to(self.device)
        votes_2 = batch['votes_2'].to(self.device)

        # 计算两组轨迹的奖励
        reward_1 = self.reward_model(obs_1, action_1).sum(dim=1)  # (B,)
        reward_2 = self.reward_model(obs_2, action_2).sum(dim=1)  # (B,)

        # Bradley-Terry 模型损失
        logits = reward_1 - reward_2
        preference_loss = -torch.mean(
            votes_1.squeeze() * F.logsigmoid(logits) + votes_2.squeeze() * F.logsigmoid(-logits)
        )

        # 优化
        optimizer.zero_grad()
        preference_loss.backward()
        optimizer.step()

        return preference_loss.item()

    def get_reward_optimizer(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]):
        """获取奖励模型的优化器"""
        return torch.optim.Adam(self.reward_model.parameters(), 
                              lr=learning_rate, 
                              weight_decay=weight_decay, 
                              betas=betas)

    # ========= inference  ============
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
        T = self.horizon

        # pad To to T
        obs = torch.full((B,T,Do), -2, dtype=nobs.dtype, device=nobs.device)
        obs[:,:To,:] = nobs[:,:To,:]

        # (B,T,Do)
        enc_obs = self.obs_encoding_net(obs)

        # Sample latents from the prior
        latents, offsets = self.state_prior.generate_latents(enc_obs)

        # un-descritize
        naction_pred = self.action_ae.decode_actions(
            latent_action_batch=(latents, offsets)
        )
        # (B,T,Da)

        # un-normalize
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
    
    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)
    
    def get_latents(self, latent_collection_loader):
        training_latents = list()
        with eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in latent_collection_loader:
                obs, act = observations.to(self.device, non_blocking=True), action.to(self.device, non_blocking=True)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    training_latents.append(latent.detach())
        training_latents_tensor = torch.cat(training_latents, dim=0)
        return training_latents_tensor

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.state_prior.get_optimizer(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

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

    def compute_loss(self, batch) -> torch.Tensor:
        """基于奖励模型计算策略损失"""
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        obs = slice_episode(obs, horizon=self.horizon, stride=self.n_obs_steps)
        action = slice_episode(action, horizon=self.horizon, stride=self.n_obs_steps)

        total_loss = 0
        for i in range(len(obs)):
            obs_slide = obs[i].to(self.device)
            action_slide = action[i].to(self.device)

            enc_obs = self.obs_encoding_net(obs_slide)
            latents, offsets = self.state_prior.generate_latents(enc_obs)
            pred_action = self.action_ae.decode_actions(latent_action_batch=(latents, offsets))

            rewards = self.reward_model(obs_slide, pred_action)  # (B, T, 1)
            discounted_rewards = torch.tensor([self.gamma**t for t in range(self.horizon)], 
                                            device=self.device) * rewards.squeeze(-1)  # (B, T)
            total_reward = discounted_rewards.sum(dim=1)  # (B,)

            policy_loss = -total_reward.mean()
            total_loss += policy_loss

        return total_loss / len(obs)
