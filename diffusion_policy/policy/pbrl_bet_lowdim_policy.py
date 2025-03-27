from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.common.slice import slice_episode
import torch
import torch.nn as nn

class BETLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            action_ae: KMeansDiscretizer, 
            obs_encoding_net: nn.Module, 
            state_prior: MinGPT,
            gamma: float,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            map_ratio: float = 0.1,
            bias_reg: float = 0.0,
            beta: float = 1.0):
        super().__init__()
        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        self.obs_encoding_net = obs_encoding_net
        self.state_prior = state_prior
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.gamma = gamma

    def predict_action(self, obs_dict: dict) -> dict:
        """推理方法保持不变"""
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon

        obs = torch.full((B, T, Do), -2, dtype=nobs.dtype, device=nobs.device)
        obs[:, :To, :] = nobs[:, :To, :]

        enc_obs = self.obs_encoding_net(obs)
        latents, offsets = self.state_prior.generate_latents(enc_obs)
        naction_pred = self.action_ae.decode_actions(latent_action_batch=(latents, offsets))
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {'action': action, 'action_pred': action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)

    def get_optimizer(self, weight_decay: float, learning_rate: float, betas: tuple) -> torch.optim.Optimizer:
        return self.state_prior.get_optimizer(weight_decay=weight_decay, learning_rate=learning_rate, betas=betas)

    def compute_loss(self, batch: dict, reward_model: nn.Module, stride: int = 1) -> torch.Tensor:
        """基于奖励模型计算策略损失"""

        observations_1 = batch["obs"].to(self.device)
        actions_1 = batch["action"].to(self.device)
        votes_1 = batch["votes"].to(self.device)
        length_1 = batch["length"].to(self.device).detach()
        observations_2 = batch["obs_2"].to(self.device)
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

        threshold = 1e-2
        diff = torch.abs(votes_1 - votes_2)
        condition_1 = (votes_1 > votes_2) & (diff >= threshold)  # votes_1 > votes_2 and diff >= threshold
        condition_2 = (votes_1 < votes_2) & (diff >= threshold)  # votes_1 < votes_2 and diff >= threshold

        votes_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_1 = torch.squeeze(votes_1, dim=-1).detach()
        votes_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_2 = torch.squeeze(votes_2, dim=-1).detach()

        batch_1 = {
            'obs': torch.tensor(observations_1, device=self.device),
            'action': torch.tensor(actions_1, device=self.device),
        }

        batch_2 = {
            'obs': torch.tensor(observations_2, device=self.device),
            'action': torch.tensor(actions_2, device=self.device),
        }

        nbatch_1 = self.normalizer.normalize(batch_1)
        nbatch_2 = self.normalizer.normalize(batch_2)

        obs_1 = nbatch_1['obs']
        action_1 = nbatch_1['action']
        obs_2 = nbatch_2['obs']
        action_2 = nbatch_2['action']

        obs_1 = slice_episode(obs_1, horizon=self.horizon, stride=stride)
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=stride)
        obs_2 = slice_episode(obs_2, horizon=self.horizon, stride=stride)
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=stride)

        total_loss = 0
        for i in range(len(obs_1)):
            obs_slide = obs_1[i].to(self.device)
            action_slide = action_1[i].to(self.device)

            enc_obs = self.obs_encoding_net(obs_slide)
            latents, offsets = self.state_prior.generate_latents(enc_obs)
            pred_action = self.action_ae.decode_actions(latent_action_batch=(latents, offsets))

            rewards = reward_model(obs_slide, pred_action)  # (B, T, 1)
            discounted_rewards = torch.tensor([self.gamma**t for t in range(self.horizon)], 
                                            device=self.device) * rewards.squeeze(-1)  # (B, T)
            total_reward = discounted_rewards.sum(dim=1)  # (B,)

            policy_loss = -total_reward.mean()
            total_loss += policy_loss

        for i in range(len(obs_2)):
            obs_slide = obs_2[i].to(self.device)
            action_slide = action_2[i].to(self.device)

            enc_obs = self.obs_encoding_net(obs_slide)
            latents, offsets = self.state_prior.generate_latents(enc_obs)
            pred_action = self.action_ae.decode_actions(latent_action_batch=(latents, offsets))

            rewards = reward_model(obs_slide, pred_action)  # (B, T, 1)
            discounted_rewards = torch.tensor([self.gamma**t for t in range(self.horizon)], 
                                            device=self.device) * rewards.squeeze(-1)  # (B, T)
            total_reward = discounted_rewards.sum(dim=1)  # (B,)

            policy_loss = -total_reward.mean()
            total_loss += policy_loss


        return total_loss / (len(obs_1)+len(obs_2))