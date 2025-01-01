from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.common.slice import slice_episode

class IbcDfoLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            dropout=0.1,
            train_n_neg=128,
            pred_n_iter=5,
            pred_n_samples=16384,
            kevin_inference=False,
            andy_train=False
        ):
        super().__init__()

        in_action_channels = action_dim * n_action_steps
        in_obs_channels = obs_dim * n_obs_steps
        in_channels = in_action_channels + in_obs_channels
        mid_channels = 1024
        out_channels = 1

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        self.normalizer = LinearNormalizer()

        self.train_n_neg = train_n_neg
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.kevin_inference = kevin_inference
        self.andy_train = andy_train
    
    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape
        s = obs.reshape(B,1,-1).expand(-1,N,-1)
        x = torch.cat([s, action.reshape(B,N,-1)], dim=-1).reshape(B*N,-1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B,N)
        return x

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
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        Ta = self.n_action_steps

        # only take necessary obs
        this_obs = nobs[:,:To]
        naction_stats = self.get_naction_stats()

        # first sample
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=this_obs.dtype)
        # (B, N, Ta, Da)

        if self.kevin_inference:
            # kevin's implementation
            noise_scale = 3e-2
            for i in range(self.pred_n_iter):
                # Compute energies.
                logits = self.forward(this_obs, samples)
                probs = F.softmax(logits, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=naction_stats['min'], max=naction_stats['max'])

            # Return target with highest probability.
            logits = self.forward(this_obs, samples)
            probs = F.softmax(logits, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            acts_n = samples[torch.arange(samples.size(0)), best_idxs, :]
        else:
            # andy's implementation
            zero = torch.tensor(0, device=self.device)
            resample_std = torch.tensor(3e-2, device=self.device)
            for i in range(self.pred_n_iter):
                # Forward pass.
                logits = self.forward(this_obs, samples) # (B, N)
                prob = torch.softmax(logits, dim=-1)

                if i < (self.pred_n_iter - 1):
                    idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                    samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                    samples += torch.normal(zero, resample_std, size=samples.shape, device=self.device)

            # Return one sample per x in batch.
            idxs = torch.multinomial(prob, num_samples=1, replacement=True)
            acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)

        action = self.normalizer['action'].unnormalize(acts_n)
        result = {
            'action': action
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def action_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算在当前策略下生成特定动作的概率。

        Args:
            policy (IbcDfoLowdimPolicy): 策略对象。
            obs (torch.Tensor): 当前的观察，形状为 (B, To, Do)。
            action (torch.Tensor): 生成的动作，形状为 (B, Ta, Da)。

        Returns:
            torch.Tensor: 动作的概率，形状为 (B,)。
        """
        # 检查输入维度
        B, To, Do = obs.shape
        B_action, Ta, Da = action.shape
        assert B == B_action, "Batch size mismatch between obs and action."
        assert To == self.n_obs_steps, f"Observation steps must be {self.n_obs_steps}, got {To}."
        assert Da == self.action_dim, f"Action dimension must be {self.action_dim}, got {Da}."

        # 归一化观察和动作
        nobs = self.normalizer['obs'].normalize(obs)  # (B, To, Do)
        
        # 为动作扩展维度以与其他样本保持一致
        naction = action.unsqueeze(1)  # (B, 1, Ta, Da)

        # 生成负样本以构造完整的动作样本集
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        neg_samples = action_dist.sample((B, self.train_n_neg, Ta)).to(dtype=naction.dtype, device=naction.device)
        # 合并正样本和负样本
        all_samples = torch.cat([naction, neg_samples], dim=1)  # (B, 1 + train_n_neg, Ta, Da)

        # 计算 logits
        logits = self.forward(nobs, all_samples)  # (B, 1 + train_n_neg)

        # 转为概率分布
        prob = F.softmax(logits, dim=-1)  # (B, 1 + train_n_neg)

        # 返回正样本的概率
        return prob[:, 0] 

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        observations_1 = batch["obs"].to(self.device)
        actions_1 = batch["action"].to(self.device)
        votes_1 = batch["votes"].to(self.device)
        observations_2 = batch["obs_2"].to(self.device)
        actions_2 = batch["action_2"].to(self.device)
        votes_2 = batch["votes_2"].to(self.device)

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

        obs_1 = slice_episode(obs_1, horizon=self.horizon, stride=self.horizon)
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=self.horizon)
        obs_2 = slice_episode(obs_2, horizon=self.horizon, stride=self.horizon)
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=self.horizon)

        # shapes
        Do = self.obs_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        T = self.horizon
        B = action_1.shape[0]

        loss = 0

        for i in range(len(obs_1)):
            obs_1_slide = obs_1[i]
            obs_1_slide = obs_1_slide[:, :self.n_obs_steps, ...]
            action_1_slide = action_1[i]
            action_1_slide = action_1_slide[:, :self.n_action_steps, ...]

            obs_2_slide = obs_2[i]
            obs_2_slide = obs_2_slide[:, :self.n_obs_steps, ...]
            action_2_slide = action_2[i]
            action_2_slide = action_2_slide[:, :self.n_action_steps, ...]

            naction_stats = self.get_naction_stats()
            action_dist = torch.distributions.Uniform(
                low=naction_stats['min'],
                high=naction_stats['max']
            )

            samples = action_dist.sample((B, self.train_n_neg, self.n_action_steps)).to(
                dtype=action_1_slide.dtype)  

            action_1_samples = torch.cat([
                obs_1_slide.unsqueeze(1), samples], dim=1)
            
            action_2_samples = torch.cat([
                obs_2_slide.unsqueeze(1), samples], dim=1)

            prob_1 = self.action_prob(obs_1_slide, action_1_samples)
            prob_2 = self.action_prob(obs_2_slide, action_2_samples)

        
        return loss


    def get_naction_stats(self):
        Da = self.action_dim
        naction_stats = self.normalizer['action'].get_output_stats()
        repeated_stats = dict()
        for key, value in naction_stats.items():
            assert len(value.shape) == 1
            n_repeats = Da // value.shape[0]
            assert value.shape[0] * n_repeats == Da
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
