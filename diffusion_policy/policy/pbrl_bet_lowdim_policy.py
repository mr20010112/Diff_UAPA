
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.policy.bet_lowdim_policy import BETLowdimPolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.utils import eval_mode

class TD3BCBETPolicy(BETLowdimPolicy):
    def __init__(self, 
                 action_ae: KMeansDiscretizer, 
                 obs_encoding_net: nn.Module, 
                 state_prior: MinGPT,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 reward_model: nn.Module,  # 已训练的奖励模型
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_freq: int = 2,
                 alpha: float = 2.5):  # BC正则化权重
        super().__init__(action_ae, obs_encoding_net, state_prior, horizon, 
                        n_action_steps, n_obs_steps)
        
        self.reward_model = reward_model
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        
        # Critic 网络
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
        
        # 目标网络
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
        
        # 初始化目标网络
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        self.step = 0

    def get_optimizers(self, lr: float, weight_decay: float) -> Dict[str, torch.optim.Optimizer]:
        return {
            'actor': torch.optim.Adam(self.state_prior.parameters(), lr=lr, weight_decay=weight_decay),
            'qf1': torch.optim.Adam(self.qf1.parameters(), lr=lr, weight_decay=weight_decay),
            'qf2': torch.optim.Adam(self.qf2.parameters(), lr=lr, weight_decay=weight_decay)
        }

    def update_target_networks(self):
        """软更新目标网络"""
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def compute_loss(self, batch: Dict[str, torch.Tensor], reward_model: nn.Module) -> Tuple[torch.Tensor, Dict]:
        """
        batch: 包含 'obs', 'action', 'next_obs' 的离线数据集
        """
        obs = batch['obs'][:, :self.n_obs_steps, :]  # (B, To, Do)
        action = batch['action']  # (B, T, Da)
        next_obs = batch['next_obs'][:, :self.n_obs_steps, :]  # (B, To, Do)

        # 归一化输入
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs'][:, :self.n_obs_steps, :]
        naction = nbatch['action']
        n_next_obs = nbatch['next_obs'][:, :self.n_obs_steps, :]

        # 计算当前 Q 值
        q1 = self.qf1(torch.cat([nobs[:, -1, :], naction[:, self.n_obs_steps-1, :]], dim=-1))
        q2 = self.qf2(torch.cat([nobs[:, -1, :], naction[:, self.n_obs_steps-1, :]], dim=-1))

        # 使用奖励模型计算奖励
        reward = reward_model(obs, action)  # 假设奖励模型接收 (state, action)

        # 获取下一动作
        with torch.no_grad():
            next_action_pred = self.predict_action({'obs': next_obs})['action']
            noise = torch.clamp(
                self.policy_noise * torch.randn_like(next_action_pred),
                -self.noise_clip, self.noise_clip
            )
            next_action = next_action_pred + noise

            # 计算目标 Q 值
            next_q1 = self.qf1_target(torch.cat([n_next_obs[:, -1, :], next_action[:, -1, :]], dim=-1))
            next_q2 = self.qf2_target(torch.cat([n_next_obs[:, -1, :], next_action[:, -1, :]], dim=-1))
            next_q = torch.min(next_q1, next_q2)
            target_q = reward + self.discount * next_q

        # Critic 损失
        qf1_loss = F.mse_loss(q1, target_q)
        qf2_loss = F.mse_loss(q2, target_q)
        critic_loss = qf1_loss + qf2_loss

        # Actor 损失（TD3 + BC）
        if self.step % self.policy_freq == 0:
            pred_action_dict = self.predict_action({'obs': obs})
            pred_action = pred_action_dict['action']
            
            # Q 值损失
            q_values = self.qf1(torch.cat([nobs[:, -1, :], pred_action[:, -1, :]], dim=-1))
            q_loss = -q_values.mean()

            # BC 损失
            bc_loss = F.mse_loss(pred_action, action[:, self.n_obs_steps-1:self.n_obs_steps-1+self.n_action_steps, :])
            
            # 总 Actor 损失
            actor_loss = q_loss + self.alpha * bc_loss
        else:
            actor_loss = torch.tensor(0.0, device=obs.device)

        self.step += 1
        if self.step % self.policy_freq == 0:
            self.update_target_networks()

        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'qf1_loss': qf1_loss,
            'qf2_loss': qf2_loss,
            'bc_loss': bc_loss if self.step % self.policy_freq == 0 else torch.tensor(0.0)
        }

    def train_step(self, batch, optimizers):
        losses = self.compute_loss(batch)
        
        # 更新 Critic
        optimizers['qf1'].zero_grad()
        losses['qf1_loss'].backward()
        optimizers['qf1'].step()
        
        optimizers['qf2'].zero_grad()
        losses['qf2_loss'].backward()
        optimizers['qf2'].step()
        
        # 更新 Actor
        if self.step % self.policy_freq == 0:
            optimizers['actor'].zero_grad()
            losses['actor_loss'].backward()
            optimizers['actor'].step()

        return losses