import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # 输入维度是观察和动作的拼接
        input_dim = obs_dim + action_dim
        
        # 定义网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 输出标量奖励
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重，使用正交初始化以提高训练稳定性"""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播，预测奖励。
        
        参数:
            obs (torch.Tensor): 观察张量，形状为 (batch_size, horizon, obs_dim)
            action (torch.Tensor): 动作张量，形状为 (batch_size, horizon, action_dim)
        
        返回:
            torch.Tensor: 奖励张量，形状为 (batch_size, horizon, 1)
        """
        # 确保输入维度匹配
        assert obs.shape[:-1] == action.shape[:-1], "obs 和 action 的 batch_size 和 horizon 必须一致"
        
        # 拼接观察和动作
        x = torch.cat([obs, action], dim=-1)  # (batch_size, horizon, obs_dim + action_dim)
        
        # 通过网络计算奖励
        rewards = self.net(x)  # (batch_size, horizon, 1)
        
        return rewards