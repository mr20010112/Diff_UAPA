import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision.models as models
import diffusion_policy.common.reward_utils as reward_utils
import logging
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        pass

    return net


class RewardModel(object):
    def __init__(self, task, observation_dim, action_dim, ensemble_size=3, activation="tanh", logger=None,
                 device="cuda"):
        self.task = task
        self.observation_dim = observation_dim  # state: env.observation_space.shape[0]
        self.action_dim = action_dim  # state: env.action_space.shape[0]
        self.ensemble_size = ensemble_size  # ensemble_size
        self.device = torch.device(device)

        # build network
        self.activation = activation
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.construct_ensemble()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # 如果需要输出到控制台
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            model = nn.Sequential(*gen_net(in_size=self.observation_dim + self.action_dim,
                                           out_size=1, H=256, n_layers=3,
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        

    def save_model(self, path):
        state_dicts = [model.state_dict() for model in self.ensemble]
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path, map_location='cpu')
        for model, state_dict in zip(self.ensemble, state_dicts):
            model.load_state_dict(state_dict)
            model.to(self.device)

    def train(self, pref_dataset, data_size, batch_size, n_epochs=1, warm_up_epochs=0, lr=1.0e-4):

        interval = math.ceil(data_size / batch_size)
        total_steps = n_epochs * interval
        warm_up_steps = warm_up_epochs * interval
        main_steps = total_steps - warm_up_steps

        self.lr = lr
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr, weight_decay=0)
        warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
        cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
        self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

        for epoch in range(1, n_epochs + 1):
            ensemble_losses = [[] for _ in range(self.ensemble_size)]
            ensemble_acc = [[] for _ in range(self.ensemble_size)]

            batch_shuffled_idx = []
            for _ in range(self.ensemble_size):
                batch_shuffled_idx.append(np.random.permutation(pref_dataset["observations"].shape[0]))

            for i in range(interval):
                self.opt.zero_grad()
                total_loss = 0
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, pref_dataset["observations"].shape[0])
                for member in range(self.ensemble_size):
                    # get batch
                    batch = index_batch(pref_dataset, batch_shuffled_idx[member][start_pt:end_pt])
                    # compute loss
                    curr_loss, correct = self._train(batch, member)
                    total_loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    ensemble_acc[member].append(correct)
                total_loss.backward()
                self.opt.step()
                self.scheduler.step()

            train_metrics = {"epoch": epoch,
                             "avg_loss": np.mean(ensemble_losses),
                             "avg_acc": np.mean(ensemble_acc)}
            for i in range(self.ensemble_size):
                train_metrics.update({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                train_metrics.update({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
            self.logger.info(f"Training metrics: {train_metrics}")

            # early stop
            if np.mean(ensemble_acc) > 0.968 and "antmaze" not in self.task:
                break

    def r3m_train(self, pref_dataset, data_size, batch_size, n_epochs=1, warm_up_epochs=0, lr=1.0e-4):
        interval = math.ceil(data_size / batch_size)
        total_steps = n_epochs * interval
        warm_up_steps = warm_up_epochs * interval
        main_steps = total_steps - warm_up_steps

        self.lr = lr
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr, weight_decay=1e-5)
        warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
        cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
        self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

        for epoch in range(1, n_epochs + 1):
            ensemble_losses = [[] for _ in range(self.ensemble_size)]
            ensemble_acc = [[] for _ in range(self.ensemble_size)]

            batch_shuffled_idx = []
            for _ in range(self.ensemble_size):
                batch_shuffled_idx.append(np.random.permutation(pref_dataset["observations"].shape[0]))

            for i in range(interval):
                self.opt.zero_grad()
                total_loss = 0
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, pref_dataset["observations"].shape[0])
                for member in range(self.ensemble_size):
                    # get batch
                    batch = index_batch(pref_dataset, batch_shuffled_idx[member][start_pt:end_pt])
                    # compute loss
                    curr_loss, correct = self._r3m_train(batch, member)
                    total_loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    ensemble_acc[member].append(correct)
                total_loss.backward()
                self.opt.step()
                self.scheduler.step()

            train_metrics = {"epoch": epoch,
                             "avg_loss": np.mean(ensemble_losses),
                             "avg_acc": np.mean(ensemble_acc)}
            for i in range(self.ensemble_size):
                train_metrics.update({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                train_metrics.update({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
            self.logger.info(f"Training metrics: {train_metrics}")

            # early stop
            if np.mean(ensemble_acc) > 0.968 and "antmaze" not in self.task:
                break

    def _train(self, batch, member):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)
        s_a_1 = np.concatenate([obs_1, act_1], axis=-1)
        s_a_2 = np.concatenate([obs_2, act_2], axis=-1)

        # get comparable labels
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)

        # get logits
        r_hat1 = self.r_hat_member(s_a_1, member)  # batch_size * len_query * 1
        r_hat2 = self.r_hat_member(s_a_2, member)
        r_hat1 = r_hat1.sum(axis=1)  # batch_size * 1
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=1)  # batch_size * 2

        # get labels
        # labels = torch.from_numpy(labels).long().to(self.device)  # TODO
        labels = torch.from_numpy(labels).to(self.device)

        # compute loss
        curr_loss = self.softXEnt_loss(r_hat, labels)

        # compute acc
        _, predicted = torch.max(r_hat.data, 1)

        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)
        return curr_loss, correct
    
    def _r3m_train(self, batch, member, lambda_reg=0.8):
        """
        使用 R³M 交替优化算法训练奖励模型。
        
        参数：
        - batch: 包含 observations, actions, observations_2, actions_2, labels 的字典
        - member: 集成模型中的索引
        - lambda_reg: R³M 中的正则化参数 λ，控制 δ 的稀疏性
        
        返回：
        - curr_loss: 当前批次的损失
        - correct: 预测准确率
        """
        # 获取批次数据
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot)

        # 转换为 PyTorch 张量
        obs_1 = reward_utils.to_torch(obs_1).to(self.device)
        act_1 = reward_utils.to_torch(act_1).to(self.device)
        obs_2 = reward_utils.to_torch(obs_2).to(self.device)
        act_2 = reward_utils.to_torch(act_2).to(self.device)
        labels = reward_utils.to_torch(labels, dtype=torch.float32).to(self.device)  # batch_size * 2

        # 获取可比较的标签
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.argmax(labels, dim=1).to(self.device)  # batch_size

        # 获取奖励预测
        r_hat1 = self.ensemble[member](obs_1, act_1)  # batch_size * len_query
        r_hat2 = self.ensemble[member](obs_2, act_2)  # batch_size * len_query
        r_hat1 = r_hat1.mean(dim=-1, keepdim=True)  # batch_size * 1
        r_hat2 = r_hat2.mean(dim=-1, keepdim=True)  # batch_size * 1

        # 初始化扰动因子 δ
        batch_size = obs_1.shape[0]
        delta = torch.zeros(batch_size, 1, device=self.device, requires_grad=False)  # batch_size * 1

        # 步骤 1：固定奖励模型参数，更新 δ
        with torch.no_grad():
            delta_r = r_hat1 - r_hat2  # batch_size * 1
            delta_update = torch.log(1.0 / lambda_reg - 1.0) - delta_r  # R³M 闭式解
            delta = torch.max(delta_update, torch.zeros_like(delta))  # 确保 δ >= 0

        # 步骤 2：固定 δ，计算损失并准备更新奖励模型参数
        r_diff = r_hat1 - r_hat2 + delta  # batch_size * 1，加入扰动因子
        p_1_2 = torch.sigmoid(r_diff)  # batch_size * 1，偏好概率
        y = labels[:, :1]  # batch_size * 1，取第一个标签（1 表示偏好 obs_1）

        # 计算加权交叉熵损失
        weights = torch.ones_like(y)
        weights[torch.where(y == 0.5)] = 0.0  # 对于无明确偏好的样本，权重为 0
        log_p_1_2 = torch.log(p_1_2 + 1e-8)  # 避免 log(0)
        log_1_minus_p_1_2 = torch.log(1 - p_1_2 + 1e-8)
        ce_loss = - (weights * (y * log_p_1_2 + (1 - y) * log_1_minus_p_1_2)).mean()

        # 添加 δ 的 L1 正则化项
        l1_loss = lambda_reg * delta.abs().mean()
        curr_loss = ce_loss + l1_loss

        # 计算准确率
        r_hat = torch.cat([r_hat1, r_hat2], dim=-1)  # batch_size * 2
        _, predicted = torch.max(r_hat.data, dim=1)  # 不使用 δ 的原始预测用于评估
        if not len(comparable_indices):
            correct = 0.7  # 与原代码保持一致
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(comparable_indices)

        return curr_loss, correct

    def r_hat_member(self, x, member):
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def get_reward_batch(self, x):
        # they say they average the rewards from each member of the ensemble,
        # but I think this only makes sense if the rewards are already normalized.
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def softXEnt_loss(self, input, target):
        logprobs = nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]


class TransformerRewardModel(RewardModel):
    def __init__(self,
                 task, observation_dim, action_dim, structure_type="transformer1",
                 ensemble_size=3, activation="tanh",
                 d_model=256, nhead=4, num_layers=1, max_seq_len=100,
                 logger=None, device="cuda"):
        self.structure_type = structure_type
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        super().__init__(
            task, observation_dim, action_dim,
            ensemble_size, activation, logger, device)
        
    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            if self.structure_type == "transformer1":
                transformer = reward_utils.PrefTransformer1(
                    self.observation_dim, self.action_dim,
                    self.max_seq_len,
                    self.d_model, self.nhead, self.num_layers
                )
            elif self.structure_type == "transformer2":
                transformer = reward_utils.PrefTransformer2(
                    self.observation_dim, self.action_dim,
                    self.d_model, self.nhead, self.num_layers
                )
            elif self.structure_type == "transformer3":
                transformer = reward_utils.PrefTransformer3(
                    self.observation_dim, self.action_dim,
                    self.d_model, self.nhead, self.num_layers
                )
            else:
                raise NotImplementedError

            self.ensemble.append(transformer.to(self.device))
            self.paramlst.extend(self.ensemble[-1].parameters())


    def _train(self, batch, member):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)

        # to_torch
        obs_1 = reward_utils.to_torch(obs_1).to(self.device)
        act_1 = reward_utils.to_torch(act_1).to(self.device)
        obs_2 = reward_utils.to_torch(obs_2).to(self.device)
        act_2 = reward_utils.to_torch(act_2).to(self.device)

        # get comparable labels
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)

        # get logits
        r_hat1 = self.ensemble[member](obs_1, act_1)  # batch_size * len_query
        r_hat2 = self.ensemble[member](obs_2, act_2)
        
        r_hat1 = r_hat1.mean(-1, keepdim=True)  # batch_size * 1
        r_hat2 = r_hat2.mean(-1, keepdim=True)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # batch_size * 2
        
        p_1_2 = 1./(1.+torch.exp(r_hat2-r_hat1)) # batch_size * 1
        y = reward_utils.to_torch(labels[:, :1], dtype=torch.float32).to(self.device) # batch_size * 1

        weights = torch.ones_like(y)
        weights[torch.where(y==0.5)] = 0.0
        
        curr_loss = - (weights*(y*torch.log(p_1_2+1e-8) + (1-y)*torch.log(1-p_1_2+1e-8))).mean()
        
        # labels = utils.to_torch(labels, dtype=torch.long).to(self.device)

        # # compute loss
        # curr_loss = self.softXEnt_loss(r_hat, labels)

        # compute acc
        _, predicted = torch.max(r_hat.data, 1)

        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)
            
        return curr_loss, correct
    
    def _r3m_train(self, batch, member, lambda_reg=0.8):
        """
        使用 R³M 交替优化算法训练奖励模型。
        
        参数：
        - batch: 包含 observations, actions, observations_2, actions_2, labels 的字典
        - member: 集成模型中的索引
        - lambda_reg: R³M 中的正则化参数 λ，控制 δ 的稀疏性
        
        返回：
        - curr_loss: 当前批次的损失
        - correct: 预测准确率
        """
        # 获取批次数据
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot)

        # 转换为 PyTorch 张量
        obs_1 = reward_utils.to_torch(obs_1).to(self.device)
        act_1 = reward_utils.to_torch(act_1).to(self.device)
        obs_2 = reward_utils.to_torch(obs_2).to(self.device)
        act_2 = reward_utils.to_torch(act_2).to(self.device)
        labels = reward_utils.to_torch(labels, dtype=torch.float32).to(self.device)  # batch_size * 2

        # 获取可比较的标签
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.argmax(labels, dim=1).to(self.device)  # batch_size

        # 获取奖励预测
        r_hat1 = self.ensemble[member](obs_1, act_1)  # batch_size * len_query
        r_hat2 = self.ensemble[member](obs_2, act_2)  # batch_size * len_query
        r_hat1 = r_hat1.mean(dim=-1, keepdim=True)  # batch_size * 1
        r_hat2 = r_hat2.mean(dim=-1, keepdim=True)  # batch_size * 1

        # 初始化扰动因子 δ
        batch_size = obs_1.shape[0]
        delta = torch.zeros(batch_size, 1, device=self.device, requires_grad=False)  # batch_size * 1

        # 步骤 1：固定奖励模型参数，更新 δ
        with torch.no_grad():
            delta_r = r_hat1 - r_hat2  # batch_size * 1
            delta_update = torch.log(1.0 / lambda_reg - 1.0) - delta_r  # R³M 闭式解
            delta = torch.max(delta_update, torch.zeros_like(delta))  # 确保 δ >= 0

        # 步骤 2：固定 δ，计算损失并准备更新奖励模型参数
        r_diff = r_hat1 - r_hat2 + delta  # batch_size * 1，加入扰动因子
        p_1_2 = torch.sigmoid(r_diff)  # batch_size * 1，偏好概率
        y = labels[:, :1]  # batch_size * 1，取第一个标签（1 表示偏好 obs_1）

        # 计算加权交叉熵损失
        weights = torch.ones_like(y)
        weights[torch.where(y == 0.5)] = 0.0  # 对于无明确偏好的样本，权重为 0
        log_p_1_2 = torch.log(p_1_2 + 1e-8)  # 避免 log(0)
        log_1_minus_p_1_2 = torch.log(1 - p_1_2 + 1e-8)
        ce_loss = - (weights * (y * log_p_1_2 + (1 - y) * log_1_minus_p_1_2)).mean()

        # 添加 δ 的 L1 正则化项
        l1_loss = lambda_reg * delta.abs().mean()
        curr_loss = ce_loss + l1_loss

        # 计算准确率
        r_hat = torch.cat([r_hat1, r_hat2], dim=-1)  # batch_size * 2
        _, predicted = torch.max(r_hat.data, dim=1)  # 不使用 δ 的原始预测用于评估
        if not len(comparable_indices):
            correct = 0.7  # 与原代码保持一致
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(comparable_indices)

        return curr_loss, correct