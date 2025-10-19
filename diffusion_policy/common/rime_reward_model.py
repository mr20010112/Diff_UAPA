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
from tqdm import tqdm
from collections import deque

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

class RunningStat:
    def __init__(self, max_len=1000):
        self.queue = deque(maxlen=max_len)
        self._mean = 0
        self._var = 0
        self._count = 0

    def update(self, x):
        self.queue.extend(x)
        self._mean = np.mean(self.queue)
        self._var = np.var(self.queue) if len(self.queue) > 1 else 0
        self._count = len(self.queue)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def max(self):
        return np.max(self.queue) if self.queue else 0

class RewardModel(object):
    def __init__(self, task, observation_dim, action_dim, ensemble_size=3, activation="tanh", logger=None, device="cuda"):
        self.task = task
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.de = ensemble_size  # Use 'de' to match train_reward terminology
        self.device = torch.device(device)
        self.reward_mean = 0
        self.reward_std = 1

        # RIME-specific attributes
        self.capacity = 10000  # Buffer capacity
        self.buffer_index = 0
        self.buffer_full = False
        self.buffer_seg1 = None
        self.buffer_seg2 = None
        self.buffer_label = None
        self.label_margin = 0.1
        self.teacher_eps_equal = 0.1
        self.threshold_alpha = 0.5
        self.threshold_beta = 1.0
        self.threshold_variance = 'prob'
        self.flipping_tau = 0.1
        self.label_target = 1.0
        self.KL_div = RunningStat(max_len=1000)
        self.update_step = 0
        self.train_batch_size = 256

        # Build network
        self.activation = activation
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.scheduler = None
        self.construct_ensemble()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def compute_global_reward_stats(self, pref_dataset):
        all_obs = np.concatenate([pref_dataset["observations"], pref_dataset["observations_2"]], axis=0)
        all_act = np.concatenate([pref_dataset["actions"], pref_dataset["actions_2"]], axis=0)
        rewards = []
        for member in range(self.de):
            r_hat = self.r_hat_member(
                batch_obs=torch.from_numpy(all_obs).float().to(self.device),
                batch_act=torch.from_numpy(all_act).float().to(self.device),
                member=member
            ).detach().cpu().numpy()
            rewards.append(r_hat)
        rewards = np.concatenate(rewards, axis=0)
        self.reward_mean = np.mean(rewards)
        self.reward_std = np.std(rewards) + 1e-6
        self.logger.info(f"Global reward stats - mean: {self.reward_mean}, std: {self.reward_std}")

    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(
                in_size=self.observation_dim + self.action_dim,
                out_size=1, H=256, n_layers=3,
                activation=self.activation
            )).float().to(self.device)
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

    def initialize_buffer(self, pref_dataset):
        max_len = pref_dataset["observations"].shape[0]
        self.buffer_seg1 = torch.from_numpy(np.concatenate([pref_dataset["observations"], pref_dataset["actions"]], axis=-1)).float().to(self.device)
        self.buffer_seg2 = torch.from_numpy(np.concatenate([pref_dataset["observations_2"], pref_dataset["actions_2"]], axis=-1)).float().to(self.device)
        self.buffer_label = np.argmax(pref_dataset["labels"], axis=1)
        self.buffer_index = max_len
        self.buffer_full = max_len >= self.capacity

    def get_threshold_beta(self):
        return self.threshold_beta

    def train_rime(self, pref_dataset, data_size, batch_size, n_epochs=1, warm_up_epochs=0, lr=1.0e-4, debug=False, trust_sample=True, label_flipping=True):
        self.train_batch_size = batch_size
        self.initialize_buffer(pref_dataset)
        
        interval = math.ceil(data_size / batch_size)
        total_steps = n_epochs * interval
        warm_up_steps = warm_up_epochs * interval
        main_steps = total_steps - warm_up_steps

        self.lr = lr
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr, weight_decay=0)
        warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
        cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
        self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

        max_len = self.capacity if self.buffer_full else self.buffer_index

        for epoch in range(1, n_epochs + 1):
            ensemble_losses = [[] for _ in range(self.de)]
            ensemble_acc = np.array([0 for _ in range(self.de)])
            
            # Compute trust samples
            p_hat_all = []
            with torch.no_grad():
                for member in range(self.de):
                    r_hat1 = self.r_hat_member(self.buffer_seg1[:max_len], member=member)
                    r_hat2 = self.r_hat_member(self.buffer_seg2[:max_len], member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                    r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # (max_len, 2)
                    p_hat_all.append(F.softmax(r_hat, dim=-1).cpu())
            
            # Predict label for all ensemble members
            p_hat_all = torch.stack(p_hat_all)  # (de, max_len, 2)
            predict_label = p_hat_all.mean(0)  # (max_len, 2)

            # Compute KL divergence
            if self.label_margin > 0 or self.teacher_eps_equal > 0:
                buffer_label = torch.tensor(self.buffer_label[:max_len].flatten()).long()
                target_label = torch.zeros_like(predict_label)
                temp_buffer_label = torch.clamp(buffer_label, min=0)
                target_label.scatter_(1, temp_buffer_label.unsqueeze(1), 1)
                mask = buffer_label == -1
                target_label[mask, :] = 0.5
            else:
                target_label = torch.zeros_like(predict_label).scatter_(
                    1, torch.from_numpy(self.buffer_label[:max_len].flatten()).long().unsqueeze(1), 1
                )
            
            KL_div = (-target_label * torch.log(predict_label + 1e-8)).sum(1)  # (max_len,)
            
            # Filter trust samples
            x = self.KL_div.max
            baseline = -np.log(x + 1e-8) + self.threshold_alpha * x
            if self.threshold_variance == 'prob':
                uncertainty = self.get_threshold_beta() * predict_label[:, 0].std(0)
            else:
                uncertainty = min(self.get_threshold_beta() * self.KL_div.var, 3.0)
            trust_sample_bool_index = KL_div < baseline + uncertainty
            trust_sample_index = np.where(trust_sample_bool_index)[0]

            # Label flipping
            flipping_threshold = -np.log(self.flipping_tau)
            flipping_sample_bool_index = KL_div > flipping_threshold
            flipping_sample_index = np.where(flipping_sample_bool_index)[0]
            
            # Update KL divergence statistics
            self.KL_div.update(KL_div[trust_sample_bool_index].cpu().numpy())

            # Determine training samples
            if trust_sample and label_flipping:
                self.buffer_label[flipping_sample_index] = 1 - self.buffer_label[flipping_sample_index]
                training_sample_index = np.concatenate([trust_sample_index, flipping_sample_index])
            elif not trust_sample and label_flipping:
                self.buffer_label[flipping_sample_index] = 1 - self.buffer_label[flipping_sample_index]
                training_sample_index = np.arange(max_len)
            elif trust_sample and not label_flipping:
                training_sample_index = trust_sample_index
            else:
                training_sample_index = np.arange(max_len)

            max_len = len(training_sample_index)
            total_batch_index = [np.random.permutation(training_sample_index) for _ in range(self.de)]
            num_epochs = int(np.ceil(max_len / self.train_batch_size))
            total = 0

            for sub_epoch in tqdm(range(num_epochs), desc=f'RIME Training: Epoch {epoch}'):
                self.opt.zero_grad()
                loss = 0.0
                last_index = (sub_epoch + 1) * self.train_batch_size
                if last_index > max_len:
                    last_index = max_len

                for member in range(self.de):
                    idxs = total_batch_index[member][sub_epoch * self.train_batch_size:last_index]
                    sa_t_1 = self.buffer_seg1[idxs]
                    sa_t_2 = self.buffer_seg2[idxs]
                    labels = self.buffer_label[idxs]
                    labels = torch.from_numpy(labels.flatten()).long().to(self.device)

                    if member == 0:
                        total += labels.size(0)

                    r_hat1 = self.r_hat_member(sa_t_1, member=member)
                    r_hat2 = self.r_hat_member(sa_t_2, member=member)
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                    r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                    if self.label_margin > 0 or self.teacher_eps_equal > 0:
                        uniform_index = labels == -1
                        labels[uniform_index] = 0
                        target_onehot = torch.zeros_like(r_hat).scatter_(1, labels.unsqueeze(1), self.label_target)
                        target_onehot += self.label_margin
                        if uniform_index.int().sum().item() > 0:
                            target_onehot[uniform_index] = 0.5
                        curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                    else:
                        curr_loss = F.cross_entropy(r_hat, labels)
                    loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())

                    _, predicted = torch.max(r_hat.data, 1)
                    correct = (predicted == labels).sum().item()
                    ensemble_acc[member] += correct

                loss.backward()
                self.opt.step()
                self.scheduler.step()

            if label_flipping:
                self.buffer_label[flipping_sample_index] = 1 - self.buffer_label[flipping_sample_index]

            ensemble_acc = ensemble_acc / total
            train_metrics = {
                "epoch": epoch,
                "avg_loss": np.mean([np.mean(losses) for losses in ensemble_losses]),
                "avg_acc": np.mean(ensemble_acc)
            }
            for i in range(self.de):
                train_metrics.update({
                    f"ensemble_{i}_loss": np.mean(ensemble_losses[i]),
                    f"ensemble_{i}_acc": ensemble_acc[i] / total
                })
            self.logger.info(f"RIME Training metrics: {train_metrics}")

            if np.mean(ensemble_acc) > 0.968 and "antmaze" not in self.task:
                break

        with torch.no_grad():
            self.compute_global_reward_stats(pref_dataset)

    def r_hat_member(self, batch_obs, batch_act=None, member=0):
        if batch_act is None:  # Input is concatenated state-action
            return self.ensemble[member](batch_obs)
        return self.ensemble[member](torch.cat([batch_obs, batch_act], dim=-1))

    def get_reward_batch(self, batch_obs, batch_act):
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(
                batch_obs=batch_obs,
                batch_act=batch_act,
                member=member
            ).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        r_mean = np.mean(r_hats, axis=0)
        
        if self.reward_mean is None or self.reward_std is None:
            self.logger.warning("Reward stats not computed, returning raw rewards!")
            return r_mean
        
        normalized_r = (r_mean - self.reward_mean) / self.reward_std
        return normalized_r

    def softXEnt_loss(self, input, target):
        logprobs = F.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def forward(self, nobs, nactions):
        with torch.no_grad():
            reward = torch.mean(torch.stack([
                self.ensemble[i](torch.cat([nobs, nactions], dim=-1))
                for i in range(self.de)
            ]), dim=0)
        return (reward - self.reward_mean) / self.reward_std

class TransformerRewardModel(RewardModel):
    def __init__(self, task, observation_dim, action_dim, structure_type="transformer1",
                 ensemble_size=3, activation="tanh", d_model=256, nhead=4, num_layers=1,
                 max_seq_len=100, logger=None, device="cuda"):
        super().__init__(task, observation_dim, action_dim, ensemble_size, activation, logger, device)
        self.structure_type = structure_type
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

    def construct_ensemble(self):
        self.ensemble = []
        self.paramlst = []
        for i in range(self.de):
            if self.structure_type == "transformer1":
                transformer = reward_utils.PrefTransformer1(
                    self.observation_dim, self.action_dim,
                    self.max_seq_len, self.d_model, self.nhead, self.num_layers
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

    def r_hat_member(self, batch_obs, batch_act=None, member=0):
        if batch_act is None:  # Input is concatenated state-action
            return self.ensemble[member](batch_obs)
        return self.ensemble[member](batch_obs, batch_act)