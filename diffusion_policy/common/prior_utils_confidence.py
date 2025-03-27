import collections
import numpy as np
# import gym
import torch.utils
from tqdm import trange
import torch
import torch.nn as nn
import math
# import d4rl
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import copy
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

import time
import logging
from datetime import datetime

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


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

def get_discrete_traj(trajectory, dim=2):
    discrete_traj = np.round(trajectory[:, :, :dim].astype(float)).astype(int)

    unique_states_per_trajectory = []
    for trajectory in discrete_traj:
        # Find unique states in the current trajectory
        unique_states = np.unique(trajectory, axis=0)
        # Convert the trajectory to a hashable type
        unique_states_per_trajectory.append(tuple(map(tuple, unique_states)))
    return unique_states_per_trajectory

def init_trajectory_dict(discrete_trajectory):

    trajectory_dict = {}
    for trajectory in discrete_trajectory:
        if trajectory in trajectory_dict:
            continue
        else:
            trajectory_dict[trajectory] = np.array([1,1])
    return trajectory_dict

def get_trajectory_dict_from_pair(traj_alpha_beta_dict, discrete_obs_1, discrete_obs_2, single_labels):
    for i in range(len(single_labels)):
        if single_labels[i] == 1:
            traj_alpha_beta_dict[discrete_obs_1[i]][0] += 1
            traj_alpha_beta_dict[discrete_obs_2[i]][1] += 1
        elif single_labels[i] == -1:
            traj_alpha_beta_dict[discrete_obs_1[i]][1] += 1
            traj_alpha_beta_dict[discrete_obs_2[i]][0] += 1
        # TODO how to treat not compaied ones?
        elif single_labels[i] == 0:
            traj_alpha_beta_dict[discrete_obs_1[i]][0] += 1
            traj_alpha_beta_dict[discrete_obs_1[i]][1] += 1
            traj_alpha_beta_dict[discrete_obs_2[i]][0] += 1
            traj_alpha_beta_dict[discrete_obs_2[i]][1] += 1
    return traj_alpha_beta_dict

def to_torch(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed

class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

class NormalComparisonModel(nn.Module):
    def __init__(self, input_dim, dense_units, dropout_rate, device):
        super(NormalComparisonModel, self).__init__()
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(4 * input_dim, dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.output = nn.Linear(dense_units // 2, 1)
        
        # Device setup
        self.device = device
        self.to(device)

    def forward(self, f1, f2):
        f1, f2 = f1.to(self.device), f2.to(self.device)
        
        # Shape of f1: (N1, L1, D), f2: (N2, L2, D)
        N1, L1, D = f1.shape
        N2, L2, _ = f2.shape

        # Expand dimensions for pairwise comparison
        f1_expanded = f1.unsqueeze(1).expand(N1, N2, L1, D)  # Shape: (N1, N2, L1, D)
        f2_expanded = f2.unsqueeze(0).expand(N1, N2, L2, D)  # Shape: (N1, N2, L2, D)
        
        # Flatten for processing through Transformer
        f1_flat = f1_expanded.reshape(-1, L1, D)  # Shape: (N1*N2, L1, D)
        f2_flat = f2_expanded.reshape(-1, L2, D)  # Shape: (N1*N2, L2, D)

        # Sequence pooling: Reduce sequence dimension
        f1_pooled = f1_flat.mean(dim=1)
        f2_pooled = f2_flat.mean(dim=1)

        # Pairwise comparison (concatenate, subtract)
        combined_features = torch.cat([
            f1_pooled,  # Reduced f1
            f2_pooled,  # Reduced f2
            f1_pooled - f2_pooled,  # Difference
        ], dim=-1)  # Shape: (N1*N2, 3*D)

        # Fully connected layers
        x = F.gelu(self.fc1(combined_features))  # Shape: (N1*N2, dense_units)
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))  # Shape: (N1*N2, dense_units // 2)
        x = self.output(x)  # Shape: (N1*N2, 1)
        
        # Sigmoid activation for [0, 1] output
        output = torch.sigmoid(x).squeeze(-1)  # Shape: (N1*N2)
        output = output.view(N1, N2)  # Reshape to (N1, N2)
        return output

class AttentionComparisonModel(nn.Module):
    def __init__(self, input_dim, dropout_rate, nhead, device):
        super(AttentionComparisonModel, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=nhead, batch_first=True)
        self.query_attn = nn.Parameter(torch.randn(1, input_dim))  # Learnable query for attention

        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(input_dim // 2, 1)
        self.device = device
        self.to(device)

    def attention_pool(self, x, query):
        # x: (batch_size, seq_len, input_dim)
        # query: (1, input_dim)
        attn_weights = torch.softmax(torch.matmul(x, query.T), dim=1)  # Compute attention weights
        weighted_sum = torch.sum(attn_weights * x, dim=1)  # Weighted sum along sequence dimension
        return weighted_sum

    def forward(self, f1, f2):
        f1, f2 = f1.to(self.device), f2.to(self.device)

        N1, L1, D = f1.shape
        N2, L2, _ = f2.shape

        f1_flat = f1.unsqueeze(1).expand(N1, N2, L1, D).reshape(-1, L1, D)
        f2_flat = f2.unsqueeze(0).expand(N1, N2, L2, D).reshape(-1, L2, D)

        attn_out, _ = self.attention(f1_flat, f2_flat, f2_flat)

        attn_reduced = self.attention_pool(attn_out, self.query_attn)

        x = F.gelu(self.fc1(attn_reduced))
        x = self.dropout(x)
        x = self.output(x)
        output = torch.sigmoid(x)
        output = output.view(N1, N2)
        return output

    def one_to_one_forward(self, f1, f2):
        f1, f2 = f1.to(self.device), f2.to(self.device)

        attn_out, _ = self.attention(f1, f2, f2)

        # Apply self-attention weighted compression
        attn_reduced = self.attention_pool(attn_out, self.query_attn)

        x = F.gelu(self.fc1(attn_reduced))
        x = self.dropout(x)
        x = self.output(x)
        output = torch.sigmoid(x)
        return output


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerEncModel(nn.Module):
    def __init__(self, data_dim, embedding_dim, nhead, num_encoder_layers, device):
        super(TransformerEncModel, self).__init__()
        self.device = device
        self.embedding = nn.Linear(data_dim, embedding_dim)
        # self.pos_encoder = self.create_positional_encoding(seq_length, embedding_dim)
        self.pos_emb = SinusoidalPosEmb(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

    def forward(self, x):
        traj_len = x.shape[1]
        x = self.embedding(x)  # Map input to embedding dimension
        pos = self.pos_emb(
            torch.arange(traj_len, device=self.device))[None,]
        x += pos  # Add positional encoding
        output = self.transformer_encoder(x)
        return output

class CausalTransformerBetaModel(nn.Module):
    def __init__(self, data_dim, embedding_dim, nhead, num_encoder_layers, output_dim, device):
        super(CausalTransformerBetaModel, self).__init__()
        self.device = device
        self.embedding = nn.Linear(data_dim, embedding_dim)
        # self.pos_encoder = self.create_positional_encoding(seq_length, embedding_dim)
        self.pos_emb = SinusoidalPosEmb(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(embedding_dim, output_dim)
        self.softplus = nn.Softplus()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, x):
        traj_len = x.shape[1]
        x = self.embedding(x)  # Map input to embedding dimension
        pos = self.pos_emb(
            torch.arange(traj_len, device=self.device))[None,]
        x += pos  # Add positional encoding
        mask = self.generate_square_subsequent_mask(traj_len).to(x.device)
        output = self.transformer_encoder(x, mask=mask)
        output = self.output_layer(output)
        # TODO mean or last one
        # output = output.mean(dim=1)  # Aggregate across the sequence dimension
        output = output[:, -1, :]  # Take the output from the last timestep
        output = self.softplus(output)
        return output

class BetaNetwork(nn.Module):
    def __init__(self, data, device=torch.device('cuda'), data_size = 500):
        super(BetaNetwork, self).__init__()

        act_data = np.concatenate((data['action'], data['action_2']), axis=0)
        obs_data = np.concatenate((data['obs'], data['obs_2']), axis=0)
        votes_data = np.concatenate((data['votes'], data['votes_2']), axis=0)

        if data_size <= obs_data.shape[0]:
            indices = np.random.randint(0, obs_data.shape[0], size=data_size)
            obs_data = obs_data[indices, ...]
            act_data = act_data[indices, ...]
            votes_data = votes_data[indices, ...]

        act_data = torch.from_numpy(act_data).float().to(device)
        obs_data = torch.from_numpy(obs_data).float().to(device)
        self.votes_data = torch.from_numpy(votes_data).to(device)
        self.lr = None
        self.device = device
        self.data = torch.concat((obs_data, act_data), dim=-1)

        class BetaModel(nn.Module):
            def __init__(self, obs_data, act_data, device=torch.device('cuda')):
                super(BetaModel, self).__init__()

                self.enc_model = TransformerEncModel(
                    data_dim = act_data.shape[-1] + obs_data.shape[-1],
                    embedding_dim = 256,
                    nhead = 4,
                    num_encoder_layers = 2,
                    device = device
                ).to(device)

                self.comp_model = AttentionComparisonModel(
                    input_dim = 256, 
                    dropout_rate = 0.3,
                    nhead = 16,
                    device = device
                ).to(device)

                self.data = torch.concat((obs_data, act_data), dim=-1)

            def forward(self, x):
                batch_f = self.enc_model(x)
                all_data_f = self.enc_model(self.data)
                bias = all_data_f.mean()
                std = all_data_f.std()
                all_data_f = (all_data_f - bias) / std
                batch_f = (batch_f - bias) / std
                output = self.comp_model(batch_f, all_data_f)
                return output
            
            def one_to_one_forward(self, x, y):
                x_f = self.enc_model(x)
                y_f = self.enc_model(y)
                all_data_f = self.enc_model(self.data)
                bias = all_data_f.mean()
                std = all_data_f.std()
                x_f = (x_f - bias) / std
                y_f = (y_f - bias) / std
                output = self.comp_model.one_to_one_forward(x_f, y_f)
                return output

       

        self.model = BetaModel(obs_data, act_data, device)
        self.opt = None #, weight_decay=1.0e-4
        self.scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=100, verbose=True)
        self.ref_model: BetaModel

    def get_alpha_beta(self, x):
        batch_comp = self.model(x).detach()
        alpha = torch.sum(batch_comp, dim=-1)
        beta = torch.sum(1 - batch_comp, dim=-1)
        # alpha = torch.sum(torch.where(batch_comp > 0.5, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device)), dim=-1)
        # beta = torch.sum(torch.where(batch_comp < 0.5, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device)), dim=-1)

        return alpha.detach(), beta.detach()

    def fit_data(self, save_dir=None, load_dir=None, num_epochs=1, warm_up_epochs=0, batch_size=1, lr=1.0e-5):
        if load_dir is None:
            interval = math.ceil(self.data.shape[0] / batch_size)
            total_steps = num_epochs * interval
            warm_up_steps = warm_up_epochs * interval
            main_steps = total_steps - warm_up_steps

            # Learning rate schedulers
            self.lr = lr
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
            warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
            cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
            self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

            logger = logging.getLogger()
            time.sleep(0.5)
            stage_time = datetime.now()
            logger.info(f'Beta Model Training Start: {stage_time} seconds')


            for epoch in range(num_epochs):
                beta_loss_all = []
                batch_shuffled_idx = np.random.permutation(self.data.shape[0])

                time.sleep(0.5)
                stage_time_last = datetime.now()
                logger.info(f'Epoch {epoch + 1} Start: {stage_time_last} seconds')


                for i in tqdm(range(interval)):

                    start_pt = i * batch_size
                    end_pt = min((i + 1) * batch_size, self.data.shape[0])
                    local_idx = batch_shuffled_idx[start_pt:end_pt]
                    batch = self.data[local_idx, ...]
                    batch_votes = self.votes_data[local_idx, ...]

                    comp = torch.sigmoid(batch_votes - self.votes_data.T)
                    pred_comp = self.model(batch)

                    beta_loss = torch.mean((comp - pred_comp) ** 2)

                    beta_loss_all.append(beta_loss)

                    self.opt.zero_grad()
                    beta_loss.backward()
                    self.opt.step()
                    self.scheduler.step()  # Update LR after each optimizer step

                beta_loss_all = torch.stack(beta_loss_all, dim=0)
                print("iteration:", epoch + 1)
                print("mean_beta_loss_all:", torch.mean(beta_loss_all).item())

                time.sleep(0.5)
                stage_time_last = datetime.now()
                logger.info(f'Epoch {epoch + 1} End: {stage_time_last} seconds')

                if save_dir is not None and (((epoch + 1) % 50 == 0) or ((epoch + 1) == num_epochs)):
                    tmp_save_dir = Path(save_dir) / f'itr_{epoch + 1}'
                    tmp_save_dir.mkdir(parents=True, exist_ok=True)
                    model_file = tmp_save_dir / 'beta_model.pth'
                    self.save_model(model_file)
            self.ref_model = copy.deepcopy(self.model)
        else:
            self.load_model(load_dir)
            self.ref_model = copy.deepcopy(self.model)

    def online_update(self, dataset, num_epochs=1, warm_up_epochs=0, batch_size=1, lr = 1.0e-6):
        interval = math.ceil(dataset["obs"].shape[0] / batch_size)
        total_steps = num_epochs * interval
        warm_up_steps = warm_up_epochs * interval
        main_steps = total_steps - warm_up_steps

        # Learning rate schedulers
        self.lr = lr
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
        cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
        self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

        logger = logging.getLogger()
        time.sleep(0.5)
        stage_time = datetime.now()
        logger.info(f'Beta Model Training Start: {stage_time} seconds')

        for epoch in range(num_epochs):
            beta_loss_all = []
            batch_shuffled_idx = np.random.permutation(dataset["obs"].shape[0])

            time.sleep(0.5)
            stage_time_last = datetime.now()
            logger.info(f'Epoch {epoch + 1} Start: {stage_time_last} seconds')

            for i in tqdm(range(interval)):

                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, dataset["obs"].shape[0])
                batch = index_batch(dataset, batch_shuffled_idx[start_pt:end_pt])

                obs_1 = batch['obs']  # batch_size * traj_len * obs_dim
                act_1 = batch['action']  # batch_size * traj_len * action_dim
                obs_2 = batch['obs_2']
                act_2 = batch['action_2']
                s_a_1 = np.concatenate([obs_1, act_1], axis=-1)
                s_a_2 = np.concatenate([obs_2, act_2], axis=-1)

                votes_1 = torch.from_numpy(batch['votes']).to(self.device)
                votes_2 = torch.from_numpy(batch['votes_2']).to(self.device)

                # threshold = 1e-3
                # diff = torch.abs(votes_1 - votes_2)
                # condition_1 = (votes_1 > votes_2) & (diff >= threshold)  # votes_1 > votes_2 and diff >= threshold
                # condition_2 = (votes_1 < votes_2) & (diff >= threshold)  # votes_1 < votes_2 and diff >= threshold

                # comp_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
                # comp_1 = torch.squeeze(comp_1, dim=-1)
                # comp_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
                # comp_2 = torch.squeeze(comp_2, dim=-1)

                comp_1 = torch.sigmoid(votes_1 - votes_2)
                comp_2 = torch.sigmoid(votes_2 - votes_1)

                pred_comp_1 = self.model.one_to_one_forward(torch.from_numpy(s_a_1).float().to(self.device), torch.from_numpy(s_a_2).float().to(self.device))
                pred_comp_2 = self.model.one_to_one_forward(torch.from_numpy(s_a_2).float().to(self.device), torch.from_numpy(s_a_1).float().to(self.device))
                # ref_comp_1 = self.ref_model.one_to_one_forward(torch.from_numpy(s_a_1).float().to(self.device), torch.from_numpy(s_a_2).float().to(self.device)).detach()
                # ref_comp_2 = self.ref_model.one_to_one_forward(torch.from_numpy(s_a_2).float().to(self.device), torch.from_numpy(s_a_1).float().to(self.device)).detach()

                beta_loss = (torch.mean((comp_1 - pred_comp_1) ** 2) + torch.mean((comp_2 - pred_comp_2) ** 2)) / 2

                beta_loss_all.append(beta_loss)

                self.opt.zero_grad()
                beta_loss.backward()
                self.opt.step()
                self.scheduler.step()  # Update LR after each optimizer step


            time.sleep(0.5)
            stage_time_now = datetime.now()
            logger.info(f'Epoch {epoch + 1} Spending time:{stage_time_now - stage_time_last} End: {stage_time_now} seconds')

            beta_loss_all = torch.stack(beta_loss_all, dim=0)
            print("iteration:", epoch + 1)
            print("mean_beta_loss_all:", torch.mean(beta_loss_all).item())

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))


class BetaImageNetwork(nn.Module):
    def __init__(self, data, policy: DiffusionTransformerHybridImagePolicy, device=torch.device('cuda'), data_size = 500, ):
        super(BetaNetwork, self).__init__()

        act_data = np.concatenate((data['action'], data['action_2']), axis=0)
        votes_data = np.concatenate((data['votes'], data['votes_2']), axis=0)
        obs_data = {key:np.concatenate((data['obs'][key], data['obs_2'][key]), axis=0) for key in data['obs']}
        if data_size <= act_data.shape[0]:
            indices = np.random.randint(0, act_data.shape[0], size=data_size)
            obs_data = {key:obs_data[key][indices, ...]for key in obs_data.keys()}
            act_data = act_data[indices, ...]
            votes_data = votes_data[indices, ...]

        act_data = torch.from_numpy(act_data).float().to(device)
        obs_data = {key: torch.from_numpy(obs_data[key]).float().to(device) for key in obs_data.keys()}

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        obs_data = policy.normalizer.normalize(obs_data)
        act_data = policy.normalizer.normalize(act_data)

        this_nobs = dict_apply(obs_data, 
                    lambda x: x.reshape(-1,*x.shape[2:]))
        nobs_features = obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(data_size, act_data.shape[1], -1)
        nobs_features.detach()


        self.votes_data = torch.from_numpy(votes_data).to(device)
        self.lr = None
        self.device = device
        self.data = torch.concat((act_data, nobs_features), dim=-1)

        class BetaModel(nn.Module):
            def __init__(self, obs_data, act_data, device=torch.device('cuda')):
                super(BetaModel, self).__init__()

                self.enc_model = TransformerEncModel(
                    data_dim = act_data.shape[-1] + obs_data.shape[-1],
                    embedding_dim = 256,
                    nhead = 4,
                    num_encoder_layers = 2,
                    device = device
                ).to(device)

                self.comp_model = AttentionComparisonModel(
                    input_dim = 256, 
                    dropout_rate = 0.3,
                    nhead = 16,
                    device = device
                ).to(device)

                self.data = torch.concat((obs_data, act_data), dim=-1)

            def forward(self, x):
                batch_f = self.enc_model(x)
                all_data_f = self.enc_model(self.data)
                bias = all_data_f.mean()
                std = all_data_f.std()
                all_data_f = (all_data_f - bias) / std
                batch_f = (batch_f - bias) / std
                output = self.comp_model(batch_f, all_data_f)
                return output
            
            def one_to_one_forward(self, x, y):
                x_f = self.enc_model(x)
                y_f = self.enc_model(y)
                all_data_f = self.enc_model(self.data)
                bias = all_data_f.mean()
                std = all_data_f.std()
                x_f = (x_f - bias) / std
                y_f = (y_f - bias) / std
                output = self.comp_model.one_to_one_forward(x_f, y_f)
                return output

       

        self.model = BetaModel(nobs_features, act_data, device)
        self.opt = None #, weight_decay=1.0e-4
        self.scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=100, verbose=True)
        self.ref_model: BetaModel

    def get_alpha_beta(self, x):
        batch_comp = self.model(x).detach()
        alpha = torch.sum(batch_comp, dim=-1)
        beta = torch.sum(1 - batch_comp, dim=-1)
        # alpha = torch.sum(torch.where(batch_comp > 0.5, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device)), dim=-1)
        # beta = torch.sum(torch.where(batch_comp < 0.5, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device)), dim=-1)

        return alpha.detach(), beta.detach()

    # def fit_data(self, save_dir=None, load_dir=None, num_epochs=1, warm_up_epochs=0, batch_size=1, lr=1.0e-5):
    #     if load_dir is None:
    #         interval = math.ceil(self.data.shape[0] / batch_size)
    #         total_steps = num_epochs * interval
    #         warm_up_steps = warm_up_epochs * interval
    #         main_steps = total_steps - warm_up_steps

    #         # Learning rate schedulers
    #         self.lr = lr
    #         self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
    #         warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
    #         cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
    #         self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

    #         for epoch in range(num_epochs):
    #             beta_loss_all = []

    #             batch_shuffled_idx = np.random.permutation(self.data.shape[0])
    #             for i in tqdm(range(interval)):

    #                 start_pt = i * batch_size
    #                 end_pt = min((i + 1) * batch_size, self.data.shape[0])
    #                 local_idx = batch_shuffled_idx[start_pt:end_pt]
    #                 batch = self.data[local_idx, ...]
    #                 batch_votes = self.votes_data[local_idx, ...]

    #                 comp = torch.sigmoid(batch_votes - self.votes_data.T)
    #                 pred_comp = self.model(batch)

    #                 beta_loss = torch.mean((comp - pred_comp) ** 2)

    #                 beta_loss_all.append(beta_loss)

    #                 self.opt.zero_grad()
    #                 beta_loss.backward()
    #                 self.opt.step()
    #                 self.scheduler.step()  # Update LR after each optimizer step

    #             beta_loss_all = torch.stack(beta_loss_all, dim=0)
    #             print("iteration:", epoch + 1)
    #             print("mean_beta_loss_all:", torch.mean(beta_loss_all).item())

    #             if save_dir is not None and (((epoch + 1) % 50 == 0) or ((epoch + 1) == num_epochs)):
    #                 tmp_save_dir = Path(save_dir) / f'itr_{epoch + 1}'
    #                 tmp_save_dir.mkdir(parents=True, exist_ok=True)
    #                 model_file = tmp_save_dir / 'beta_model.pth'
    #                 self.save_model(model_file)
    #         self.ref_model = copy.deepcopy(self.model)
    #     else:
    #         self.load_model(load_dir)
    #         self.ref_model = copy.deepcopy(self.model)

    def online_update(self, dataset, num_epochs=1, warm_up_epochs=0, batch_size=1, lr = 1.0e-6):
        interval = math.ceil(dataset["obs"].shape[0] / batch_size)
        total_steps = num_epochs * interval
        warm_up_steps = warm_up_epochs * interval
        main_steps = total_steps - warm_up_steps

        # Learning rate schedulers
        self.lr = lr
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
        cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
        self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])

        for epoch in range(num_epochs):
            beta_loss_all = []

            batch_shuffled_idx = np.random.permutation(dataset["obs"].shape[0])
            for i in tqdm(range(interval)):

                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, dataset["obs"].shape[0])
                batch = index_batch(dataset, batch_shuffled_idx[start_pt:end_pt])

                obs_1 = batch['obs']  # batch_size * traj_len * obs_dim
                act_1 = batch['action']  # batch_size * traj_len * action_dim
                obs_2 = batch['obs_2']
                act_2 = batch['action_2']
                s_a_1 = np.concatenate([obs_1, act_1], axis=-1)
                s_a_2 = np.concatenate([obs_2, act_2], axis=-1)

                votes_1 = torch.from_numpy(batch['votes']).to(self.device)
                votes_2 = torch.from_numpy(batch['votes_2']).to(self.device)

                # threshold = 1e-3
                # diff = torch.abs(votes_1 - votes_2)
                # condition_1 = (votes_1 > votes_2) & (diff >= threshold)  # votes_1 > votes_2 and diff >= threshold
                # condition_2 = (votes_1 < votes_2) & (diff >= threshold)  # votes_1 < votes_2 and diff >= threshold

                # comp_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
                # comp_1 = torch.squeeze(comp_1, dim=-1)
                # comp_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
                # comp_2 = torch.squeeze(comp_2, dim=-1)

                comp_1 = torch.sigmoid(votes_1 - votes_2)
                comp_2 = torch.sigmoid(votes_2 - votes_1)

                pred_comp_1 = self.model.one_to_one_forward(torch.from_numpy(s_a_1).float().to(self.device), torch.from_numpy(s_a_2).float().to(self.device))
                pred_comp_2 = self.model.one_to_one_forward(torch.from_numpy(s_a_2).float().to(self.device), torch.from_numpy(s_a_1).float().to(self.device))
                # ref_comp_1 = self.ref_model.one_to_one_forward(torch.from_numpy(s_a_1).float().to(self.device), torch.from_numpy(s_a_2).float().to(self.device)).detach()
                # ref_comp_2 = self.ref_model.one_to_one_forward(torch.from_numpy(s_a_2).float().to(self.device), torch.from_numpy(s_a_1).float().to(self.device)).detach()

                beta_loss = (torch.mean((comp_1 - pred_comp_1) ** 2) + torch.mean((comp_2 - pred_comp_2) ** 2)) / 2

                beta_loss_all.append(beta_loss)

                self.opt.zero_grad()
                beta_loss.backward()
                self.opt.step()
                self.scheduler.step()  # Update LR after each optimizer step

            beta_loss_all = torch.stack(beta_loss_all, dim=0)
            print("iteration:", epoch + 1)
            print("mean_beta_loss_all:", torch.mean(beta_loss_all).item())

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))