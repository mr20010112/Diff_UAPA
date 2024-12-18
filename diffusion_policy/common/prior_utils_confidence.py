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


class AttentionComparisonModel(nn.Module):
    def __init__(self, input_dim, dense_units, dropout_rate, nhead, device):
        super(AttentionComparisonModel, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=nhead, batch_first=True)
        self.fc1 = nn.Linear(3 * input_dim, dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.output = nn.Linear(dense_units // 2, 1)
        self.device = device
        self.to(device)

    def forward(self, f1, f2):
        f1, f2 = f1.to(self.device), f2.to(self.device)

        # # Normalize input features
        # f1 = F.normalize(f1, p=2, dim=-1)
        # f2 = F.normalize(f2, p=2, dim=-1)

        N1, L1, D = f1.shape
        N2, L2, _ = f2.shape

        f1_flat = f1.unsqueeze(1).expand(N1, N2, L1, D).reshape(-1, L1, D)
        f2_flat = f2.unsqueeze(0).expand(N1, N2, L2, D).reshape(-1, L2, D)

        attn_out, _ = self.attention(f1_flat, f2_flat, f2_flat)

        # Add residual connection
        attn_res = attn_out + f1_flat

        # Reduce sequence dimension
        attn_reduced = attn_out.sum(dim=1)
        f1_reduced = f1_flat.sum(dim=1)
        attn_res_reduced = attn_res.sum(dim=1)

        # Concatenate features
        combined = torch.cat([f1_reduced, attn_res_reduced, f1_reduced - attn_reduced], dim=-1)

        x = F.gelu(self.fc1(combined))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.output(x)
        output = torch.sigmoid(x)
        output = output.view(N1, N2)
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

class MLPBetaModel(nn.Module):
    def __init__(self, data_dim, device):
        super(MLPBetaModel, self).__init__()
        self.device = device
        self.network = nn.Sequential(*gen_net(in_size=data_dim,
                                            out_size=2, H=64, n_layers=3,
                                            activation=None)).float().to(self.device)
        # self.backbone_net = nn.Sequential(
        #     nn.Linear(data_dim, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(),
        # ).to(self.device)
        #
        # self.alpha_net = nn.Sequential(
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 1),  # Assuming alpha is a single value
        # ).to(self.device)
        #
        # self.beta_net = nn.Sequential(
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 1),  # Assuming alpha is a single value
        # ).to(self.device)
        self.activation = nn.LeakyReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        output = self.network(x)
        output = output.mean(dim=1)  # Aggregate across the sequence dimension
        output = self.softplus(output)

        # output = self.backbone_net(x)
        # alpha = self.softplus(self.alpha_net(output).mean(dim=1)).squeeze(-1)
        # beta = self.softplus(self.beta_net(output).mean(dim=1)).squeeze(-1)
        # return alpha, beta
        return output

class BetaNetwork(nn.Module):
    def __init__(self, data, lr=1.0e-4, device=torch.device('cuda'), data_size = 500):
        super(BetaNetwork, self).__init__()
        self.obs_dim = data['obs'].shape[-1]
        self.act_dim = data['action'].shape[-1]

        act_data = np.concatenate((data['action'], data['action_2']), axis=0)
        obs_data = np.concatenate((data['obs'], data['obs_2']), axis=0)
        votes_data = np.concatenate((data['votes'], data['votes_2']), axis=0)

        if data_size <= obs_data.shape[0]:
            indices = np.random.permutation(data_size)
            obs_data = obs_data[indices, ...]
            act_data = act_data[indices, ...]
            votes_data = votes_data[indices, ...]

        act_data = torch.from_numpy(act_data).float().to(device)
        obs_data = torch.from_numpy(obs_data).float().to(device)
        self.votes_data = torch.from_numpy(votes_data).to(device)
        self.lr = lr
        self.device = device

        class BetaModel(nn.Module):
            def __init__(self, act_data, obs_data, device=torch.device('cuda')):
                super(BetaModel, self).__init__()

                self.enc_model = TransformerEncModel(
                    data_dim = act_data.shape[-1] + obs_data.shape[-1],
                    embedding_dim = 256,
                    nhead = 4,
                    num_encoder_layers = 1,
                    device = device
                ).to(device)

                self.comp_model = AttentionComparisonModel(
                    input_dim = 256, 
                    dense_units = 256, 
                    dropout_rate = 0.3,
                    nhead = 4,
                    device = device
                ).to(device)

                self.data = torch.concat((act_data, obs_data), dim=-1)

            def forward(self, x):
                batch_f = self.enc_model(x)
                all_data_f = self.enc_model(self.data)
                bias = all_data_f.mean()
                std = all_data_f.std()
                all_data_f = (all_data_f - bias) / std
                batch_f = (batch_f - bias) / std
                output = self.comp_model(batch_f, all_data_f)
                return output

        self.model = BetaModel(act_data, obs_data, device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr) #, weight_decay=1.0e-4
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=100, verbose=True) #patience=100, verbose=True

    def get_alpha_beta(self, x):
        batch_comp = self.model(x).detach()
        alpha = torch.sum(2 * torch.clamp(batch_comp - 0.5, min=0), dim=-1)
        beta = torch.sum(2 * torch.clamp(0.5 - batch_comp, min=0), dim=-1)
        # alpha, beta = self.model(x)
        return alpha.detach(), beta.detach()

    def fit_data(self, dataset, save_dir=None, load_dir=None, num_epochs=1, batch_size=1):
        if load_dir is None:
            interval = math.ceil(dataset["obs"].shape[0] / batch_size)
            for epoch in range(num_epochs):

                beta_loss_all = []

                batch_shuffled_idx = np.random.permutation(dataset["obs"].shape[0])
                for i in tqdm(range(interval)):

                    start_pt = i * batch_size
                    end_pt = min((i + 1) * batch_size, dataset["obs"].shape[0])
                    batch = index_batch(dataset, batch_shuffled_idx[start_pt:end_pt])

                    # get batch
                    obs_1 = batch['obs']  # batch_size * traj_len * obs_dim
                    act_1 = batch['action']  # batch_size * traj_len * action_dim
                    obs_2 = batch['obs_2']
                    act_2 = batch['action_2']
                    votes_1 = batch['votes']
                    votes_2 = batch['votes_2']
                    votes_1 = torch.from_numpy(votes_1).to(self.device)  # Shape: (N1, feature_dim)
                    votes_2 = torch.from_numpy(votes_2).to(self.device)  # Shape: (N2, feature_dim)
                    s_a_1 = np.concatenate([obs_1, act_1], axis=-1)
                    s_a_2 = np.concatenate([obs_2, act_2], axis=-1)

                    comp_1 = torch.sigmoid(votes_1 - self.votes_data.T)  # Shape: (N_1, N_2)
                    comp_2 = torch.sigmoid(votes_2 - self.votes_data.T)  # Shape: (N_1, N_2)

                    # comp_1 = comp_1.squeeze(-1)  # Shape: (N1, M)
                    # comp_2 = comp_2.squeeze(-1)  # Shape: (N2, M)

                    pred_comp_1 = self.model(torch.from_numpy(s_a_1).float().to(self.device))
                    pred_comp_2 = self.model(torch.from_numpy(s_a_2).float().to(self.device))

                    beta_loss = torch.sum((pred_comp_1 - comp_1) ** 2) + torch.sum((pred_comp_2 - comp_2) ** 2)

                    beta_loss_all.append(beta_loss)

                    beta_loss.backward()
                    self.opt.zero_grad()
                    self.opt.step()

                # Scheduler step
                avg_loss = torch.stack(beta_loss_all).mean().item()
                self.scheduler.step(avg_loss)

                beta_loss_all = torch.stack(beta_loss_all, dim=0)
                print("iteration:", epoch + 1)
                print("mean_beta_loss_all:", torch.mean(beta_loss_all).item())

                if save_dir is not None and ((epoch+1) % 50 or (epoch+1) == num_epochs) == 0:
                    tmp_save_dir= Path(save_dir) / f'itr_{epoch+1}'
                    tmp_save_dir.mkdir(parents=True, exist_ok=True)
                    model_file = tmp_save_dir / 'beta_model.pth'
                    self.save_model(model_file)
        else:
            self.load_model(load_dir)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))


class PrefTransformer1(nn.Module):
    ''' Transformer Structure used in Preference Transformer.
    
    Description:
        This structure holds a causal transformer, which takes in a sequence of observations and actions, 
        and outputs a sequence of latent vectors. Then, pass the latent vectors through self-attention to
        get a weight vector, which is used to weight the latent vectors to get the final preference score.
    
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - max_seq_len: maximum length of sequence
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        max_seq_len: int = 100,
        d_model: int = 256, nhead: int = 4, num_layers: int = 1, 
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.causual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers
        )
        self.mask = nn.Transformer.generate_square_subsequent_mask(2*self.max_seq_len)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.r_proj = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        if self.mask.device != obs.device: self.mask = self.mask.to(obs.device)
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.causual_transformer(x, self.mask[:2*traj_len,:2*traj_len])[:, 1::2]
        # x: (batch_size, traj_len, d_model)

        q = self.q_proj(x) # (batch_size, traj_len, d_model)
        k = self.k_proj(x) # (batch_size, traj_len, d_model)
        r = self.r_proj(x) # (batch_size, traj_len, 1)
        
        w = torch.softmax(q@k.permute(0, 2, 1)/np.sqrt(self.d_model), -1).mean(-2)
        # w: (batch_size, traj_len)
        
        z = (w * r.squeeze(-1)) # (batch_size, traj_len)
        
        return torch.tanh(z)


class PrefTransformer2(nn.Module):
    ''' Preference Transformer with no causal mask and no self-attention but one transformer layer to get the weight vector.
    
    Description:
        This structure has no causal mask and no self-attention.
        Instead, it uses one transformer layer to get the weight vector.
        
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        d_model: int, nhead: int, num_layers: int, 
    ):
        super().__init__()
        while num_layers < 2: num_layers += 1
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers - 1
        )
        self.value_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))
        self.weight_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act
        
        x = self.transformer(x)[:, 1::2]
        v = self.value_layer(x)
        w = torch.softmax(self.weight_layer(x), 1)
        return (w*v).squeeze(-1)
    

class PrefTransformer3(nn.Module):
    ''' Preference Transformer with no causal mask and no weight vector.
    
    Description:
        This structure has no causal mask and even no weight vector.
        Instead, it directly outputs the preference score.
        
    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''
    def __init__(self,
        observation_dim: int, action_dim: int, 
        d_model: int, nhead: int, num_layers: int, 
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 
            num_layers
        )
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]
        
        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act
        
        x = self.transformer(x)[:, 1::2]
        return self.output_layer(x).squeeze(-1)


class MLPDiffusion(nn.Module):
    def __init__(
        self,
        obs_dim,
        acs_dim,
        num_step = 200,
        beta_start = 1e-4,
        beta_end = 1e-1,
        traj_len = 200,
        data_dim = 1,
        device="cuda",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.num_step = num_step
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.traj_len = traj_len
        self.data_dim = data_dim
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(obs_dim + acs_dim + data_dim + 1, 256),  # input dim: obs_dim + acs_dim + t_dim(=1) + x=data_dim
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, data_dim),
        ).to(self.device)

        self.init_paras()

    def init_paras(self):
        self.beta = torch.linspace(
            start=self.beta_start, end=self.beta_end, steps=self.num_step
        ).view(-1, 1).repeat(1, self.traj_len).to(self.device)
        self.sigma = torch.sqrt(self.beta)
        self.alpha = 1 - self.beta
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha = 1 - self.alpha
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.multiplier2 = self.one_minus_alpha / self.sqrt_one_minus_alpha_bar
        self.multiplier1 = 1 / self.sqrt_alpha

    def forward(self, obs, action, x, t):  # eps_theta
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        if not isinstance(obs, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float)
        obs = obs.to(self.device)
        action = action.to(self.device)
        t = t.unsqueeze(-1)
        input = torch.cat([obs, action, x, t / self.num_step], dim=2)
        ret = self.network(input)
        return ret

    def reverse_sample(self, obs, action, x_t, t):  # from intermediate noise to data
        mul2_t = self.multiplier2.gather(0, t).unsqueeze(-1)
        mul1_t = self.multiplier1.gather(0, t).unsqueeze(-1)

        eps_theta = self.forward(obs, action, x_t, t)
        mean = mul1_t * (x_t - mul2_t * eps_theta)

        sigma_z = torch.gather(self.sigma, 0, t).unsqueeze(-1) * torch.randn_like(
            x_t, device=self.device
        )

        return mean + sigma_z

    def sample(self, obs, action, during_training=False):  # from pure noise to data (true forward)
        if during_training is False:
            if not isinstance(obs, torch.Tensor):
                if isinstance(obs, list):
                    obs = np.array(obs)
                obs = torch.tensor(obs, dtype=torch.float)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.int64)
            obs = obs.to(self.device)
            action = action.to(self.device)

        batch_size = obs.shape[0]
        x = torch.randn([batch_size, self.traj_len, self.data_dim], device=self.device)
        for t in reversed(range(self.num_step)):
            x = self.reverse_sample(
                obs,
                action,
                x,
                torch.tensor(t).repeat(batch_size, self.traj_len).to(self.device),
            ).detach()
        return torch.tanh(x)

    def compute_loss(self, obs, action, x_0):  # x_0 is true data, x_t / x is noise
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.long)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_step, size=[batch_size], device=self.device).view(-1, 1)
        t = t.repeat(1, self.traj_len)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar.gather(0, t).unsqueeze(-1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar.gather(0, t).unsqueeze(-1)
        eps = torch.randn_like(x_0, device=self.device)
        eps_theta = self.forward(
            obs,
            action,
            sqrt_alpha_bar_t * x_0 + eps * sqrt_one_minus_alpha_bar_t,
            t,
        )
        return torch.square(eps - eps_theta).mean()


    # def fit(self, dataset):
    #     loss_list = []
    #     opt = torch.optim.Adam(self.network.parameters(), lr=self.lr)
    #
    #     tmp = []
    #
    #     for _ in tqdm(range(self.max_iter)):
    #
    #         obs, action, reward, next_obs, next_action, done, next_val = dataset.sample(
    #             self.batch_size, self.h_to_learn
    #         )
    #
    #         obs = torch.tensor(obs, dtype=torch.float).to(self.device)
    #         action = torch.tensor(action, dtype=torch.long).to(self.device)
    #         reward = torch.tensor(reward, dtype=torch.float).to(self.device)
    #         next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
    #         next_action = torch.tensor(next_action, dtype=torch.long).to(self.device)
    #         done = torch.tensor(done, dtype=torch.float).to(self.device)
    #         next_val = torch.tensor(next_val, dtype=torch.float).to(self.device)
    #
    #         ret = reward + (1 - done) * next_val
    #         ret = ret.detach().float()
    #
    #         loss = self.compute_loss(
    #             obs,
    #             action,
    #             x_0=ret,
    #         )
    #         loss_list.append(loss.item())
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #
    #     return self, loss_list, tmp

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class DistributionalPrefTransformer(nn.Module):

    def __init__(self,
                 observation_dim: int, action_dim: int,
                 max_seq_len: int = 100,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 1,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)

        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.causual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True),
            num_layers
        )
        self.mask = nn.Transformer.generate_square_subsequent_mask(2 * self.max_seq_len)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.mean_proj = nn.Linear(d_model, 1)
        self.log_std_proj = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        if self.mask.device != obs.device: self.mask = self.mask.to(obs.device)
        batch_size, traj_len = obs.shape[:2]

        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos

        x = torch.empty((batch_size, 2 * traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.causual_transformer(x, self.mask[:2 * traj_len, :2 * traj_len])[:, 1::2]
        # x: (batch_size, traj_len, d_model)

        q = self.q_proj(x)  # (batch_size, traj_len, d_model)
        k = self.k_proj(x)  # (batch_size, traj_len, d_model)
        mean = self.mean_proj(x)  # (batch_size, traj_len, 1)
        log_std = self.log_std_proj(x)  # (batch_size, traj_len, 1)
        # log_std = torch.tanh(log_std)
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        r = normal.rsample()  # (batch_size, traj_len, 1)

        w = torch.softmax(q @ k.permute(0, 2, 1) / np.sqrt(self.d_model), -1).mean(-2)
        # w: (batch_size, traj_len)

        z = (w * r.squeeze(-1))  # (batch_size, traj_len)

        return torch.tanh(z)