import collections
import numpy as np
# import gym
import torch.utils
from tqdm import trange
import torch
import torch.nn as nn
import math
# import d4rl
import cv2
import concurrent.futures
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from diffusion_policy.model.vision.realrobot_image_obs_encoder import RealRobotImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer

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


def dirichlet_kl_divergence_loss(alpha, prior):
    """
    KL divergence between two Dirichlet distributions.
    
    Parameters:
        alpha (torch.Tensor): Dirichlet parameters for the first distribution (shape: [batch_size, n]).
        prior (torch.Tensor): Dirichlet parameters for the second distribution (shape: [batch_size, n]).
    
    Returns:
        torch.Tensor: KL divergence for each batch (shape: [batch_size]).
    """
    # Add small epsilon to avoid numerical instability
    epsilon = 1e-8
    alpha = alpha + epsilon
    prior = prior + epsilon

    # KL divergence terms
    analytical_kld = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(prior, dim=1))
    analytical_kld += torch.sum(torch.lgamma(prior), dim=1)
    analytical_kld -= torch.sum(torch.lgamma(alpha), dim=1)

    # Difference term
    minus_term = alpha - prior

    # Digamma term
    digamma_term = torch.digamma(alpha) - torch.digamma(torch.sum(alpha, dim=1)).unsqueeze(-1)

    # Final KL term
    analytical_kld += torch.sum(minus_term * digamma_term, dim=1)

    return analytical_kld


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

class TransformerBetaModel(nn.Module):
    def __init__(self, data_dim, embedding_dim, nhead, num_encoder_layers, output_dim, device):
        super(TransformerBetaModel, self).__init__()
        self.device = device
        self.embedding = nn.Linear(data_dim, embedding_dim)
        # self.pos_encoder = self.create_positional_encoding(seq_length, embedding_dim)
        self.pos_emb = SinusoidalPosEmb(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.sigmoid = nn.Sigmoid()
        self.encode_layer = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(embedding_dim, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        traj_len = x.shape[1]
        x = self.embedding(x)  # Map input to embedding dimension
        pos = self.pos_emb(
            torch.arange(traj_len, device=self.device))[None,]
        x += pos  # Add positional encoding
        output = self.transformer_encoder(x)
        output = self.sigmoid(output)
        output = self.encode_layer(output)
        output = self.relu(output)
        output = self.output_layer(output)
        # TODO mean or last one
        output = output.mean(dim=1)  # Aggregate across the sequence dimension
        # output = output[:, -1, :]  # Take the output from the last timestep
        output = self.softplus(output)
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
    def __init__(self,observation_dim, action_dim, obs_encoder: RealRobotImageObsEncoder=None, lr=3e-4, device=torch.device('cuda'),
                 model_type='Transformer', beta_coef=0.1):
        super(BetaNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lr = lr
        self.beta_coef = beta_coef
        self.device = device
        self.obs_encoder = obs_encoder

        if 'Causal' in model_type:
            self.model = CausalTransformerBetaModel(
                data_dim=self.observation_dim + self.action_dim,
                embedding_dim=256,
                nhead=4,
                num_encoder_layers=1,
                output_dim=2,
                device=self.device,
            ).to(self.device)
        elif 'Transformer' in model_type:
            self.model = TransformerBetaModel(
                data_dim = self.observation_dim + self.action_dim,
                embedding_dim = 256,
                nhead = 4,
                num_encoder_layers = 1,
                output_dim = 2,
                device = self.device,
            ).to(self.device)
        else:
            self.model = MLPBetaModel(
                data_dim = self.observation_dim + self.action_dim,
                device = self.device,
            )

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=10)

    def get_alpha_beta(self, x):
        alpha_beta = self.model(x).detach()
        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        # alpha, beta = self.model(x)
        return alpha.detach(), beta.detach()

    def kl_regularizer_loss(self, batch_size, alpha, beta):
        prior = torch.tensor(np.asarray(batch_size * [[1, 1]]), dtype=torch.float32).to(self.device)
        analytical_kld_loss = dirichlet_kl_divergence_loss(
            alpha=torch.stack([alpha, beta], dim=1),
            prior=prior).mean()
        return analytical_kld_loss


    def fit_data(self, dataset, save_dir=None, load_dir=None, num_epochs=1, batch_size=1):
        if load_dir is None:
            interval = math.ceil(dataset["obs"].shape[0] / batch_size)
            for epoch in range(num_epochs):

                beta_loss_1 = []
                beta_loss_2 = []
                beta_loss_3 = []
                beta_loss_4 = []
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
                    s_a_1 = np.concatenate([obs_1, act_1], axis=-1)
                    s_a_2 = np.concatenate([obs_2, act_2], axis=-1)

                    conditions_1 = [
                        np.all(batch['votes'] == 1, axis=1),
                        np.all(batch['votes'] == 0, axis=1),
                        np.all(batch['votes'] == 0.5, axis=1)
                        ]
                    values = [1, 0, 0]
                    single_labels_1 = torch.from_numpy(np.select(conditions_1, values)).float().to(self.device)

                    conditions_2 = [
                        np.all(batch['votes_2'] == 1, axis=1),
                        np.all(batch['votes_2'] == 0, axis=1),
                        np.all(batch['votes_2'] == 0.5, axis=1)
                        ]
                    values = [1, 0, 0]
                    single_labels_2 = torch.from_numpy(np.select(conditions_2, values)).float().to(self.device)

                    pred_1_alpha_beta = self.model(torch.from_numpy(s_a_1).float().to(self.device)) # batch_size * 2
                    pred_1_alpha = pred_1_alpha_beta[:, 0]
                    pred_1_beta = pred_1_alpha_beta[:, 1]


                    pred_2_alpha_beta = self.model(torch.from_numpy(s_a_2).float().to(self.device))
                    pred_2_alpha = pred_2_alpha_beta[:, 0]
                    pred_2_beta = pred_2_alpha_beta[:, 1]
                    # if equal, then discard
                    # TODO maybe if equal, then both towards 1 (both preferred)

                    loss_1 = torch.mean((torch.log(pred_1_alpha) + torch.log(pred_2_beta))* single_labels_1 \
                            + (torch.log(pred_2_alpha) + torch.log(pred_1_beta)) * single_labels_2)

                    # var_1 = (pred_1_alpha * pred_1_beta) / ((pred_1_alpha + pred_1_beta) ** 2 * (pred_1_alpha + pred_1_beta + 1))
                    
                    # var_2 = (pred_2_alpha * pred_2_beta) / ((pred_2_alpha + pred_2_beta) ** 2 * (pred_2_alpha + pred_2_beta + 1))

                    # loss_2 = (torch.clamp(torch.mean(torch.sqrt(var_1)) - torch.sqrt(torch.tensor(1 / 324, dtype=torch.float32, device=self.device)), min=0)) ** 2 \
                    #         + (torch.clamp(torch.mean(torch.sqrt(var_2)) - torch.sqrt(torch.tensor(1 / 324, dtype=torch.float32, device=self.device)), min=0)) ** 2
                    loss_2 = torch.mean(torch.clamp(pred_1_alpha - 25, min=0) ** 2) + torch.mean(torch.clamp(pred_2_alpha - 25, min=0) ** 2) \
                            + torch.mean(torch.clamp(pred_1_beta - 25, min=0) ** 2) + torch.mean(torch.clamp(pred_2_beta - 25, min=0) ** 2)
                    
                    # loss_3 = torch.log(4 / pred_1_alpha.std()) + (pred_1_alpha.var() + (pred_1_alpha.mean() - 12.5) ** 2) / (2 * 16) -0.5 \
                    #         + torch.log(4 / pred_2_alpha.std()) + (pred_2_alpha.var() + (pred_2_alpha.mean() - 12.5) ** 2) / (2 * 16) -0.5 \
                    #         + torch.log(4 / pred_1_beta.std()) + (pred_1_beta.var() + (pred_1_beta.mean() - 12.5) ** 2) / (2 * 16) -0.5 \
                    #         + torch.log(4 / pred_2_beta.std()) + (pred_2_beta.var() + (pred_2_beta.mean() - 12.5) ** 2) / (2 * 16) -0.5

                    controls_alpha_1 = torch.distributions.Normal(torch.mean(single_labels_1) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_1)).rsample((pred_1_alpha.shape[0],))
                    controls_alpha_1 = torch.sort(controls_alpha_1, dim=0)[0].to(self.device)

                    controls_alpha_2 = torch.distributions.Normal(torch.mean(single_labels_2) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_2)).rsample((pred_2_alpha.shape[0],))
                    controls_alpha_2 = torch.sort(controls_alpha_2, dim=0)[0].to(self.device)

                    controls_beta_1 = torch.distributions.Normal(torch.mean(single_labels_2) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_2)).rsample((pred_2_alpha.shape[0],))
                    controls_beta_1 = torch.sort(controls_beta_1, dim=0)[0].to(self.device)

                    controls_beta_2 = torch.distributions.Normal(torch.mean(single_labels_1) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_1)).rsample((pred_1_alpha.shape[0],))
                    controls_beta_2 = torch.sort(controls_beta_2, dim=0)[0].to(self.device)

                    loss_3 = torch.mean((torch.sort(pred_1_alpha, dim=0)[0] - controls_alpha_1) ** 2) \
                            + torch.mean((torch.sort(pred_2_alpha, dim=0)[0] - controls_alpha_2) ** 2) \
                            + torch.mean((torch.sort(pred_1_beta, dim=0)[0] - controls_beta_1) ** 2) \
                            + torch.mean((torch.sort(pred_2_beta, dim=0)[0] - controls_beta_2) ** 2)

                    loss_4 = self.kl_regularizer_loss(pred_1_alpha.shape[0], alpha=pred_1_alpha, beta=pred_1_beta) \
                            + self.kl_regularizer_loss(pred_2_alpha.shape[0], alpha=pred_2_alpha, beta=pred_2_beta)

                    beta_loss = -loss_1 + loss_2 + loss_3 + self.beta_coef*loss_4

                    beta_loss_1.append(loss_1)
                    beta_loss_2.append(loss_2)
                    beta_loss_3.append(loss_3)
                    beta_loss_4.append(loss_4)

                    beta_loss_all.append(beta_loss)

                    self.opt.zero_grad()
                    beta_loss.backward()
                    self.opt.step()

                # Scheduler step
                avg_loss = torch.stack(beta_loss_all).mean().item()
                self.scheduler.step(avg_loss)

                beta_loss_1 = torch.stack(beta_loss_1, dim=0)
                beta_loss_2 = torch.stack(beta_loss_2, dim=0)
                beta_loss_3 = torch.stack(beta_loss_3, dim=0)
                beta_loss_4 = torch.stack(beta_loss_4, dim=0)
                beta_loss_all = torch.stack(beta_loss_all, dim=0)
                print("iteration:", epoch + 1)
                print("mean_beta_loss_data:", torch.mean(beta_loss_1).item())
                print("mean_beta_loss_control:", torch.mean(beta_loss_2).item())
                print("mean_beta_loss_control_2:", torch.mean(beta_loss_3).item())
                print("mean_beta_loss_kl:", torch.mean(beta_loss_4).item())
                print("mean_beta_loss_all:", torch.mean(beta_loss_all).item())

                if save_dir is not None and (epoch+1) % 200 == 0:
                    tmp_save_dir= Path(save_dir) / f'itr_{epoch+1}'
                    tmp_save_dir.mkdir(parents=True, exist_ok=True)
                    model_file = tmp_save_dir / 'beta_model.pth'
                    self.save_model(model_file)
        else:
            self.load_model(load_dir)

    def fit_data_discrete(self, dataset):

        obs_1 = dataset['observations']  # batch_size * traj_len * obs_dim
        obs_2 = dataset['observations_2']

        conditions = [np.all(dataset['labels'] == [1, 0], axis=1),
                      np.all(dataset['labels'] == [0, 1], axis=1),
                      np.all(dataset['labels'] == [0.5, 0.5], axis=1)]
        values = [1, -1, 0]
        single_labels = torch.from_numpy(np.select(conditions, values)).float().to(self.device)

        discrete_obs_1 = get_discrete_traj(obs_1, dim=2)
        discrete_obs_2 = get_discrete_traj(obs_2, dim=2)

        traj_alpha_beta_dict = init_trajectory_dict(discrete_obs_1 + discrete_obs_2)
        self.traj_alpha_beta_dict = get_trajectory_dict_from_pair(traj_alpha_beta_dict, discrete_obs_1, discrete_obs_2,
                                                                  single_labels)

    def get_alpha_beta_discrete(self, x, dim=2):
        traj = x.cpu().numpy()
        discrete_obs = get_discrete_traj(traj, dim=dim)[0]
        alpha = self.traj_alpha_beta_dict[discrete_obs][0]
        beta = self.traj_alpha_beta_dict[discrete_obs][1]
        return alpha, beta

    def get_normalized_alpha_beta_discrete(self, x, dim=2):
        traj = x.cpu().numpy()
        discrete_obs = get_discrete_traj(traj, dim=dim)[0]
        alpha = self.traj_alpha_beta_dict[discrete_obs][0]
        beta = self.traj_alpha_beta_dict[discrete_obs][1]
        # Normalize alpha and beta
        alpha_normalized = alpha / 10
        beta_normalized = beta / 10
        return alpha_normalized, beta_normalized

    def get_rescaled_alpha_beta_discrete(self, x, dim=2, rescale=10):
        traj = x.cpu().numpy()
        discrete_obs = get_discrete_traj(traj, dim=dim)[0]
        alpha = self.traj_alpha_beta_dict[discrete_obs][0]
        beta = self.traj_alpha_beta_dict[discrete_obs][1]
        # Normalize alpha and beta
        alpha_normalized = alpha / rescale
        beta_normalized = beta / rescale
        return alpha_normalized, beta_normalized

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=self.device))

class RealRobotBetaNetwork(nn.Module):
    def __init__(self,observation_dim, action_dim, obs_encoder: RealRobotImageObsEncoder, normalizer: LinearNormalizer,
                 lr=3e-4, device=torch.device('cuda'), model_type='Transformer', beta_coef=0.1):
        super(RealRobotBetaNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lr = lr
        self.beta_coef = beta_coef
        self.device = device
        self.obs_encoder = obs_encoder
        self.normalizer = normalizer

        if 'Causal' in model_type:
            self.model = CausalTransformerBetaModel(
                data_dim=self.observation_dim + self.action_dim,
                embedding_dim=256,
                nhead=4,
                num_encoder_layers=1,
                output_dim=2,
                device=self.device,
            ).to(self.device)
        elif 'Transformer' in model_type:
            self.model = TransformerBetaModel(
                data_dim = self.observation_dim + self.action_dim,
                embedding_dim = 256,
                nhead = 4,
                num_encoder_layers = 1,
                output_dim = 2,
                device = self.device,
            ).to(self.device)
        else:
            self.model = MLPBetaModel(
                data_dim = self.observation_dim + self.action_dim,
                device = self.device,
            )

        params = list(self.model.parameters()) + list(self.obs_encoder.parameters())
        self.opt = torch.optim.Adam(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', patience=10)

    def get_alpha_beta(self, x):
        alpha_beta = self.model(x).detach()
        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        # alpha, beta = self.model(x)
        return alpha.detach(), beta.detach()

    def kl_regularizer_loss(self, batch_size, alpha, beta):
        prior = torch.tensor(np.asarray(batch_size * [[1, 1]]), dtype=torch.float32).to(self.device)
        analytical_kld_loss = dirichlet_kl_divergence_loss(
            alpha=torch.stack([alpha, beta], dim=1),
            prior=prior).mean()
        return analytical_kld_loss


    def fit_data(self, dataset, save_dir=None, load_dir=None, num_epochs=1, warm_up_epochs=0, batch_size=1, lr=1.0e-5):
        def decode_image(data):
            return cv2.imdecode(data, 1)
        
        if load_dir is None:
            interval = math.ceil(dataset["action"].shape[0] / batch_size)
            total_steps = num_epochs * interval
            warm_up_steps = warm_up_epochs * interval
            main_steps = total_steps - warm_up_steps

            # Learning rate schedulers
            self.lr = lr
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
            warm_up_scheduler = LinearLR(self.opt, start_factor=1e-8, end_factor=1.0, total_iters=warm_up_steps)
            cosine_scheduler = CosineAnnealingLR(self.opt, T_max=main_steps)
            self.scheduler = SequentialLR(self.opt, schedulers=[warm_up_scheduler, cosine_scheduler], milestones=[warm_up_steps])


            sequence_length = dataset["action"].shape[1]
            camera_keys = dataset["obs"]["images"].keys()
            qpos_keys = [key for key in dataset["obs"].keys() if key != 'images']

            for epoch in range(num_epochs):

                beta_loss_1 = []
                beta_loss_2 = []
                beta_loss_3 = []
                beta_loss_4 = []
                beta_loss_all = []

                batch_shuffled_idx = np.random.permutation(dataset["action"].shape[0])
                for i in tqdm(range(interval)):

                    start_pt = i * batch_size
                    end_pt = min((i + 1) * batch_size, dataset["action"].shape[0])
                    indices = batch_shuffled_idx[start_pt:end_pt]
                    batch = {}
                    batch["action"] = dataset["action"][indices]
                    batch["action_2"] = dataset["action_2"][indices]
                    batch["votes"] = dataset["votes"][indices]
                    batch["votes_2"] = dataset["votes_2"][indices]
                    batch['obs'] = {}
                    batch['obs_2'] = {}
                    batch['obs']['images'] = {}
                    batch['obs_2']['images'] = {}
                    compress_len = dataset["compress_len"][indices].squeeze(-1)
                    compress_len_2 = dataset["compress_len_2"][indices].squeeze(-1)

                    for key in qpos_keys:
                        batch['obs'][key] = dataset["obs"][key][indices]
                        batch['obs_2'][key] = dataset["obs_2"][key][indices]

                    for key in camera_keys:
                        image_data_1 = dataset["obs"]["images"][key][indices]
                        image_data_2 = dataset["obs_2"]["images"][key][indices]
                        decompressed_images = []
                        decompressed_images_2 = []
                        for k in range(len(image_data_1)):
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                results = executor.map(decode_image, \
                                        image_data_1[k, :compress_len[k, 0]])
                                results_2 = executor.map(decode_image, \
                                        image_data_2[k, :compress_len_2[k, 0]])
                                decompressed_image = list(results)
                                decompressed_image_2 = list(results_2)
                                decompressed_images.append(np.array(decompressed_image))
                                decompressed_images_2.append(np.array(decompressed_image_2))
                        
                        decompressed_images = np.array(decompressed_images)
                        decompressed_images = np.einsum('k h w c -> k c h w', decompressed_images)
                        decompressed_images = decompressed_images / 255.0

                        decompressed_images_2 = np.array(decompressed_images_2)
                        decompressed_images_2 = np.einsum('k h w c -> k c h w', decompressed_images_2)
                        decompressed_images_2 = decompressed_images_2 / 255.0

                        batch['obs']['images'][key] = torch.from_numpy(decompressed_images).to(self.device)
                        batch['obs_2']['images'][key] = torch.from_numpy(decompressed_images_2).to(self.device)

                    batch = dict_apply(batch, torch.from_numpy)

                    # get batch
                    obs_1 = self.normalizer.normalize(batch['obs'])  # batch_size * traj_len * obs_dim
                    act_1 = self.normalizer['action'].normalize(batch['action']) # batch_size * traj_len * action_dim
                    obs_2 = self.normalizer.normalize(batch['obs_2'])
                    act_2 = self.normalizer['action_2'].normalize(batch['action_2'])

                    this_nobs = dict_apply(obs_1, lambda x: x.reshape(-1, *x.shape[2:]))
                    nobs_features = self.obs_encoder(this_nobs)
                    global_cond = nobs_features.reshape(batch_size, sequence_length, -1)
                    this_nobs_2 = dict_apply(obs_2, lambda x: x.reshape(-1, *x.shape[2:]))
                    nobs_features_2 = self.obs_encoder(this_nobs_2)
                    global_cond_2 = nobs_features_2.reshape(batch_size, sequence_length, -1)

                    
                    s_a_1 = np.concatenate([global_cond, act_1], axis=-1)
                    s_a_2 = np.concatenate([global_cond_2, act_2], axis=-1)

                    conditions_1 = [
                        np.all(batch['votes'] == 1, axis=1),
                        np.all(batch['votes'] == 0, axis=1),
                        np.all(batch['votes'] == 0.5, axis=1)
                        ]
                    values = [1, 0, 0]
                    single_labels_1 = torch.from_numpy(np.select(conditions_1, values)).float().to(self.device)

                    conditions_2 = [
                        np.all(batch['votes_2'] == 1, axis=1),
                        np.all(batch['votes_2'] == 0, axis=1),
                        np.all(batch['votes_2'] == 0.5, axis=1)
                        ]
                    values = [1, 0, 0]
                    single_labels_2 = torch.from_numpy(np.select(conditions_2, values)).float().to(self.device)

                    pred_1_alpha_beta = self.model(torch.from_numpy(s_a_1).float().to(self.device)) # batch_size * 2
                    pred_1_alpha = pred_1_alpha_beta[:, 0]
                    pred_1_beta = pred_1_alpha_beta[:, 1]


                    pred_2_alpha_beta = self.model(torch.from_numpy(s_a_2).float().to(self.device))
                    pred_2_alpha = pred_2_alpha_beta[:, 0]
                    pred_2_beta = pred_2_alpha_beta[:, 1]
                    # if equal, then discard
                    # TODO maybe if equal, then both towards 1 (both preferred)

                    loss_1 = torch.mean((torch.log(pred_1_alpha) + torch.log(pred_2_beta))* single_labels_1 \
                            + (torch.log(pred_2_alpha) + torch.log(pred_1_beta)) * single_labels_2)

                    # var_1 = (pred_1_alpha * pred_1_beta) / ((pred_1_alpha + pred_1_beta) ** 2 * (pred_1_alpha + pred_1_beta + 1))
                    
                    # var_2 = (pred_2_alpha * pred_2_beta) / ((pred_2_alpha + pred_2_beta) ** 2 * (pred_2_alpha + pred_2_beta + 1))

                    # loss_2 = (torch.clamp(torch.mean(torch.sqrt(var_1)) - torch.sqrt(torch.tensor(1 / 324, dtype=torch.float32, device=self.device)), min=0)) ** 2 \
                    #         + (torch.clamp(torch.mean(torch.sqrt(var_2)) - torch.sqrt(torch.tensor(1 / 324, dtype=torch.float32, device=self.device)), min=0)) ** 2
                    loss_2 = torch.mean(torch.clamp(pred_1_alpha - 25, min=0) ** 2) + torch.mean(torch.clamp(pred_2_alpha - 25, min=0) ** 2) \
                            + torch.mean(torch.clamp(pred_1_beta - 25, min=0) ** 2) + torch.mean(torch.clamp(pred_2_beta - 25, min=0) ** 2)
                    
                    # loss_3 = torch.log(4 / pred_1_alpha.std()) + (pred_1_alpha.var() + (pred_1_alpha.mean() - 12.5) ** 2) / (2 * 16) -0.5 \
                    #         + torch.log(4 / pred_2_alpha.std()) + (pred_2_alpha.var() + (pred_2_alpha.mean() - 12.5) ** 2) / (2 * 16) -0.5 \
                    #         + torch.log(4 / pred_1_beta.std()) + (pred_1_beta.var() + (pred_1_beta.mean() - 12.5) ** 2) / (2 * 16) -0.5 \
                    #         + torch.log(4 / pred_2_beta.std()) + (pred_2_beta.var() + (pred_2_beta.mean() - 12.5) ** 2) / (2 * 16) -0.5

                    controls_alpha_1 = torch.distributions.Normal(torch.mean(single_labels_1) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_1)).rsample((pred_1_alpha.shape[0],))
                    controls_alpha_1 = torch.sort(controls_alpha_1, dim=0)[0].to(self.device)

                    controls_alpha_2 = torch.distributions.Normal(torch.mean(single_labels_2) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_2)).rsample((pred_2_alpha.shape[0],))
                    controls_alpha_2 = torch.sort(controls_alpha_2, dim=0)[0].to(self.device)

                    controls_beta_1 = torch.distributions.Normal(torch.mean(single_labels_2) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_2)).rsample((pred_2_alpha.shape[0],))
                    controls_beta_1 = torch.sort(controls_beta_1, dim=0)[0].to(self.device)

                    controls_beta_2 = torch.distributions.Normal(torch.mean(single_labels_1) * 12.5, \
                                        (12.5 / 3) * torch.mean(single_labels_1)).rsample((pred_1_alpha.shape[0],))
                    controls_beta_2 = torch.sort(controls_beta_2, dim=0)[0].to(self.device)

                    loss_3 = torch.mean((torch.sort(pred_1_alpha, dim=0)[0] - controls_alpha_1) ** 2) \
                            + torch.mean((torch.sort(pred_2_alpha, dim=0)[0] - controls_alpha_2) ** 2) \
                            + torch.mean((torch.sort(pred_1_beta, dim=0)[0] - controls_beta_1) ** 2) \
                            + torch.mean((torch.sort(pred_2_beta, dim=0)[0] - controls_beta_2) ** 2)

                    loss_4 = self.kl_regularizer_loss(pred_1_alpha.shape[0], alpha=pred_1_alpha, beta=pred_1_beta) \
                            + self.kl_regularizer_loss(pred_2_alpha.shape[0], alpha=pred_2_alpha, beta=pred_2_beta)

                    beta_loss = -loss_1 + loss_2 + loss_3 + self.beta_coef*loss_4

                    beta_loss_1.append(loss_1)
                    beta_loss_2.append(loss_2)
                    beta_loss_3.append(loss_3)
                    beta_loss_4.append(loss_4)

                    beta_loss_all.append(beta_loss)

                    self.opt.zero_grad()
                    beta_loss.backward()
                    self.opt.step()

                # Scheduler step
                avg_loss = torch.stack(beta_loss_all).mean().item()
                self.scheduler.step(avg_loss)

                beta_loss_1 = torch.stack(beta_loss_1, dim=0)
                beta_loss_2 = torch.stack(beta_loss_2, dim=0)
                beta_loss_3 = torch.stack(beta_loss_3, dim=0)
                beta_loss_4 = torch.stack(beta_loss_4, dim=0)
                beta_loss_all = torch.stack(beta_loss_all, dim=0)
                print("iteration:", epoch + 1)
                print("mean_beta_loss_data:", torch.mean(beta_loss_1).item())
                print("mean_beta_loss_control:", torch.mean(beta_loss_2).item())
                print("mean_beta_loss_control_2:", torch.mean(beta_loss_3).item())
                print("mean_beta_loss_kl:", torch.mean(beta_loss_4).item())
                print("mean_beta_loss_all:", torch.mean(beta_loss_all).item())

                if save_dir is not None and (epoch+1) % 200 == 0:
                    tmp_save_dir= Path(save_dir) / f'itr_{epoch+1}'
                    tmp_save_dir.mkdir(parents=True, exist_ok=True)
                    model_file = tmp_save_dir / 'beta_model.pth'
                    self.save_model(model_file)
        else:
            self.load_model(load_dir)

    def fit_data_discrete(self, dataset):

        obs_1 = dataset['observations']  # batch_size * traj_len * obs_dim
        obs_2 = dataset['observations_2']

        conditions = [np.all(dataset['labels'] == [1, 0], axis=1),
                      np.all(dataset['labels'] == [0, 1], axis=1),
                      np.all(dataset['labels'] == [0.5, 0.5], axis=1)]
        values = [1, -1, 0]
        single_labels = torch.from_numpy(np.select(conditions, values)).float().to(self.device)

        discrete_obs_1 = get_discrete_traj(obs_1, dim=2)
        discrete_obs_2 = get_discrete_traj(obs_2, dim=2)

        traj_alpha_beta_dict = init_trajectory_dict(discrete_obs_1 + discrete_obs_2)
        self.traj_alpha_beta_dict = get_trajectory_dict_from_pair(traj_alpha_beta_dict, discrete_obs_1, discrete_obs_2,
                                                                  single_labels)

    def get_alpha_beta_discrete(self, x, dim=2):
        traj = x.cpu().numpy()
        discrete_obs = get_discrete_traj(traj, dim=dim)[0]
        alpha = self.traj_alpha_beta_dict[discrete_obs][0]
        beta = self.traj_alpha_beta_dict[discrete_obs][1]
        return alpha, beta

    def get_normalized_alpha_beta_discrete(self, x, dim=2):
        traj = x.cpu().numpy()
        discrete_obs = get_discrete_traj(traj, dim=dim)[0]
        alpha = self.traj_alpha_beta_dict[discrete_obs][0]
        beta = self.traj_alpha_beta_dict[discrete_obs][1]
        # Normalize alpha and beta
        alpha_normalized = alpha / 10
        beta_normalized = beta / 10
        return alpha_normalized, beta_normalized

    def get_rescaled_alpha_beta_discrete(self, x, dim=2, rescale=10):
        traj = x.cpu().numpy()
        discrete_obs = get_discrete_traj(traj, dim=dim)[0]
        alpha = self.traj_alpha_beta_dict[discrete_obs][0]
        beta = self.traj_alpha_beta_dict[discrete_obs][1]
        # Normalize alpha and beta
        alpha_normalized = alpha / rescale
        beta_normalized = beta / rescale
        return alpha_normalized, beta_normalized

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