import collections
import numpy as np
import gym
from tqdm import trange
import torch
import torch.nn as nn
import math


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Creates a learning rate schedule with a linear warmup phase followed by a constant learning rate.
    
    Args:
        optimizer: The optimizer for which to adjust the learning rate (e.g., PyTorch optimizer).
        num_warmup_steps: Number of steps for the linear warmup phase.
        last_epoch: The index of the last epoch (default: -1 for starting from scratch).
    
    Returns:
        A scheduler object with a get_lr method to compute the learning rate for each step.
    """
    class ConstantScheduleWithWarmup:
        def __init__(self, optimizer, num_warmup_steps, last_epoch=-1):
            self.optimizer = optimizer
            self.num_warmup_steps = max(num_warmup_steps, 1)  # Ensure at least 1 to avoid division by zero
            self.last_epoch = last_epoch
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]  # Store initial learning rates
            
        def step(self):
            """
            Updates the learning rate for each parameter group in the optimizer.
            Called at each training step.
            """
            self.last_epoch += 1
            lr = self.get_lr()
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr[i]
        
        def get_lr(self):
            """
            Computes the learning rate for each parameter group based on the current step.
            
            Returns:
                List of learning rates for each parameter group.
            """
            if self.last_epoch < self.num_warmup_steps:
                # Linear warmup: lr increases linearly from 0 to base_lr over num_warmup_steps
                multiplier = self.last_epoch / self.num_warmup_steps
            else:
                # Constant phase: lr remains at base_lr
                multiplier = 1.0
            return [base_lr * multiplier for base_lr in self.base_lrs]
    
    return ConstantScheduleWithWarmup(optimizer, num_warmup_steps, last_epoch)

class RunningMeanStd:
    def __init__(self, mean=0, std=1.0, epsilon=np.finfo(np.float32).eps.item(), 
                 mode='common', lr=0.1):
        self.mean, self.var = mean, std
        self.max = mean
        self.count = 0
        self.eps = epsilon
        self.lr = lr
        self.mode = mode

    def update(self, data_array) -> None:
        """Add a batch of item into RMS with the same shape, modify mean/var/count."""
        batch_mean, batch_var = np.mean(data_array, axis=0), np.var(data_array, axis=0)
        batch_count = len(data_array)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        if self.mode == 'common':
            new_mean = self.mean + delta * batch_count / total_count
            new_max = self.max + (np.max(data_array)-self.max) / total_count
        else:
            new_mean = self.mean + delta * self.lr
            new_max = self.max + (np.max(data_array)-self.max) * self.lr
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count
        self.max = new_max

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        """
        x: (batch,)
        return: (batch, dim)
        """
        device, half_dim = x.device, self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)          # (B, half_dim)
        return torch.cat([emb.sin(), emb.cos()], dim=-1) # (B, dim)

def to_torch(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)

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


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-5 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


@torch.no_grad()
def reward_from_preference(
    dataset: D4RLDataset,
    reward_model,
    batch_size: int = 256,
    reward_model_type: str = "transformer",
    device="cuda"
):
    data_size = dataset["rewards"].shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset["rewards"])
    
    if reward_model_type == "transformer":
        max_seq_len = reward_model.max_seq_len
        for each in reward_model.ensemble:
            each.eval()
 
        obs, act = [], []
        ptr = 0
        for i in trange(data_size):
            
            if len(obs) < max_seq_len:
                obs.append(dataset["observations"][i])
                act.append(dataset["actions"][i])
            
            if dataset["terminals"][i] > 0 or i == data_size - 1 or len(obs) == max_seq_len:
                tensor_obs = to_torch(np.array(obs)[None,], dtype=torch.float32).to(device)
                tensor_act = to_torch(np.array(act)[None,], dtype=torch.float32).to(device)
                
                new_reward = 0
                for each in reward_model.ensemble:
                    new_reward += each(tensor_obs, tensor_act).detach().cpu().numpy()
                new_reward /= len(reward_model.ensemble)
                if tensor_obs.shape[1] <= -1:
                    new_r[ptr:ptr+tensor_obs.shape[1]] = dataset["rewards"][ptr:ptr+tensor_obs.shape[1]]
                else:
                    new_r[ptr:ptr+tensor_obs.shape[1]] = new_reward
                ptr += tensor_obs.shape[1]
                obs, act = [], []
    else:
        for i in trange(interval):
            start_pt = i * batch_size
            end_pt = (i + 1) * batch_size

            observations = dataset["observations"][start_pt:end_pt]
            actions = dataset["actions"][start_pt:end_pt]
            obs_act = np.concatenate([observations, actions], axis=-1)

            new_reward = reward_model.get_reward_batch(obs_act).reshape(-1)
            new_r[start_pt:end_pt] = new_reward
    
    dataset["rewards"] = new_r.copy()
    
    # rr = dataset["rewards"].copy()
    # fr = new_r.copy()
    
    # rr = (rr-rr.min())/(rr.max()-rr.min())
    # fr = (fr-fr.min())/(fr.max()-fr.min())
    
    # rr_n_bins, _ = np.histogram(rr, 10, (0, 1))
    # fr_n_bins, _ = np.histogram(fr, 10, (0, 1))
    
    # print(rr_n_bins)
    # print(fr_n_bins)
    
    return dataset

class SinusoidalPosmb(nn.Module):
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
        
        self.causal_transformer = nn.TransformerEncoder(
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
        
        # Generate a new mask for the current sequence length
        mask = nn.Transformer.generate_square_subsequent_mask(2*traj_len).to(obs.device)
        
        pos = self.pos_emb(torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos
        
        x = torch.empty((batch_size, 2*traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.causal_transformer(x, mask)[:, 1::2]
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