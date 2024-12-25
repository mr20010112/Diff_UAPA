from typing import Optional, Dict
import numpy as np
from diffusion_policy.common.pref_replay_buffer import PrefReplayBuffer
import torch

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

class PrefSequenceSampler:
    def __init__(self,
                 replay_buffer: 'PrefReplayBuffer',
                 sequence_length: int,
                 episode_mask: Optional[np.ndarray]=None,
                 keys: Optional[Dict[str, int]] = None,
                 ):
        """
        Initializes a sampler for the preference replay buffer.
        
        Parameters:
        - replay_buffer: PrefReplayBuffer instance from which to sample data.
        - sequence_length: The length of sequences to sample.
        - pad_before, pad_after: Padding before and after sequences (optional).
        - keys: Optional dictionary to specify specific keys and limits on how much data to load.
        - episode_mask: Mask indicating valid episodes for sampling.
        """
        super().__init__()
        assert sequence_length >= 1

        if keys is None:
            keys = list(replay_buffer.data.keys())

        # Store generated indices
        self.keys = keys
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.episode_mask = episode_mask

    def __len__(self):
        
        return np.sum(self.episode_mask)

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Samples the sequence of data based on the provided index (idx).
        
        Parameters:
        - idx: The index from which to sample an episode sequence.
        
        Returns:
        - A dictionary containing the sampled data for the specified keys and votes.
        """
        result = {}
        indices = np.where(self.episode_mask)[0]
        result = self.replay_buffer.get_pref_episode(indices[idx])

        for key in result:
            value = result[key]
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value)
            elif isinstance(value, (np.float32, np.float64, float, int)):
                result[key] = torch.tensor(value, dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported type {type(value)} for key '{key}'")

        return result