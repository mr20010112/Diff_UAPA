from typing import Optional, Dict
import numpy as np
from diffusion_policy.common.pref_replay_buffer import PrefReplayBuffer
import torch

class PrefSequenceSampler:
    def __init__(self,
                 replay_buffer: 'PrefReplayBuffer',
                 sequence_length: int,
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

    def __len__(self):
        return self.replay_buffer.data['obs'].shape[0]

    def sample_sequence(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Samples the sequence of data based on the provided index (idx).
        
        Parameters:
        - idx: The index from which to sample an episode sequence.
        
        Returns:
        - A dictionary containing the sampled data for the specified keys and votes.
        """
        result = {}

        result = self.replay_buffer.get_pref_episode(idx)

        for key in result:
            result[key] = torch.from_numpy(result[key])

        return result