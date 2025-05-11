import torch
import numpy as np

def slice_episode(episode, horizon, stride):

    is_torch = hasattr(episode, 'cuda')
        
    shape = episode.shape
    N, T = shape[:2] 
    feature_dim =shape[2:]

    sliced_fragments = []

    for start in range(0, T, stride):
        end = start + horizon

        if end > T:
            start = max(0, T - horizon)
            end = T

        fragment = episode[:, start:end, ...]
        sliced_fragments.append(fragment)

        if end == T:
            break

    if is_torch:
        return torch.stack(sliced_fragments)
    else:
        return np.stack(sliced_fragments)


def slice_episode_time(episode, horizon, stride):
    
    is_torch = hasattr(episode, 'cuda')
    
    shape = episode.shape
    T = shape[0]
    
    sliced_fragments = []
    
    for start in range(0, T, stride):
        end = start + horizon
        
        if end > T:
            start = max(0, T - horizon)
            end = T
        
        fragment = episode[start:end, ...]
        sliced_fragments.append(fragment)
        
        if end == T:
            break
    
    if is_torch:
        return torch.stack(sliced_fragments)
    else:
        return np.stack(sliced_fragments)