import torch

def slice_episode(episode, horizon, stride):
        
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

    return torch.stack(sliced_fragments)
