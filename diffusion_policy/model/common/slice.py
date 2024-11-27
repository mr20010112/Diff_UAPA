import torch

def slice_episode(episode, horizon, stride):
    N, T, feature_dim = episode.shape
    sliced_fragments = []

    # 以 stride 为步幅进行切片
    for start in range(0, T, stride):
        end = start + horizon

        # 如果剩余步数不足 horizon，则调整开始和结束索引以取到完整的最后一个片段
        if end > T:
            start = max(0, T - horizon)
            end = T

        fragment = episode[:, start:end, :]
        sliced_fragments.append(fragment)

        # 如果 end 已到达总长度，则结束循环
        if end == T:
            break

    # 将片段列表转换为 PyTorch 张量
    return torch.stack(sliced_fragments)

