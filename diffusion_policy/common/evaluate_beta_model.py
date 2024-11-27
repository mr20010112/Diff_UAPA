import torch
from rlhf.reward_model import MLPRewardModel, TransformerRewardModel,ItrTransformerRewardModel
from utils import BetaNetwork
from rlhf.train_reward_model import load_queries_with_indices
import numpy as np
from fast_track.generate_d4rl_fake_labels import load_local_offline_dataset
from env.maze import maps
import h5py


beta_model = BetaNetwork(4, 2, 5e-5, 'cuda', 'CausalTransformer')

beta_model_path = '/home/mrq/project/diffusion_policy-main-test/data/beta_model/kitchen/itr_1600/beta_model.pth'

beta_model.load_model(beta_model_path)
# cfg = {}
# cfg['max_episode_length'] = maps.GUARDED_MAZE_1_MAXSTEPS
# cfg['data_paths'] = maps.GUARDED_MAZE_1_DATAPATH
# cfg['max_seq_len'] = 600
# dataset = load_local_offline_dataset(cfg)
load_dir = '/home/mrq/project/diffusion_policy-main-test/data/pref_data/kitchen/kitchen_prefdata.h5'
with h5py.File(load_dir, 'r') as f:
    dataset = f

alpha_beta_list_avoid = []
alpha_beta_list_risky = []
alpha_list_avoid = []
alpha_list_risky = []
beta_list_avoid = []
beta_list_risky = []
# mean
episode_step = 0
for i in range(len(dataset['observations']-1)):
    if dataset['terminals'][i] == False:
        episode_step += 1
        i += 1
        continue
    else:
        obs = dataset['observations'][i-episode_step:i+1]
        acs = dataset['actions'][i - episode_step:i + 1]
        x = np.concatenate([obs,acs],axis=-1)
        alpha, beta = beta_model.get_alpha_beta(torch.from_numpy(x).unsqueeze(0).to(beta_model.device))

        if dataset['risky'][i] == False:
            alpha_list_avoid.append(alpha.cpu().numpy())
            beta_list_avoid.append(beta.cpu().numpy())
            alpha_beta_list_avoid.append(torch.cat([alpha,beta]))
        else:
            alpha_list_risky.append(alpha.cpu().numpy())
            beta_list_risky.append(beta.cpu().numpy())
            alpha_beta_list_risky.append(torch.cat([alpha,beta]))
        episode_step = 0

print(f'avoid_num: {len(alpha_beta_list_avoid)}')
print(f'risky_num: {len(alpha_beta_list_risky)}')
print(f'mean_alpha_avoid: {np.array(alpha_list_avoid).mean()}')
print(f'mean_beta_avoid: {np.array(beta_list_avoid).mean()}')
print(f'mean_alpha_risky: {np.array(alpha_list_risky).mean()}')
print(f'mean_beta_risky: {np.array(beta_list_risky).mean()}')

# plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define the x values where we want to compute the PDF
x = np.linspace(0, 1, 100)

# Compute the PDF of Beta(1, 1)
pdf_beta_avoid = beta.pdf(x, np.array(alpha_list_avoid).mean(), np.array(beta_list_avoid).mean())

# Compute the PDF of Beta(2, 2)
pdf_beta_risky = beta.pdf(x, np.array(alpha_list_risky).mean(), np.array(beta_list_risky).mean())

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_beta_avoid, label='Avoid', color='blue')
plt.plot(x, pdf_beta_risky, label='Risky', color='red')
plt.title('Beta Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig('Beta.png')



































