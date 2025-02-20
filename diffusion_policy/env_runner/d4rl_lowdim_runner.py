import os
import collections
import math
import tqdm
import numpy as np
import torch
import dill
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.d4rl.d4rl_lowdim_wrapper import D4rlLowdimWrapper
from gym import make as gym_make

class D4rlEnvRunner(BaseLowdimRunner):
    def __init__(self,
                 output_dir,
                 env_name,
                 n_train=10,
                 train_start_idx=0,
                 n_test=22,
                 test_start_seed=10000,
                 max_steps=1000,
                 n_obs_steps=2,
                 n_action_steps=8,
                 n_latency_steps=0,
                 past_action=False,
                 tqdm_interval_sec=5.0,
                 n_envs=None):

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # Handle latency steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        def env_fn():
            env = gym_make(env_name)
            return MultiStepWrapper(
                env,
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixes = list()
        env_init_fn_dills = list()

        # Training environments
        for i in range(n_train):
            train_idx = train_start_idx + i

            def init_fn(env, seed=train_idx):
                env.seed(seed)

            env_seeds.append(train_idx)
            env_prefixes.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # Testing environments
        for i in range(n_test):
            seed = test_start_seed + i

            def init_fn(env, seed=seed):
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixes.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # Create asynchronous environment
        env = AsyncVectorEnv(env_fns)

        self.env_name = env_name
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixes = env_prefixes
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)

            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            obs = env.reset()
            policy.reset()

            done = False
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {self.env_name} {chunk_idx+1}/{n_chunks}",
                             leave=False, mininterval=self.tqdm_interval_sec)

            while not done:
                obs_dict = {
                    'obs': obs[:, :self.n_obs_steps].astype(np.float32)
                }
                obs_dict = dict_apply(obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Invalid action detected.")

                obs, reward, done, info = env.step(action)
                done = np.all(done)

                pbar.update(action.shape[1])
            pbar.close()

            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        cumulative_rewards = collections.defaultdict(list)
        log_data = {}


        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixes[i]
            cumulative_reward = np.sum(np.array(all_rewards[i]))
            cumulative_rewards[prefix].append(cumulative_reward)
            log_data[prefix + f'sim_cumulative_reward_{seed}'] = cumulative_reward

        for prefix, value in cumulative_rewards.items():
            name = prefix + 'cumulative_rewards'
            value = np.mean(value)
            log_data[name] = value

        return log_data
