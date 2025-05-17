import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces
import multiprocessing as mp
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

module_logger = logging.getLogger(__name__)

class KitchenLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            dataset_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=280,
            n_obs_steps=2,
            n_action_steps=8,
            render_hw=(240,360),
            fps=12.5,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            abs_action=False,
            robot_noise_ratio=0.1,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        task_fps = 12.5
        steps_per_render = int(max(task_fps // fps, 1))

        def env_fn():
            from diffusion_policy.env.kitchen.v0 import KitchenAllV0
            from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
            env = KitchenAllV0(use_abs_action=abs_action)
            env.robot_noise_ratio = robot_noise_ratio
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    KitchenLowdimWrapper(
                        env=env,
                        init_qpos=None,
                        init_qvel=None,
                        render_hw=tuple(render_hw)
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps
                if n_obs_steps >= n_action_steps else n_action_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        all_init_qpos = np.load(pathlib.Path(dataset_dir) / "all_init_qpos.npy")
        all_init_qvel = np.load(pathlib.Path(dataset_dir) / "all_init_qvel.npy")
        module_logger.info(f'Loaded {len(all_init_qpos)} known initial conditions.')

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis
            init_qpos = None
            init_qvel = None
            if i < len(all_init_qpos):
                init_qpos = all_init_qpos[i]
                init_qvel = all_init_qvel[i]

            def init_fn(env, init_qpos=init_qpos, init_qvel=init_qvel, enable_render=enable_render):
                from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set initial condition
                assert isinstance(env.env.env, KitchenLowdimWrapper)
                env.env.env.init_qpos = init_qpos
                env.env.env.init_qvel = init_qvel
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set initial condition
                assert isinstance(env.env.env, KitchenLowdimWrapper)
                env.env.env.init_qpos = None
                env.env.env.init_qvel = None

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        def dummy_env_fn():
            # Avoid importing or using env in the main process
            # to prevent OpenGL context issue with fork.
            # Create a fake env whose sole purpos is to provide 
            # obs/action spaces and metadata.
            env = gym.Env()
            env.observation_space = gym.spaces.Box(
                -8, 8, shape=(60,), dtype=np.float32)
            env.action_space = gym.spaces.Box(
                -8, 8, shape=(9,), dtype=np.float32)
            env.metadata = {
                'render.modes': ['human', 'rgb_array', 'depth_array'],
                'video.frames_per_second': 12
            }
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps
                if n_obs_steps >= n_action_steps else n_action_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
            return env
        
        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec


    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        last_info = [None] * n_inits

        # initialize lists to collect data
        observations, actions, rewards, terminals = [[] for _ in range(n_envs)], [[] for _ in range(n_envs)], \
        [[] for _ in range(n_envs)], [[] for _ in range(n_envs)]

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
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            if self.n_obs_steps < self.n_action_steps:

                # start rollout
                obs = env.reset()
                past_action = None
                policy.reset()

                for i in range(n_envs):
                    observations[i].extend(obs[i, -1, ...].reshape(1, -1))

                pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BlockPushLowdimRunner {chunk_idx+1}/{n_chunks}", 
                    leave=False, mininterval=self.tqdm_interval_sec)
                all_done = False
                while not all_done:
                    # create obs dict
                    np_obs_dict = {
                        'obs': obs[:, -self.n_obs_steps:, ...].astype(np.float32)
                    }
                    if self.past_action and (past_action is not None):
                        np_obs_dict['past_action'] = past_action[
                            :, -(self.n_obs_steps-1):].astype(np.float32)
                    
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, 
                        lambda x: torch.from_numpy(x).to(
                            device=device))

                    # run policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict, 
                        lambda x: x.detach().to('cpu').numpy())
                    
                    action = np_action_dict['action']

                    # step env
                    # obs, reward, done, info = env.step(action)
                    obs, reward, done, info = env.step(action)
                    all_done = np.all(done)
                    # past_action = action
                    past_action = action

                    # collect data
                    for i in range(n_envs):
                        if not all_done:
                            observations[i].extend(obs[i, ...])
                        else:
                            observations[i].extend(obs[i, :-1, ...])
                        actions[i].extend(action[i, ...])
                        # rewards[i].extend([reward[i]] * len(obs[i, ...]))
                        if not done[i]:
                            terminals[i].extend([False] * len(action[i, ...]))
                        else:
                            terminals[i].extend([False] * (len(action[i, ...])-1) + [True])
                        
                    # update pbar
                    pbar.update(action.shape[1])
                pbar.close()

            else:

                # start rollout
                obs = env.reset()
                past_action = None
                policy.reset()

                for i in range(n_envs):
                    observations[i].extend(obs[i, -1, ...].reshape(1, -1))

                pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BlockPushLowdimRunner {chunk_idx+1}/{n_chunks}", 
                    leave=False, mininterval=self.tqdm_interval_sec)
                all_done = False
                while not all_done:
                    # create obs dict
                    np_obs_dict = {
                        'obs': obs.astype(np.float32)
                    }
                    if self.past_action and (past_action is not None):
                        np_obs_dict['past_action'] = past_action[
                            :, -(self.n_obs_steps-1):].astype(np.float32)
                    
                    # device transfer
                    obs_dict = dict_apply(np_obs_dict, 
                        lambda x: torch.from_numpy(x).to(
                            device=device))

                    # run policy
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_dict)

                    # device_transfer
                    np_action_dict = dict_apply(action_dict, 
                        lambda x: x.detach().to('cpu').numpy())
                    
                    action = np_action_dict['action']

                    # step env
                    # obs, reward, done, info = env.step(action)
                    obs, reward, done, info = env.step(action)
                    all_done = np.all(done)
                    # past_action = action
                    past_action = action

                    # collect data
                    for i in range(n_envs):
                        if not all_done:
                            observations[i].extend(obs[i, -(self.n_action_steps):, ...])
                        else:
                            observations[i].extend(obs[i, -(self.n_action_steps):-1, ...])
                        actions[i].extend(action[i, ...])
                        # rewards[i].extend([reward[i]] * len(obs[i, ...]))
                        if not done[i]:
                            terminals[i].extend([False] * len(action[i, ...]))
                        else:
                            terminals[i].extend([False] * (len(action[i, ...])-1) + [True])
                        
                    # update pbar
                    pbar.update(action.shape[1])
                pbar.close()


            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

            for i in range(n_envs):
                episode_reward = np.array(all_rewards[i])
                indices = np.argwhere(episode_reward == 1.0).flatten()
                if indices.size == 0:
                    rewards[i].extend([0.0] * len(episode_reward))
                else:
                    prev_index = 0
                    for j, idx in enumerate(indices):
                        rewards[i].extend([1.0 / (idx - prev_index)] * (idx - prev_index))
                        prev_index = idx 
                    
                    rewards[i].extend([0.0] * (len(episode_reward) - prev_index))


        # reward is number of tasks completed, max 7
        # use info to record the order of task completion?
        # also report the probability of completing n tasks (different aggregation of reward).

        # log
        log_data = dict()
        prefix_total_reward_map = collections.defaultdict(list)
        prefix_n_completed_map = collections.defaultdict(list)
        
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            this_rewards = all_rewards[i]
            total_reward = np.sum(this_rewards) / 7
            prefix_total_reward_map[prefix].append(total_reward)

            n_completed_tasks = len(last_info[i]['completed_tasks'])
            prefix_n_completed_map[prefix].append(n_completed_tasks)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in prefix_total_reward_map.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        for prefix, value in prefix_n_completed_map.items():
            n_completed = np.array(value)
            for i in range(7):
                n = i + 1
                p_n = np.mean(n_completed >= n)
                name = prefix + f'p_{n}'
                log_data[name] = p_n

        final_observations, final_actions, final_rewards, final_terminals = [], [], [], []

        for i in range(n_envs):
            idx = np.argmax(terminals[i]) + 1
            final_observations.extend(observations[i][:idx])
            final_actions.extend(actions[i][:idx])
            final_rewards.extend(rewards[i][:idx])
            final_terminals.extend(terminals[i][:idx])

        episode_data = {
            'observations': np.array(final_observations),
            'actions': np.array(final_actions),
            'rewards': np.array(final_rewards),
            'terminals': np.array(final_terminals),
        }


        # Return the data in the desired format (2D arrays: time x feature dimension)
        return log_data, episode_data