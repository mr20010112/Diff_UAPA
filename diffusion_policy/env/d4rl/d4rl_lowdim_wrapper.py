from typing import List, Dict, Optional
import gym
import numpy as np
from gym.spaces import Box
import d4rl

class D4rlLowdimWrapper(gym.Env):
    def __init__(self, 
                 env: gym.Env, 
                 obs_keys: Optional[List[str]] = None,
                 init_state: Optional[np.ndarray] = None,
                 render_hw=(256, 256),
                 render_mode='rgb_array'):
        """
        Wrapper for d4rl environments.

        Args:
            env (gym.Env): The d4rl environment to wrap.
            obs_keys (List[str], optional): Observation keys to extract (if required).
            init_state (np.ndarray, optional): Initial state to reset to.
            render_hw (tuple): Height and width for rendering.
            render_mode (str): Render mode ('rgb_array' or 'human').
        """
        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_mode = render_mode
        self._seed = None
        self.seed_state_map = {}

        # Action space
        self.action_space = env.action_space

        # Observation space
        if obs_keys:
            obs_example = self.get_observation()
            low = np.full_like(obs_example, fill_value=-1)
            high = np.full_like(obs_example, fill_value=1)
            self.observation_space = Box(
                low=low,
                high=high,
                shape=low.shape,
                dtype=low.dtype
            )
        else:
            self.observation_space = env.observation_space

    def get_observation(self):
        """
        Extracts the relevant observations based on specified obs_keys.
        """
        raw_obs = self.env.reset()  # Example reset for observation extraction
        if self.obs_keys:
            obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
            return obs
        return raw_obs

    def seed(self, seed=None):
        """
        Sets the seed for reproducibility.
        """
        np.random.seed(seed=seed)
        self.env.seed(seed)
        self._seed = seed

    def reset(self):
        """
        Resets the environment to the initial state or a random state.
        """
        if self.init_state is not None:
            self.env.set_state(self.init_state)
        elif self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                self.env.set_state(self.seed_state_map[seed])
            else:
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.sim.get_state()
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            self.env.reset()

        obs = self.get_observation()
        return obs

    def step(self, action):
        """
        Steps the environment with the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            obs: The next observation.
            reward: The reward received.
            done: Whether the episode has ended.
            info: Additional info.
        """
        raw_obs, reward, done, info = self.env.step(action)
        if self.obs_keys:
            obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        else:
            obs = raw_obs
        return obs, reward, done, info

    def render(self, mode=None):
        """
        Renders the environment.

        Args:
            mode: The render mode (e.g., 'rgb_array', 'human').

        Returns:
            Rendered image or other output depending on the mode.
        """
        mode = mode or self.render_mode
        h, w = self.render_hw
        if mode == 'rgb_array':
            return self.env.render(mode=mode, width=w, height=h)
        else:
            return self.env.render(mode=mode)

# Test function to validate the wrapper
def test():
    import matplotlib.pyplot as plt

    env = gym.make('hopper-medium-v2')  # Example d4rl environment
    wrapper = D4RLWrapper(env=env)

    wrapper.seed(42)
    obs = wrapper.reset()

    for _ in range(5):
        action = wrapper.action_space.sample()
        obs, reward, done, info = wrapper.step(action)
        if done:
            break

    img = wrapper.render()
    if img is not None:
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    test()
