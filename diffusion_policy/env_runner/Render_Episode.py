import os
import pathlib
import wandb
from gym.spaces import Box
import numpy as np
import cv2  # for saving videos
import tqdm
from diffusion_policy.env_runner.kitchen_lowdim_runner import KitchenLowdimRunner

# Assuming 'env' is your environment instance, e.g., KitchenLowdimWrapper

def render_episode(env: KitchenLowdimRunner, episode, render_hw=(240, 360), fps=12.5, output_dir='data/rlhf/video', name = None):
    # Create the output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    video_path = os.path.join(output_dir, f"episode_{wandb.util.generate_id()}.mp4")
    
    # Set up video writer using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, render_hw[::-1])

    # Reset environment with the first observation
    env.reset()

    for t, (obs, action) in enumerate(zip(episode['obs'], episode['actions'])):
        # Perform step in the environment with the action
        env.step(action)

        # Render frame from the environment
        frame = env.render(mode='rgb_array')
        
        # Convert frame to BGR (OpenCV uses BGR, while Gym renders in RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        video_writer.write(frame)

    # Release video writer resources
    video_writer.release()
    print(f"Episode rendered and saved at {video_path}")

    return video_path

# Example to loop over multiple episodes:
def render_all_episodes(env, episodes, output_dir='./videos', render_hw=(240, 360)):
    for episode_idx, episode in enumerate(tqdm.tqdm(episodes)):
        render_episode(env, episode, render_hw, output_dir=output_dir)

# Assuming episodes are in a list of dictionaries
# episodes = [{'observations': ..., 'actions': ...}, ...]
render_all_episodes(env, episodes)
