import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusers import DDPMScheduler  # Import the noise scheduler

# Example initialization parameters
obs_dim = 9  # Adjust according to your actual observation dimension
action_dim = 2  # Adjust according to your action dimension
n_obs_steps = 128
n_action_steps = 50
horizon = 150

# Create dummy data for observations and actions
obs_data = torch.randn(1000, n_obs_steps, obs_dim)  # Example shape for observations
action_data = torch.randn(1000, n_obs_steps, action_dim)  # Example shape for actions

# Initialize and fit the normalizer
normalizer = LinearNormalizer()
normalizer.fit({'obs': obs_data, 'action': action_data})

# Debugging: Check if 'obs' exists in params_dict
print(f"Keys in params_dict: {normalizer.params_dict.keys()}")
if 'obs' not in normalizer.params_dict:
    raise ValueError("The 'obs' key is missing from the normalizer.")
else:
    print("'obs' key is correctly initialized.")

# Check the structure of 'obs' in params_dict
print(f"Structure of 'obs' in params_dict: {normalizer.params_dict['obs']}")

# Initialize the noise scheduler properly
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)  # Adjust based on your requirements

# Initialize the policy model (replace None with actual model initialization)
model = None  # Replace with ConditionalUnet1D model initialization

policy = DiffusionUnetLowdimPolicy(
    model=model,
    noise_scheduler=noise_scheduler,  # Now correctly initialized
    horizon=horizon,
    obs_dim=obs_dim,
    action_dim=action_dim,
    n_action_steps=n_action_steps,
    n_obs_steps=n_obs_steps
)

# Set the normalizer in the policy
policy.set_normalizer(normalizer)
print("Normalizer has been set in the policy.")

# Create a dummy observation dictionary
obs_dict = {'obs': torch.randn(1, n_obs_steps, obs_dim)}  # Adjust based on your input data format

# Debugging: Check the shape and structure of obs_dict['obs']
print(f"obs_dict['obs'] shape: {obs_dict['obs'].shape}")

# Run predict_action and catch potential errors
try:
    result = policy.predict_action(obs_dict)
    print("Result from predict_action:", result)
except Exception as e:
    print(f"Error during predict_action: {e}")
