import torch
import hydra
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace

checkpoint = "data/outputs/2025.05.13/11.51.34_train_diffusion_real_robot0_real_robot/checkpoints/epoch=0000-train_action_mse_error=0.316.ckpt"
output_dir = "./Distributional-DPO-Robotics/data/pnp_output" # XXX XXX XXX
device = 'cuda:1'

    # load checkpoint
payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace = cls(cfg, output_dir=output_dir)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys='optimizer', include_keys=None)

policy_resolution = (cfg["shape_meta"]["obs"]["cam_left"]["shape"][2], cfg["shape_meta"]["obs"]["cam_left"]["shape"][1])

# get policy from workspace
policy = workspace.model
if cfg.training.use_ema:
    policy = workspace.ema_model

device = torch.device(device)
policy.to(device)
policy.eval()