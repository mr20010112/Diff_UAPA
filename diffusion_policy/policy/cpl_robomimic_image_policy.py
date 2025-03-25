from typing import Dict
import torch
import numpy as np
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
import torch.nn.functional as F
from typing import Optional, Tuple

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from diffusion_policy.model.common.slice import slice_episode

class RobomimicImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            gamma,
            horizon,
            n_action_steps,
            n_obs_steps,
            use_map=False,
            bias_reg=0.25,
            beta=1.0,
            map_ratio=1.0,
            algo_name='bc_rnn',
            obs_type='image',
            task_name='square',
            dataset_type='ph',
            crop_shape=(76,76)
        ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)

        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.config = config
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_map = use_map
        self.gamma = gamma
        self.beta = beta
        self.bias_reg = bias_reg

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_dict = self.normalizer(obs_dict)
        robomimic_obs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result

    def reset(self):
        self.model.reset()

    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer) :
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, ref_policy: Optional[BaseImagePolicy] = None, obs_keys: Optional[np.ndarray[str]] = None, avg_Traj_loss=0.0) -> torch.Tensor:
        # normalize input
        assert 'valid_mask' not in batch

        To = self.n_obs_steps
        Ta = self.n_action_steps

        observations_1 = {key:batch[key].to(self.device) for key in obs_keys}
        actions_1 = batch["action"].to(self.device)
        votes_1 = batch["votes"].to(self.device)
        length_1 = batch["length"].to(self.device).detach()
        observations_2 = {key:batch[f'{key}_2'].to(self.device) for key in obs_keys}
        actions_2 = batch["action_2"].to(self.device)
        votes_2 = batch["votes_2"].to(self.device)
        length_2 = batch["length_2"].to(self.device).detach()

        threshold = 1e-2
        diff = torch.abs(votes_1 - votes_2)
        condition_1 = (votes_1 > votes_2) & (diff >= threshold)  # votes_1 > votes_2 and diff >= threshold
        condition_2 = (votes_1 < votes_2) & (diff >= threshold)  # votes_1 < votes_2 and diff >= threshold

        votes_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_1 = torch.squeeze(votes_1, dim=-1).detach()
        votes_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_2 = torch.squeeze(votes_2, dim=-1).detach()

        #exchange data
        mask = condition_2.squeeze(-1)

        actions_1[mask], actions_2[mask] = actions_2[mask], actions_1[mask]

        for key in obs_keys:
            observations_1[key][mask], observations_2[key][mask] = (
                observations_2[key][mask],
                observations_1[key][mask]
            )

        length_1[mask], length_2[mask] = length_2[mask], length_1[mask]

        # normalize input
        obs_1 = self.normalizer.normalize(observations_1)
        action_1 = self.normalizer['action'].normalize(actions_1)
        obs_2 = self.normalizer.normalize(observations_2)
        action_2 = self.normalizer['action'].normalize(actions_2)

        # slice episode
        stride = self.horizon

        obs_1 = {key:slice_episode(obs_1[key], horizon=self.horizon, stride=stride) for key in obs_1.keys()}
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=stride)
        obs_2 = {key:slice_episode(obs_2[key], horizon=self.horizon, stride=stride) for key in obs_2.keys()}
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=stride)

        loss = 0
        traj_loss_1, traj_loss_2 = 0, 0
        immatation_loss_1, immatation_loss_2 = 0, 0

        for i in range(len(action_1)):
            obs_1_slide = {key:obs_1[key][i] for key in obs_1.keys()}
            action_1_slide = action_1[i]

            robomimic_batch = {
                'obs': obs_1_slide,
                'actions': action_1_slide
            }
            input_batch = self.model.process_batch_for_training(robomimic_batch)
            predictions = self.model._forward_training(input_batch)
            loss_1 = self.model._compute_losses(predictions, input_batch)
            loss_1 = loss_1["action_loss"]

            mask_1 = (self.horizon + (i-1)*stride) <= length_1
            mask_1 = mask_1.int()

            traj_loss_1 += traj_loss_1 - (loss_1*torch.tensor(self.gamma**(i*self.horizon), device=self.device))
            immatation_loss_1 = immatation_loss_1 + loss_1

        for i in range(len(action_2)):
            obs_2_slide = {key:obs_2[key][i] for key in obs_2.keys()}
            action_2_slide = action_2[i]
            
            robomimic_batch_2 = {
                'obs': obs_2_slide,
                'actions': action_2_slide
            }
            input_batch_2 = self.model.process_batch_for_training(robomimic_batch_2)
            predictions_2 = self.model._forward_training(input_batch_2)
            loss_2 = self.model._compute_losses(predictions_2, input_batch_2)
            loss_2 = loss_2["action_loss"]

            mask_2 = (self.horizon + (i-1)*stride) <= length_2
            mask_2 = mask_2.int()

            traj_loss_2 = traj_loss_2 - (loss_2*torch.tensor(self.gamma**(i*self.horizon), device=self.device))
            immatation_loss_2 = immatation_loss_2 + loss_2

        mean_loss_1 = torch.mean(torch.abs(traj_loss_1 - self.bias_reg *  traj_loss_2))
        # mean_loss_2 = torch.mean(torch.abs(traj_loss_2 - self.bias_reg *  traj_loss_1))
        immatation_loss = immatation_loss_1 + immatation_loss_2

        mle_loss_1 = -F.logsigmoid(self.beta*(traj_loss_1 - self.bias_reg * traj_loss_2)) + immatation_loss/((len(action_1) + len(action_2))*self.horizon)
        mle_loss_2 = -F.logsigmoid(self.beta*(traj_loss_2 - self.bias_reg * traj_loss_1)) + immatation_loss/((len(action_1) + len(action_2))*self.horizon)

        loss += mle_loss_1#mle_loss_1

        return loss #torch.mean(loss)
    
    def on_epoch_end(self, epoch):
        self.model.on_epoch_end(epoch)

    def get_optimizer(self):
        return self.model.optimizers['policy']


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg.shape_meta

    policy = RobomimicImagePolicy(shape_meta=shape_meta)

