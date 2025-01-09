from typing import Dict, Tuple
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F
import einops
from typing import Optional, Tuple

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.libraries.batch_loss_fn import BatchFocalLoss, soft_cross_entropy
from diffusion_policy.model.bet.utils import eval_mode
from diffusion_policy.model.common.slice import slice_episode

class BETLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            action_ae: KMeansDiscretizer, 
            obs_encoding_net: nn.Module, 
            state_prior: MinGPT,
            gamma,
            horizon,
            n_action_steps,
            n_obs_steps,
            map_ratio=0.1,
            bias_reg=0.0,
            beta=1.0):
        super().__init__()
    
        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        self.obs_encoding_net = obs_encoding_net
        self.state_prior = state_prior
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.gamma = gamma
        self.beta = beta

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon

        # pad To to T
        obs = torch.full((B,T,Do), -2, dtype=nobs.dtype, device=nobs.device)
        obs[:,:To,:] = nobs[:,:To,:]

        # (B,T,Do)
        enc_obs = self.obs_encoding_net(obs)

        # Sample latents from the prior
        latents, offsets = self.state_prior.generate_latents(enc_obs)

        # un-descritize
        naction_pred = self.action_ae.decode_actions(
            latent_action_batch=(latents, offsets)
        )
        # (B,T,Da)

        # un-normalize
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)
    
    def get_latents(self, latent_collection_loader):
        training_latents = list()
        with eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in latent_collection_loader:
                obs, act = observations.to(self.device, non_blocking=True), action.to(self.device, non_blocking=True)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    training_latents.append(latent.detach())
        training_latents_tensor = torch.cat(training_latents, dim=0)
        return training_latents_tensor

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.state_prior.get_optimizer(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def get_pred_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.state_prior.predict_offsets:
            target_latents, target_offsets = target_latents
        is_soft_target = (target_latents.shape[-1] == self.state_prior.vocab_size) and (
            self.state_prior.vocab_size != 1
        )
        if is_soft_target:
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(target_latents.size(0),-1)
            if self.state_prior.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = BatchFocalLoss(gamma=self.state_prior.focal_loss_gamma)
        if self.state_prior.predict_offsets:
            # print(obs_rep._version)
            output, _ = self.state_prior.model(obs_rep)
            logits = output[:, :, : self.state_prior.vocab_size]
            offsets = output[:, :, self.state_prior.vocab_size :]
            batch = logits.shape[0]
            seq = logits.shape[1]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.state_prior.vocab_size,
                A=self.state_prior.action_dim,
            )
            # calculate (optionally soft) cross entropy and offset losses
            class_loss = criterion(logits, target_latents)
            # offset loss is only calculated on the target class
            # if soft targets, argmax is considered the target class
            selected_offsets = offsets[
                torch.arange(offsets.size(0)),
                target_latents.view(-1).argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1),
            ]
            offset_loss = self.state_prior.offset_loss_scale * F.mse_loss(
                selected_offsets.view(batch, -1, self.state_prior.action_dim), target_offsets, reduction='none'
            )

            offset_loss = offset_loss.mean(dim=(1, 2))
            loss = offset_loss + class_loss
        else:
            logits, _ = self.state_prior.model(obs_rep)
            loss = criterion(logits, target_latents)

        return loss

    def compute_loss(self, batch, ref_policy: Optional[BaseLowdimPolicy] = None) -> torch.Tensor:
        # normalize input
        assert 'valid_mask' not in batch

        To = self.n_obs_steps

        observations_1 = batch["obs"].to(self.device).detach()
        actions_1 = batch["action"].to(self.device).detach()
        votes_1 = batch["votes"].to(self.device).detach()
        observations_2 = batch["obs_2"].to(self.device).detach()
        actions_2 = batch["action_2"].to(self.device).detach()
        votes_2 = batch["votes_2"].to(self.device).detach()

        threshold = 1e-2
        diff = torch.abs(votes_1 - votes_2)
        condition_1 = (votes_1 > votes_2) & (diff >= threshold)  # votes_1 > votes_2 and diff >= threshold
        condition_2 = (votes_1 < votes_2) & (diff >= threshold)  # votes_1 < votes_2 and diff >= threshold

        votes_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_1 = torch.squeeze(votes_1, dim=-1).detach()
        votes_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_2 = torch.squeeze(votes_2, dim=-1).detach()

        threshold = 1e-2
        diff = torch.abs(votes_1 - votes_2)
        condition_1 = (votes_1 > votes_2) & (diff >= threshold)  # votes_1 > votes_2 and diff >= threshold
        condition_2 = (votes_1 < votes_2) & (diff >= threshold)  # votes_1 < votes_2 and diff >= threshold

        votes_1 = torch.where(condition_1, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_1 = torch.squeeze(votes_1, dim=-1).detach()
        votes_2 = torch.where(condition_2, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        votes_2 = torch.squeeze(votes_2, dim=-1).detach()

        batch_1 = {
            'obs': torch.tensor(observations_1, device=self.device),
            'action': torch.tensor(actions_1, device=self.device),
        }

        batch_2 = {
            'obs': torch.tensor(observations_2, device=self.device),
            'action': torch.tensor(actions_2, device=self.device),
        }

        nbatch_1 = self.normalizer.normalize(batch_1)
        nbatch_2 = self.normalizer.normalize(batch_2)

        obs_1 = nbatch_1['obs']
        action_1 = nbatch_1['action']
        obs_2 = nbatch_2['obs']
        action_2 = nbatch_2['action']

        obs_1 = slice_episode(obs_1, horizon=self.horizon, stride=self.horizon)
        action_1 = slice_episode(action_1, horizon=self.horizon, stride=self.horizon)
        obs_2 = slice_episode(obs_2, horizon=self.horizon, stride=self.horizon)
        action_2 = slice_episode(action_2, horizon=self.horizon, stride=self.horizon)

        traj_loss_1, traj_loss_2 = 0, 0
        immatation_loss_1, immatation_loss_2 = 0, 0

        for i in range(len(obs_1)):
            obs_1_slide = obs_1[i]
            obs_1_slide[:,To:,:] = -2
            action_1_slide = action_1[i]

            enc_obs_1 = self.obs_encoding_net(obs_1_slide)
            latent_1 = self.action_ae.encode_into_latent(action_1_slide, enc_obs_1)

            loss_1 = self.get_pred_loss(
                obs_rep=enc_obs_1.clone(),
                target_latents=latent_1,
            )

            ref_loss_1 = ref_policy.get_pred_loss(
                obs_rep=enc_obs_1.clone(),
                target_latents=latent_1,
            ).detach()

            # _, loss_1 = self.state_prior.get_latent_and_loss(
            #     obs_rep=enc_obs_1.clone(),
            #     target_latents=latent_1,
            # )

            #traj_loss_1 += -torch.norm(action_1_slide-pred_action_1['action'], dim=-1)*(self.gamma ** (i*self.n_action_steps + torch.arange(0, self.n_action_steps, device=self.device))).reshape(1,-1)
            traj_loss_1 = traj_loss_1 - ((loss_1 - ref_loss_1)*torch.tensor(self.gamma**(i*self.horizon), device=self.device))
            immatation_loss_1 = immatation_loss_1 + loss_1

            obs_2_slide = obs_2[i]
            obs_2_slide[:,To:,:] = -2
            action_2_slide = action_2[i]

            enc_obs_2 = self.obs_encoding_net(obs_2_slide)
            latent_2 = self.action_ae.encode_into_latent(action_2_slide, enc_obs_2)

            loss_2 = self.get_pred_loss(
                obs_rep=enc_obs_2.clone(),
                target_latents=latent_2,
            )

            ref_loss_2 = ref_policy.get_pred_loss(
                obs_rep=enc_obs_2.clone(),
                target_latents=latent_2,
            ).detach()

            # _, loss_2 = self.state_prior.get_latent_and_loss(
            #     obs_rep=enc_obs_2.clone(),
            #     target_latents=latent_2,
            # )

            #traj_loss_2 += -torch.norm(action_2_slide-pred_action_2['action'], dim=-1)*(self.gamma ** (i*self.n_action_steps + torch.arange(0, self.n_action_steps, device=self.device))).reshape(1,-1)
            traj_loss_2 = traj_loss_2 - ((loss_2 - ref_loss_2)*torch.tensor(self.gamma**(i*self.horizon), device=self.device))
            immatation_loss_2 = immatation_loss_2 + loss_2

        # traj_loss_1 = torch.sum(traj_loss_1, dim=-1)
        # traj_loss_2 = torch.sum(traj_loss_2, dim=-1)
        diff_loss = torch.mean(torch.abs(traj_loss_1 - traj_loss_2))

        mle_loss_1 = -F.logsigmoid(self.beta*(traj_loss_1 - traj_loss_2)) + immatation_loss_1/(2*len(obs_1)*self.n_obs_steps) + immatation_loss_2/(2*len(obs_2)*self.n_obs_steps)
        mle_loss_2 = -F.logsigmoid(self.beta*(traj_loss_2 - traj_loss_1)) + immatation_loss_1/(2*len(obs_1)*self.n_obs_steps) + immatation_loss_2/(2*len(obs_2)*self.n_obs_steps)

        loss = (votes_1.to(self.device) * mle_loss_1 + votes_2.to(self.device) * mle_loss_2)

        return torch.mean(loss)
