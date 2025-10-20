Uncertainty-aware Preference Alignment for Diffusion Policies
This repository is the official implementation of the NeurIPS 2025 paper: Uncertainty-aware Preference Alignment for Diffusion Policies.
The implementation is based on the code from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

Abstract
Recent advancements in diffusion policies have demonstrated promising performance in decision-making tasks. To align these policies with human preferences, a common approach is incorporating Preference-based Reinforcement Learning (PbRL) into policy tuning. However, since preference data is practically collected from populations with different backgrounds, a key challenge lies in handling the inherent uncertainties in people's preferences during policy updates.
Setup Instructions
Create Python Environment

Ensure Conda is installed before proceeding.

Create a Conda environment and install the required packages:
conda env create -f conda_environment.yaml



Download Training Data
The following datasets are required for training:

Robomimic Data
Franka Kitchen
D4RL

Training Instructions
Behavior Cloning
Activate the Conda environment and log in to Weights & Biases (wandb) if you haven't already. Train a reference policy through behavior cloning:
python train.py --config-dir=. --config-name=train_diffusion_transformer_lowdim_workspace.yaml training.seed=42

This will create a directory in the format data/outputs/yyyy.mm.dd/hh.mm.ss_<method_name>_<task_name> where configuration files, logs, and checkpoints are saved.
Generate Trajectory Data
Trajectory data is generated using the policy trained via behavior cloning (requires a checkpoint). Alternatively, you can generate your own dataset with different settings:
# Collect lowdim Robomimic data
python train.py --config-dir=diffusion_policy/config --config-name=datacollect_bet_lowdim_can_online_workspace.yaml training.seed=42

# Collect lowdim Franka Kitchen data
python train.py --config-dir=diffusion_policy/config --config-name=datacollect_diffusion_transformer_lowdim_kitchen_workspace.yaml training.seed=42

Preference-based Reinforcement Learning (PbRL)
Further fine-tune the diffusion policy using preference-based reinforcement learning techniques:
# PbRL for diffusion policy in Robomimic task
python train.py --config-dir=diffusion_policy/config --config-name=pbrl_diffusion_transformer_lowdim_lift_online_workspace.yaml

# PbRL for diffusion policy in D4RL task
python train.py --config-dir=diffusion_policy/config --config-name=pbrl_diffusion_transformer_lowdim_halfcheetah_online_workspace.yaml

Evaluate Results
Run the evaluation script:
python eval.py --checkpoint data/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output

Citation
If you find our work useful in your research, please consider citing our paper:
@inproceedings{miao2025uncertainty,
  title={Uncertainty-aware Preference Alignment for Diffusion Policies},
  author={Miao, Runqing and Xu, Sheng and Zhao, Runyi and Chan, Wai Kin (Victor) and Liu, Guiliang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  url={[https://neurips.cc/virtual/2025/poster/116057](https://neurips.cc/virtual/2025/poster/116057)}
}

License
This project is licensed under the MIT License. See the LICENSE file for details.