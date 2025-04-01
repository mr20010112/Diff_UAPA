# Uncertainty-aware Preference Alignment for Diffusion Policies

1. The implementation is based on the code from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

## Create Python Environment 
1. Please install the conda before proceeding.
2. Create a conda environment and install the packages:
   
```
conda env create -f conda_environment.yaml
```


## Download Training Data
[Robomimic Data](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)
[Franka Kitchen](https://github.com/google-research/relay-policy-learning/raw/refs/heads/master/kitchen_demos_multitask.zip?download=)
[D4RL](https://sites.google.com/view/d4rl-anonymous/)

## Behaviour Cloning
Activate conda environment and login to [wandb](https://wandb.ai) (if you haven't already).
Now we're going to train a reference policy through behaviour cloning.
```console
python train.py --config-dir=. --config-name=train_diffusion_transformer_lowdim_workspace.yaml training.seed=42
```
This will create a directory in format `data/outputs/yyyy.mm.dd/hh.mm.ss_<method_name>_<task_name>` where configs, logs and checkpoints are written to.
## Generate Trajectory Data
Note that we have generated the trajectory data through the policy trained by behaviour cloning.

Alternatively, you can also generate your own dataset with different settings:
```
# collect lowdim robomimic data
python train.py --config-dir=diffusion_policy/config --config-name=datacollect_bet_lowdim_can_online_workspace.yaml training.seed=42

# collect lowdim franka kitchen data
python train.py --config-dir=diffusion_policy/config --config-name=datacollect_diffusion_transformer_lowdim_kitchen_workspace.yaml training.seed=42
```

## Pbrl for Algorithms

```
# pbrl for diffusion policy in robomimic task
python train.py --config-dir=diffusion_policy/config --config-name=pbrl_diffusion_transformer_lowdim_lift_online_workspace.yaml

# pbrl for diffusion policy in d4rl task
python train.py --config-dir=diffusion_policy/config --config-name=pbrl_diffusion_transformer_lowdim_halfcheetah_online_workspace.yaml
```

## Evaluate Results
Run the evaluation script:
```console
python eval.py --checkpoint data/0550-test_mean_score=0.969.ckpt --output_dir data/pusht_eval_output
```

