"""
Train a residual policy using RL (not used)
"""


import argparse
import torch

from omegaconf import OmegaConf

from cuesim.environments import get_env_options, get_envs
from cuelearner.active_learning.policies import get_policy
from cuelearner.common.utils import omega_to_yaml
from torch.utils.tensorboard import SummaryWriter

from cuelearner.active_learning.trainers.residual_rl import Trainer
from cuelearner.active_learning.policies import load_policy_from_checkpoint
from cuesim.heuristic_agents import get_heuristic

from cuesim.utils import uniform_2d_numpy

def main(config):
    env_options = get_env_options(config.env_type, config.get('env_options', None))
    train_env, val_env = get_envs(seed=config.seed, options=env_options, n=2)

    # load base policy
    if ".ckpt" in config.base_policy:
        policy = load_policy_from_checkpoint(config.base_policy)
        policy.set_eval()
    else:
        # assume it's a heuristic
        policy = get_heuristic(config.base_policy, train_env.sim.options["physics"])


    env_info = train_env.env_info
    env_info['state_dim'] += 2
    env_info['action_set'] = uniform_2d_numpy(interval=0.1, angles_range=(-1.01, 1.01))
    residual_policy = get_policy(
        policy_type=config.policy.type,
        **env_info,
        **config.policy.args,
        seed=config.seed,
    )

    trainer = Trainer(**config.trainer, seed=config.seed)
    omega_to_yaml(config, fname=trainer.log_dir.get_path("trainer_config.yaml"))
    trainer.fit(policy, residual_policy, train_env, val_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    default_config = "configs_exp/2024_10_11_FINETUNING/rl_residual/gravity_bias_easiest.yaml"
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Training for config at {args.config} for seed: {args.seed}")
    config = OmegaConf.load(args.config)
    config.trainer.verbose = True
    config.seed = args.seed
    main(config)
