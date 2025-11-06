"""
Trains a DDQN agent (optionally starting from a checkpoint)
for the environment Cuesim/ThreeBallHard-Cuelearner

Used for baselines
"""

import argparse
import torch

from omegaconf import OmegaConf

from cuesim.environments import get_env_options, get_envs
from cuelearner.active_learning.policies import get_policy
from cuelearner.common.utils import omega_to_yaml
from torch.utils.tensorboard import SummaryWriter

from cuelearner.active_learning.trainers.baseline_trainer import Trainer


def main(config):
    env_options = get_env_options(config.env_type, config.get('env_options', None))
    train_env, val_env = get_envs(seed=config.seed, options=env_options, n=2)

    policy = get_policy(
        policy_type=config.policy.type,
        **train_env.env_info,
        **config.policy.args,
        seed=config.seed,
    )

    if 'policy_checkpoint' in config is not None:
        print("Starting training from checkpoint: ")
        print(f"  {config.policy_checkpoint}")
        policy.load_state_dict(torch.load(config.policy_checkpoint))
        policy.wipe_optimizer()

    trainer = Trainer(**config.trainer, seed=config.seed)
    omega_to_yaml(config, fname=trainer.log_dir.get_path("trainer_config.yaml"))
    trainer.fit(policy, train_env, val_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    default_config = "configs/ddqn_baseline.yaml"
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Training for config at {args.config} for seed: {args.seed}")
    config = OmegaConf.load(args.config)
    config.trainer.verbose = True
    config.seed = args.seed
    main(config)
