"""
Train the auxiliary exploration policy using behavior cloning and dataset aggregation
"""

import argparse

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from cuesim.environments import get_env_options, get_envs
from cuesim.heuristic_agents import get_heuristic
from cuelearner.active_learning.policies import get_policy
from cuelearner.common.utils import omega_to_yaml
from cuelearner.active_learning.trainers.dagger_prior_billiards import Trainer
from cuelearner.active_learning.policies import load_policy_from_checkpoint
from cuelearner.common.players import get_player


def main(config):
    env_options = get_env_options(config.env_type, config.get("env_options", None))

    train_env, val_env = get_envs(seed=config.seed, options=env_options, n=2)
    # expert = get_player(config.advice_agent_type, config)
    expert = load_policy_from_checkpoint(config.expert_path)

    # load base policy
    if ".ckpt" in config.base_policy:
        policy = load_policy_from_checkpoint(config.base_policy)
        policy.set_eval()
    else:
        # assume it's a heuristic
        policy = get_heuristic(config.base_policy, train_env.sim.options["physics"])

    behavior_cloner = get_policy(
        "behavior_cloner",
        **train_env.env_info,
        **config.behavior_cloner.args,
        seed=config.seed,
    )

    trainer = Trainer(**config.trainer, seed=config.seed)
    omega_to_yaml(config, fname=trainer.log_dir.get_path("trainer_config.yaml"))
    trainer.fit(
        policy=policy,
        behavior_cloner=behavior_cloner,
        train_env=train_env,
        val_env=val_env,
        expert=expert,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    default_config = "configs_exp/2025_02_20_dagger_exploration_policies/dagger_policy.yaml"
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Training for config at {args.config} for seed: {args.seed}")
    config = OmegaConf.load(args.config)
    config.trainer.verbose = True
    config.seed = args.seed
    main(config)
