"""
Train relative advice policy on top of a base policy (e.g. this relative advice policy (advice policy)
is then used in train_w_advice.py)
"""

import argparse

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from cuesim.environments import get_env_options, get_envs
from cuesim.heuristic_agents import get_heuristic
from cuelearner.active_learning.policies import get_policy
from cuelearner.common.utils import omega_to_yaml
from cuelearner.active_learning.trainers.improvement_advice import Trainer
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

    advice_cloner = get_policy(
        "advice_cloner",
        **train_env.env_info,
        **config.advice_cloner.args,
        seed=config.seed,
    )

    trainer = Trainer(**config.trainer, seed=config.seed)
    omega_to_yaml(config, fname=trainer.log_dir.get_path("trainer_config.yaml"))
    trainer.fit(
        policy=policy,
        advice_cloner=advice_cloner,
        train_env=train_env,
        val_env=val_env,
        expert=expert,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    default_config = "configs_exp/2024_05_31_adaptation_friction/advice_v1.yaml"
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    print(f"Training for config at {args.config} for seed: {args.seed}")
    config = OmegaConf.load(args.config)
    config.trainer.verbose = True
    config.seed = args.seed
    main(config)
