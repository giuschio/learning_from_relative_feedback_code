"""
Train with an auxiliary policy aiding in exploration.

configs:

- "configs_exp/2024_09_26_FINAL_EXPLORATION/heuristic/training.yaml"
  Uses a simple heuristic (hit the closest ball) to guide exploration

- "configs_exp/2024_09_26_FINAL_EXPLORATION/1K_labels/1K_labels_seed_0.yaml"
  Uses a policy trained using 1K labels of relative feedback to guide exploration

- "configs_exp/2024_09_26_FINAL_EXPLORATION/1K_labels_tamer/1K_labels_seed_0.yaml"
  Uses a policy trained using 1K labels of scalar feedback to guide exploration

- "configs_exp/2024_09_26_FINAL_EXPLORATION/05K_labels_dagger/05K_labels_seed_0.yaml"
  Uses a policy trained using 500 demonstrations to guide exploration

"""

import argparse
import torch

from omegaconf import OmegaConf


from cuesim.environments import get_env_options, get_envs
from cuelearner.active_learning.policies import get_policy, load_policy_from_checkpoint
# from cuelearner.active_learning.policies import get_ensemble
from cuelearner.active_learning.policies.ddqn import DDQN

from cuelearner.common.utils import omega_to_yaml

from torch.utils.tensorboard import SummaryWriter

from cuelearner.active_learning.trainers.exploration_advice_billiards import Trainer
from cuelearner.common.players import get_player
from cuesim.heuristic_agents import get_heuristic


def main(config):
    env_options = get_env_options(config.env_type)
    train_env, val_env = get_envs(seed=config.seed, options=env_options, n=2)
    expert = get_player(config.advice_agent_type, config)

    policy = get_policy(
        policy_type=config.policy.type,
        **train_env.env_info,
        **config.policy.args,
        seed=config.seed,
    )

    if "advice_checkpoint" in config:
        advice_agent = load_policy_from_checkpoint(config.advice_checkpoint)
    elif "advice_heuristic" in config:
        advice_agent = get_heuristic(config.advice_heuristic, train_env.sim.options["physics"])

    checkpoint = None
    if checkpoint is not None:
        print("Starting training from checkpoint: ")
        print(f"  {checkpoint}")
        policy.load_state_dict(torch.load(checkpoint))
        policy.wipe_optimizer()

    trainer = Trainer(**config.trainer, seed=config.seed)
    omega_to_yaml(config, fname=trainer.log_dir.get_path("trainer_config.yaml"))
    trainer.fit(
        policy=policy,
        advice_cloner=advice_agent,
        train_env=train_env,
        val_env=val_env,
        expert=expert,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    default_config = "configs_exp/2024_09_26_FINAL_EXPLORATION/05K_labels_dagger/05K_labels_seed_0.yaml"
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Training for config at {args.config} for seed: {args.seed}")
    config = OmegaConf.load(args.config)
    config.trainer.verbose = True
    config.seed = args.seed
    main(config)
