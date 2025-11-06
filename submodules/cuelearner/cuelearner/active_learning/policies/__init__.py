import os
import yaml
import torch
import numpy as np

from cuelearner.active_learning.policies.td3 import TD3
from cuelearner.active_learning.policies.ddqn import DDQN
from cuelearner.active_learning.policies.advice_cloner import AdviceCloner
from cuelearner.active_learning.policies.behavior_cloner import BehaviorCloner
from cuelearner.active_learning.policies.ensemble import EnsemblePolicy
from cuelearner.common.data_structures import ReplayBuffer


class AdviceWrapper:
    def __init__(self, base_policy, advice_policy) -> None:
        self.base_policy = base_policy
        self.advice_policy = advice_policy

    def select_action(self, state):
        action = self.base_policy.select_action(state)
        action = self.advice_policy.select_action(state=state, action=action)
        return action


__policies = {
    # "td3": TD3,
    "ddqn": DDQN,
    "advice_cloner": AdviceCloner,
    "behavior_cloner": BehaviorCloner
}


def get_policy(policy_type, **kwargs):
    P = __policies[policy_type]
    return P(**kwargs)


def get_ensemble(policy_type, n, **kwargs):
    assert "seed" in kwargs

    P = __policies[policy_type]
    ensemble = list()
    for _ in range(n):
        kwargs["seed"] += 1
        ensemble.append(P(**kwargs))

    return EnsemblePolicy(ensemble)

class RandomPolicy:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def select_action(self, state):
        theta = np.random.uniform(0, 2 * np.pi)
        # Create the 2D vector with unit norm
        return np.array([np.cos(theta), np.sin(theta)])

    def select_action_batch(self, state):
        return np.array([self.select_action(s) for s in state])
    
class NoAdvicePolicy:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def select_action(self, state, action: np.ndarray):
        return action.copy()

    def select_action_batch(self, state, action):
        return action.copy()
    

def load_replay_buffer_from_checkpoint(fpath, device):
    return ReplayBuffer.from_file(fpath, device)


def load_policy_from_checkpoint(fpath):
    if fpath == "random":
        return RandomPolicy()
    elif fpath == "no_advice":
        return NoAdvicePolicy()

    hpath = os.path.join(os.path.dirname(fpath), "hparams.yaml")
    with open(hpath, "r") as file:
        hparams = yaml.safe_load(file)

    policy_type = hparams["policy_type"]
    hparams = hparams["hparams"]

    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    P = __policies[policy_type]
    instance = P(**hparams)
    instance.load_state_dict(torch.load(fpath, map_location=map_location))
    return instance


def load_base_policy_for_checkpoint(fpath):
    hpath = os.path.join(os.path.dirname(os.path.dirname(fpath)), "trainer_config.yaml")
    with open(hpath, "r") as file:
        hparams = yaml.safe_load(file)

    return load_policy_from_checkpoint(hparams['policy_checkpoint'])

def extract_base_policy_for_checkpoint(fpath):
    hpath = os.path.join(os.path.dirname(os.path.dirname(fpath)), "trainer_config.yaml")
    with open(hpath, "r") as file:
        hparams = yaml.safe_load(file)

    return hparams['policy_checkpoint']


def load_ensemble_from_checkpoint(fpath):
    hpath = os.path.join(os.path.dirname(fpath), "hparams.yaml")
    with open(hpath, "r") as file:
        hparams = yaml.safe_load(file)

    state_dict = torch.load(fpath)
    model_names = hparams.keys()

    ensemble = list()
    for model_name in model_names:
        hparams_i = hparams[model_name]
        state_dict_i = state_dict[model_name]

        policy_type_i = hparams_i["policy_type"]
        hparams_i = hparams_i["hparams"]
        P = __policies[policy_type_i]
        instance = P(**hparams_i)
        instance.load_state_dict(state_dict_i)
        ensemble.append(instance)

    return EnsemblePolicy(ensemble)
