import numpy as np


from cuesim.environments import get_env_options, get_envs
from cuesim.heuristic_agents import DirectShotAgent
from cuesim.utils import rotate_vec, angle

from cuelearner.active_learning.policies import load_policy_from_checkpoint
from cuelearner.common.models import q_map


def compute_advice(action, optimal_action, advice_step_size):
    next_actions = [
        rotate_vec(vector=action, degrees=d)
        for d in [-1.0 * advice_step_size, 0.0, advice_step_size]
    ]
    distances = np.array([angle(optimal_action, a) for a in next_actions])
    return next_actions[np.argmin(np.abs(distances))]


# def compute_advice_logit(action, optimal_action, advice_step_size):
#     next_actions = [
#         rotate_vec(vector=action, degrees=d)
#         for d in [-1.0 * advice_step_size, 0.0, advice_step_size]
#     ]
#     distances = np.array([angle(optimal_action, a) for a in next_actions])
#     return np.argmin(np.abs(distances))


class HeuristicPlayer:
    def __init__(self, config) -> None:
        env_options = get_env_options(config.env_type)
        self.expert_env = get_envs(seed=0, options=env_options, n=1)[0]
        self.heuristic = DirectShotAgent(self.expert_env.env_options["physics"])

    def get_action(self, state, info, action):
        # get action proposal
        heuristic_action = self.heuristic(info)
        # improve the action using the expert environment
        self.expert_env.reset_to(info)
        new_action = self.expert_env.get_closest_robust_action(heuristic_action)
        return new_action

    def get_advice(self, state, info, action):
        new_action = self.get_action(state, info, action)
        return compute_advice(action=action, optimal_action=new_action)


class LearnedPlayer:
    def __init__(self, config) -> None:
        self.expert_model = load_policy_from_checkpoint(config.expert_path).q

    def get_action(self, state, info, action):
        actions, est_rewards = q_map(self.expert_model, state, bin_size=1.0)
        expert_action = actions[np.argmax(est_rewards)]
        return expert_action

    def get_advice(self, state, info, action):
        new_action = self.get_action(state, info, action)
        return compute_advice(action=action, optimal_action=new_action)


class LearnedAugmentedPlayer:
    def __init__(self, config) -> None:
        self.expert_model = load_policy_from_checkpoint(config.expert_path).q
        env_options = get_env_options(config.env_type)
        self.expert_env = get_envs(seed=0, options=env_options, n=1)[0]

        self.heuristic = DirectShotAgent(self.expert_env.env_options["physics"])

    def get_action(self, state, info, action):
        actions, est_rewards = q_map(self.expert_model, state, bin_size=0.1)
        expert_action = actions[np.argmax(est_rewards)]

        self.expert_env.reset_to(info)
        new_action = self.expert_env.get_closest_greedy_action(
            expert_action, search_width=10.0
        )
        new_action = new_action if new_action is not None else expert_action
        return new_action

    def get_advice(self, state, info, action, advice_step_size):
        new_action = self.get_action(state, info, action)
        return compute_advice(
            action=action, optimal_action=new_action, advice_step_size=advice_step_size
        )

    # def get_advice_logit(self, state, info, action):
    #     new_action = self.get_action(state, info, action)
    #     return compute_advice_logit(action=action, optimal_action=new_action)


__players = {
    "learned": LearnedPlayer,
    "learned_augmented": LearnedAugmentedPlayer,
    "heuristic": HeuristicPlayer,
}


def get_player(player_type, config):
    return __players[player_type](config)
