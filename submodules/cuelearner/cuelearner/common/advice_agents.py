import numpy as np

from cuelearner.common.models import q_map
from cuelearner.common.utils import rotate_vector, angle


def argmax_local(v, threshold=0.9):
    # Extend the vector to handle wrap-around
    extended_vector = np.concatenate([[v[-1]], v, [v[0]]])

    # Find local maxima
    local_maxima_indices = []
    for i in range(1, len(extended_vector) - 1):
        if (
            extended_vector[i] > extended_vector[i - 1]
            and extended_vector[i] > extended_vector[i + 1]
            and extended_vector[i] > threshold
        ):
            # Adjust index for original vector
            original_index = i - 1
            local_maxima_indices.append(original_index)

    return np.array(local_maxima_indices, dtype=np.int32)


def argmin_local(v, threshold=-0.9):
    return argmax_local(-1.0 * v, threshold=-1.0 * threshold)


def get_closest_expert_action(expert, state, action, threshold=0.9):
    actions, est_rewards = q_map(expert, state, bin_size=0.1)
    indexes = argmax_local(est_rewards, threshold)
    if len(indexes) == 0:
        return None
    expert_actions = actions[indexes]

    # Find the closest expert action to the original action
    angles = np.array([angle(action, e) for e in expert_actions])
    closest_expert_action = expert_actions[np.argmin(np.abs(angles))]
    return closest_expert_action


def get_closest_interesting_action(expert, state, action):
    actions, est_rewards = q_map(expert, state)
    indexes = np.concatenate((argmax_local(est_rewards), argmin_local(est_rewards)))
    if len(indexes) == 0:
        return None
    expert_actions = actions[indexes]

    # Find the closest expert action to the original action
    angles = np.array([angle(action, e) for e in expert_actions])
    closest_expert_action = expert_actions[np.argmin(np.abs(angles))]
    return closest_expert_action


def get_locally_improved_action(expert, state, action, threshold=0.9):
    closest_ea = get_closest_expert_action(expert, state, action, threshold)
    if closest_ea is None:
        return None
    # possible next actions
    next_actions = [rotate_vector(action, d) for d in [-5, 0, 5]]
    # Choose the next best action (could be the same as before)
    angles = np.array([angle(closest_ea, a) for a in next_actions])
    return next_actions[np.argmin(np.abs(angles))]


def get_globally_improved_action(expert, state, action):
    actions, est_rewards = q_map(expert, state, bin_size=0.1)
    expert_action = actions[np.argmax(est_rewards)]
    if expert_action is None:
        return None
    # possible next actions
    next_actions = [rotate_vector(action, d) for d in [-5, 0, 5]]
    # Choose the next best action (could be the same as before)
    angles = np.array([angle(expert_action, a) for a in next_actions])
    return next_actions[np.argmin(np.abs(angles))]
