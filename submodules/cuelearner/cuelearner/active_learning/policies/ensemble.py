import numpy as np

from cuelearner.common.utils import angle


def find_clusters(vectors, radius):
    clusters = []

    for vector in vectors:
        added = False
        for cluster in clusters:
            # Check if vector is within the diameter for all members of the cluster
            if all(angle(vector, member) < 2 * radius for member in cluster):
                cluster.append(vector)
                added = True
                break
        if not added:
            clusters.append([vector])

    return clusters


def find_largest_cluster_center(vectors, radius=2.0):
    clusters = find_clusters(vectors, radius)
    largest_cluster = max(clusters, key=len)
    cluster_center = np.mean(largest_cluster, axis=0)
    return cluster_center


class EnsemblePolicy:
    def __init__(self, policies: list) -> None:
        self.policies = policies

    def __getitem__(self, index):
        return self.policies[index]

    def __len__(self):
        return len(self.policies)

    @property
    def hparams(self):
        return {f"model_{i}": self[i].hparams for i in range(len(self))}

    def fit(self, replay_buffer, *args, **kwargs):
        res = [self[i].fit(replay_buffer, *args, **kwargs) for i in range(len(self))]
        res = {key: sum(d[key] for d in res) / len(res) for key in res[0]}
        return res

    def select_action(self, *args, **kwargs):
        return self[0].select_action(*args, **kwargs)

    def select_action_ensemble(self, *args, **kwargs):
        actions = [self[i].select_action(*args, **kwargs) for i in range(len(self))]
        return actions

    def get_state_dict(self):
        return {f"model_{i}": self[i].get_state_dict() for i in range(len(self))}

    def set_train(self):
        [self[i].set_train() for i in range(len(self))]

    def set_eval(self):
        [self[i].set_eval() for i in range(len(self))]
