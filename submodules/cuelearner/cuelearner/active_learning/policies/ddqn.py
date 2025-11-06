import itertools
import inspect
import numpy as np
import os
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.utils import clip_grad_norm_ as clip_grad

from copy import deepcopy as cpd
from cuelearner.common.models import QNetwork
from cuelearner.common.utils import clip_norm
import matplotlib.pyplot as plt
import io


def sample_argmax(q_values):
    return torch.argmax(q_values, dim=1)


def sample_softmax(q_values, tau=1.0):
    probabilities = F.softmax(q_values / tau, dim=1)
    return torch.multinomial(probabilities, 1).squeeze()


def sample_thompson(q_values):
    # TODO: this makes quite a lot of assumptions on the Q-values...
    # if not used, throw away
    # Normalize Q-values by their sum to get probabilities
    q_values = torch.clamp(q_values, min=0) + 0.01
    q_values_sum = q_values.sum(dim=1, keepdim=True)
    probabilities = q_values / q_values_sum
    # Handle potential division by zero
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    # Sample from the probability distribution for each state
    return torch.multinomial(probabilities, 1).squeeze()


__sampling_functions = dict(
    argmax=sample_argmax, softmax=sample_softmax, thompson=sample_thompson
)


def angle(v1, v2):
    """
    Calculate the positive angle in degrees between two vectors.
    """
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Calculate the angle in degrees
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))


def get_actions(q_net, states, action_set, sampling="argmax"):
    N = states.size(0)
    M = action_set.size(0)

    # Expand states to match each action
    expanded_states = states.unsqueeze(1).expand(-1, M, -1).reshape(N * M, -1)

    # Repeat actions to match the expanded states
    repeated_actions = action_set.repeat(N, 1)

    # Call Q function on each state-action pair
    q_values = q_net(expanded_states, repeated_actions).reshape(N, M)

    idx = __sampling_functions[sampling](q_values)
    return action_set[idx]


def get_entropy(q_net, states, action_set):
    N = states.size(0)
    M = action_set.size(0)

    # Expand states to match each action
    expanded_states = states.unsqueeze(1).expand(-1, M, -1).reshape(N * M, -1)

    # Repeat actions to match the expanded states
    repeated_actions = action_set.repeat(N, 1)

    # Call Q function on each state-action pair
    q_values = q_net(expanded_states, repeated_actions).reshape(N, M)
    action_probs = torch.softmax(q_values, dim=1)
    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1)

    # Step 3: Normalize the entropy
    max_entropy = torch.log(torch.tensor(M, dtype=q_values.dtype, device=q_values.device))  # log(m)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def find_vectors_within_angle(action_set, target_vector, interval_degrees, epsilon=0.01):
    """
    Find all vectors in the action set within a specified angular interval of the target vector.

    Parameters:
        action_set (torch.Tensor): A tensor of shape (n, 2), where each row is a unit norm 2D vector.
        target_vector (torch.Tensor): A 1D tensor of shape (2,) representing the target vector with unit norm.
        interval_degrees (float): The angular interval in degrees within which to select vectors.
        epsilon (float): A small value added to the interval for tolerance.

    Returns:
        torch.Tensor: A subset of action_set containing vectors within the specified interval.
    """
    # Convert interval from degrees to radians and find the cosine
    interval_radians = torch.deg2rad(torch.tensor(interval_degrees + epsilon))
    cos_interval = torch.cos(interval_radians)
    
    # Compute dot products between target_vector and each vector in action_set
    dot_products = torch.mv(action_set, target_vector)
    
    # Select vectors whose dot product with target_vector is above cos_interval
    mask = dot_products >= cos_interval
    return action_set[mask]


class DDQN(object):
    policy_type = "ddqn"

    def __init__(
        self,
        state_dim,
        action_dim,
        min_action,
        max_action,
        action_set,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        target_update_freq=2,
        hidden_sizes_critic=[400, 300],
        training_sampling="softmax",
        eval_sampling="argmax",
        optimizer="sgd",
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=0.0,
        fit_policy="fixed",
        seed=None,
        reduced_action_set=None,
        patience=10,
        **kwargs,
    ):
        self.__hparams = self.__get_params(inspect.currentframe())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(kwargs) > 0:
            print("The following args were specified but are not used by the policy:")
            [print(f"- {k}") for k in kwargs.keys()]

        self.__rng = torch.Generator()
        self.__rng.manual_seed(seed)

        # --- initialize network given a seed
        torch.manual_seed(seed)
        self.q = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes_critic,
        ).to(self.device)
        # ---
        self.q_target = cpd(self.q)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.OPT = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)[optimizer]
        self.critic_optimizer = self.OPT(
            self.q.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        self.q_0 = cpd(self.q)
        self.q_target_0 = cpd(self.q_target)

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.target_update_freq = target_update_freq
        self.total_it = 0

        self.to_torch = lambda x: torch.tensor(
            x, dtype=torch.float32, device=self.device
        )

        self.action_set = self.to_torch(action_set)
        reduced_action_set = self.action_set if reduced_action_set is None else reduced_action_set
        # most of the time during training is spent calculating the value function
        # to set the Q-value target
        # if the action set is very big, this can take a long time
        # it is sometimes beneficial to perform the argmax on a slightly reduced action set
        self.reduced_action_set = self.to_torch(reduced_action_set)
        self.batch_size = batch_size

        self.training_sampling = training_sampling
        self.eval_sampling = eval_sampling
        self.patience = patience

        self.fit = dict(
            restarts=self.fit_restart,
            fixed=self.fit_fixed,
        )[fit_policy]
        self.is_privileged = False

    def select_action_train(self, state):
        return self.__select_action(state, self.training_sampling)

    def select_action(self, state, action=None):
        return self.__select_action(state, self.eval_sampling)
    
    def select_action_near(self, state, action, interval):
        action_set = find_vectors_within_angle(self.action_set, self.to_torch(action), interval)
        return self.__select_action_near(state, action_set, self.eval_sampling)
    
    def __select_action_near(self, state, action_set, sampling):
        # numpy array
        with torch.no_grad():
            state = self.to_torch(state).unsqueeze(0)
            action = get_actions(self.q, state, action_set, sampling).squeeze(0)
        return action.detach().cpu().numpy()

    def select_action_train_batch(self, state):
        return np.array(
            [self.__select_action(s, self.training_sampling) for s in state]
        )

    def select_action_batch(self, state):
        return np.array([self.__select_action(s, self.eval_sampling) for s in state])

    def __select_action(self, state, sampling):
        # numpy array
        with torch.no_grad():
            state = self.to_torch(state).unsqueeze(0)
            action = get_actions(self.q, state, self.action_set, sampling).squeeze(0)
        return action.detach().cpu().numpy()
    
    def get_feature_vector(self, state, action):
        # numpy array
        with torch.no_grad():
            state = self.to_torch(state)
            action = self.to_torch(action)
            
            # Check if state and action are batched; if not, add batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
        
        feat = self.q.get_feature_vector(state, action)
        
        return feat.detach().cpu().numpy()

    def get_q_estimate(self, state, action):
        # numpy arrays
        with torch.no_grad():
            state = self.to_torch(state).unsqueeze(0)
            action = self.to_torch(action).unsqueeze(0)
            q_estimate = self.q(state, action).squeeze(0)
        return q_estimate.item()

    def __get_action_uncertainty(self, state):
        with torch.no_grad():
            state = self.to_torch(state).unsqueeze(0)
            uncertainty = get_entropy(self.q, state, self.action_set)
        return uncertainty.detach().cpu().numpy()
    
    def get_action_uncertainty(self, state):
        return self.__get_action_uncertainty(state)
    
    def get_action_uncertainty_batch(self, state):
        return np.array([self.__get_action_uncertainty(s) for s in state])

    def __get_params(self, frame) -> dict:
        args, _, _, values = inspect.getargvalues(frame)
        values.pop("self")
        values.pop("kwargs")
        return values

    @property
    def hparams(self):
        return {"policy_type": self.policy_type, "hparams": self.__hparams}

    def loss(self, batch):
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]

        if "reward_n" in batch:
            done_n = batch["done_n"]
            reward_n = batch["reward_n"]
            next_state_n = batch["next_state_n"]
            lookahead = batch["lookahead"]

        with torch.no_grad():
            if self.discount > 0.0:
                next_state = batch["next_state"]
                done = batch["done"]
                # most of the time during training is spent here
                # time grows if the discretization of actions gets finer
                # Select action according to policy and add clipped noise
                noise = torch.randn(
                    action.size(), dtype=action.dtype, generator=self.__rng
                )
                noise = noise.to(action.device) * self.policy_noise
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                # Box action space
                # next_action = (self.ac_target.pi(next_state) + noise).clamp(
                #     -self.max_action, self.max_action
                # )
                # spherical action space
                # select next actions
                next_action = get_actions(self.q, next_state, self.reduced_action_set, "argmax")
                next_action = clip_norm(
                    next_action + noise,
                    min_norm=self.min_action,
                    max_norm=self.max_action,
                )
                target_Q = reward + (1.0 - done) * self.discount * self.q_target(
                    next_state, next_action
                )

                if "reward_n" in batch:
                    next_action_n = get_actions(
                        self.q, next_state_n, self.reduced_action_set, "argmax"
                    )
                    target_Q_n = reward_n + (1.0 - done_n) * (
                        self.discount**lookahead
                    ) * self.q_target(next_state_n, next_action_n)

            else:
                target_Q = reward

        # Get current Q estimates
        current_Q = self.q(state, action)
        if "reward_n" in batch:
            # Compute critic loss
            critic_loss = 0.5 * F.mse_loss(current_Q, target_Q) + 0.5 * F.mse_loss(
                current_Q, target_Q_n
            )
        else:
            critic_loss = F.mse_loss(current_Q, target_Q)
        return critic_loss

    def train(self, batch):
        self.total_it += 1
        critic_loss = self.loss(batch)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.gradient_clipping > 0:
            clip_grad(self.q.parameters(), max_norm=self.gradient_clipping)
        self.critic_optimizer.step()

        if self.total_it % self.target_update_freq == 0:
            # update the target network
            for param, target_param in zip(
                self.q.parameters(), self.q_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        # the trainer expects critic_loss and actor_loss
        return critic_loss.item()

    def val(self, batch):
        return self.loss(batch).item()

    def fit_fixed(self, replay_buffer, n_steps):
        train_losses = [
            self.train(replay_buffer.sample(self.batch_size, "train"))
            for _ in range(n_steps)
        ]
        with torch.no_grad():
            val_losses = [
                self.val(replay_buffer.sample(self.batch_size, "val"))
                for _ in range(n_steps)
            ]

        return {
            "critic_loss_train": np.mean(train_losses),
            "critic_loss_val": np.mean(val_losses),
        }

    # def fit_restart(self, replay_buffer, n_steps):
    #     self.total_it = 0
    #     self.q, self.q_target = cpd(self.q_0), cpd(self.q_target_0)
    #     self.critic_optimizer = self.OPT(self.q.parameters(), lr=self.learning_rate)

    #     td = Subset(replay_buffer, replay_buffer.train_indexes)
    #     vd = Subset(replay_buffer, replay_buffer.val_indexes)

    #     td = DataLoader(td, batch_size=min(self.batch_size, len(td)), shuffle=True)
    #     vd = DataLoader(vd, batch_size=min(self.batch_size, len(vd)))

    #     patience, patience_counter = 10, 0
    #     best_val_loss = float("inf")
    #     best_train_loss = float("inf")

    #     q_pt, q_target_pt = None, None
    #     for epoch in range(1000):
    #         self.q.train()
    #         self.q_target.train()
    #         training_loss = 0.0
    #         for idx, batch in enumerate(td):
    #             training_loss += self.train(batch)[0]
    #         training_loss = training_loss / (idx + 1)

    #         val_loss = 0.0
    #         self.q.eval()
    #         self.q_target.eval()
    #         with torch.no_grad():  # No gradients needed
    #             for idx, batch in enumerate(vd):
    #                 val_loss += self.val(batch)[0]
    #         val_loss = val_loss / (idx + 1)

    #         # Early stopping check
    #         if val_loss < best_val_loss:
    #             patience_counter = 0
    #             best_val_loss = val_loss
    #             best_train_loss = training_loss
    #             q_pt, q_target_pt = cpd(self.q), cpd(self.q_target)
    #         else:
    #             patience_counter += 1  # Increment patience

    #         if patience_counter >= patience:
    #             break

    #     self.q, self.q_target = cpd(q_pt), cpd(q_target_pt)
    #     res = {"critic_loss_train": best_train_loss, "critic_loss_val": best_val_loss}
    #     return res

    def fit_restart(self, replay_buffer, n_steps):
        self.total_it = 0
        self.q, self.q_target = cpd(self.q_0), cpd(self.q_target_0)
        self.wipe_optimizer()

        td = Subset(replay_buffer, replay_buffer.train_indexes)
        vd = Subset(replay_buffer, replay_buffer.val_indexes)

        td = DataLoader(td, batch_size=min(self.batch_size, len(td)), shuffle=True)
        vd = DataLoader(vd, batch_size=min(self.batch_size, len(vd)))

        best_val_loss, best_train_loss = float("inf"), float("inf")
        best_q, best_q_target = None, None
        patience_counter = 0

        for i in range(1000):
            self.q.train()
            self.q_target.train()
            training_loss = sum(self.train(batch) for batch in td) / len(td)

            self.q.eval()
            self.q_target.eval()
            val_loss = sum(self.val(batch) for batch in vd) / len(vd)

            if val_loss < best_val_loss:
                best_val_loss, best_train_loss = val_loss, training_loss
                best_q, best_q_target = cpd(self.q), cpd(self.q_target)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break
        print(f"trained for {i} epochs with patience {self.patience}")

        self.q, self.q_target = cpd(best_q), cpd(best_q_target)
        return {"critic_loss_train": best_train_loss, "critic_loss_val": best_val_loss}

    def get_state_dict(self):
        state = {
            "critic": self.q.state_dict(),
            "critic_target": self.q_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state):
        self.q.load_state_dict(state["critic"])
        self.q_target.load_state_dict(state["critic_target"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def wipe_optimizer(self):
        self.critic_optimizer = self.OPT(
            self.q.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def set_train(self):
        self.q.train()
        self.q_target.train()
        self.training = True

    def set_eval(self):
        self.q.eval()
        self.q_target.eval()
        self.training = False

    @staticmethod
    def _freeze_layers_util(model, n_trainable_layers):
        # Get a list of all layers (named_parameters will give parameters with their names)
        layers = list(model.named_parameters())
        # Calculate the index from where we should stop freezing
        freeze_up_to = len(layers) - n_trainable_layers
        # Freeze layers up to the calculated index
        for i, (name, param) in enumerate(layers):
            if i < freeze_up_to:
                param.requires_grad = False  # Freeze
                print(f"Freezing layer: {name}")
            else:
                param.requires_grad = True  # Unfreeze
                print(f"Unfreezing layer: {name}")

    def freeze_layers(self, n_trainable_layers):
        self._freeze_layers_util(self.q, n_trainable_layers)
        self._freeze_layers_util(self.q_0, n_trainable_layers)

