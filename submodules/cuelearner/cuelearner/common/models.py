import torch
import torch.nn as nn
import torch.nn.functional as F

from cuelearner.common.utils import clip_norm_pytorch as clip_norm


"""
Assumptions:
the reward is not bounded
the action is bounded within a hypersphere
    (i.e. the norm of the action vector is between min_action and max_action)

"""


def initialize_weights(model, init_type="xavier"):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif init_type == "he":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif init_type == "lecun":
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity="sigmoid")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # Add other initializations as needed


class BaseMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(BaseMLP, self).__init__()

        # Input layer
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_sizes[0])])

        # Hidden layers
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)
    
    def get_feature_vector(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassificationMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(ClassificationMLP, self).__init__()

        # Input layer
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_sizes[0])])

        # Hidden layers
        layer_sizes = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output(x)
        # Logits to be used with nn.CrossEntropyLoss for classification
        return x

    def get_probabilities(self, x):
        """
        Computes the forward pass and returns the probabilities of each class.
        """
        # Apply softmax to convert logits to probabilities
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QNetwork(BaseMLP):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(QNetwork, self).__init__(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, states, actions):
        actions = clip_norm(actions, 1, 1)
        x = torch.cat([states, actions], dim=1)
        x = super().forward(x)
        return x
    
    def get_feature_vector(self, states, actions):
        actions = clip_norm(actions, 1, 1)
        x = torch.cat([states, actions], dim=1)
        x = super().get_feature_vector(x)
        return x


class PNetwork(BaseMLP):
    def __init__(self, state_dim, action_dim, hidden_sizes, min_action, max_action):
        super(PNetwork, self).__init__(state_dim, action_dim, hidden_sizes)

        self.min_action = min_action
        self.max_action = max_action

    def forward(self, states):
        x = super().forward(states)

        actions = clip_norm(x, self.min_action, self.max_action)
        return actions


class ActionOptimizerNetwork(BaseMLP):
    # (state, action) -> (slightly better action)
    def __init__(self, state_dim, action_dim, hidden_sizes, min_action, max_action):
        super(ActionOptimizerNetwork, self).__init__(
            state_dim + action_dim, action_dim, hidden_sizes
        )

        self.min_action = min_action
        self.max_action = max_action

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        action_residuals = super().forward(x)
        return action_residuals


class ActionOptimizerClassifier(ClassificationMLP):
    # (state, action) -> (slightly better action)
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(ActionOptimizerClassifier, self).__init__(
            state_dim + action_dim, 3, hidden_sizes
        )

    def forward(self, states, actions):
        # actions = clip_norm(actions)
        x = torch.cat([states, actions], dim=1)
        next_action_logits = super().forward(x)

        return next_action_logits


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes_actor,
        hidden_sizes_critic,
        min_action,
        max_action,
    ):
        super().__init__()
        # build policy and value functions
        self.pi = PNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes_actor,
            min_action=min_action,
            max_action=max_action,
        )

        self.q1 = QNetwork(state_dim, action_dim, hidden_sizes_critic)
        self.q2 = QNetwork(state_dim, action_dim, hidden_sizes_critic)


def uniform_2d(interval=1.0):
    # Sample angles uniformly between 0 and 360 degrees with the specified tolerance
    angles_deg = torch.arange(0, 360, interval)
    angles_rad = torch.deg2rad(angles_deg)
    # Convert angles to actions: each action is (cos(angle), sin(angle))
    sampled_actions = torch.stack((torch.cos(angles_rad), torch.sin(angles_rad)), dim=1)
    return sampled_actions


def q_map(q_net, state, bin_size):
    """
    Returns a tuple of actions, rewards
    """
    # sample actions
    state = torch.tensor(state, dtype=torch.float32)
    actions = uniform_2d(interval=bin_size).to("cuda")
    states = state.repeat(actions.shape[0], 1).to("cuda")

    with torch.no_grad():
        q_values = q_net(states, actions).flatten()
    return actions.detach().cpu().numpy(), q_values.detach().cpu().numpy()


def bradley_terry_loss(rewards_good, rewards_bad):
    # Calculate probabilities of A being preferred over B
    eps = 1e-8  # Small epsilon for numerical stability
    # Calculate probabilities of A being preferred over B
    prob_a_pref_over_b = torch.exp(rewards_good) / (
        torch.exp(rewards_good) + torch.exp(rewards_bad) + eps
    )
    target_preferences = torch.ones_like(rewards_good)
    # Compute binary cross-entropy loss
    loss = F.binary_cross_entropy(prob_a_pref_over_b, target_preferences)
    return loss
