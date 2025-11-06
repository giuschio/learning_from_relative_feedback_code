import inspect
import numpy as np
import os
import torch
import torch.nn.functional as F
import yaml


from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.utils import clip_grad_norm_ as clip_grad

from copy import deepcopy as cpd
from cuelearner.common.models import PNetwork
from cuelearner.common.utils import angle, signed_angle, rotate_vector


import random


class BehaviorCloner(object):
    policy_type = "behavior_cloner"

    def __init__(
        self,
        state_dim,
        action_dim,
        min_action,
        max_action,
        batch_size,
        hidden_sizes_behavior_cloner,
        optimizer,
        learning_rate,
        weight_decay,
        gradient_clipping,
        fit_policy,
        patience,
        seed=None,
        **kwargs,
    ):
        self.__hparams = self.__get_params(inspect.currentframe())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(kwargs) > 0:
            print(
                "The following args were specified but are not used by the BC policy:"
            )
            [print(f"- {k}") for k in kwargs.keys()]

        # --- initialize network given a seed
        torch.manual_seed(seed)
        self.bc_network = PNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes_behavior_cloner,
            min_action=min_action,
            max_action=max_action,
        ).to(self.device)

        self.OPT = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)[optimizer]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.optimizer = self.OPT(
            self.bc_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.n_0 = cpd(self.bc_network)

        self.total_it = 0
        self.batch_size = batch_size
        # self.fit = dict(fixed=self.fit_fixed, restarts=self.fit_restart)[fit_policy]
        self.fit = self.fit_restart
        self.patience = patience
        self.__rng = np.random.default_rng(seed)
        # needed from the trainer to know that this has the same action
        # space as the policy
        self.is_privileged = False

    def select_action(self, state, action=None):
        # add None action so that it complies with the same API as the advice cloner
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.bc_network(state)
        return action.cpu().numpy().flatten()
    
    def select_action_batch(self, state, action=None):
        return np.array([self.select_action(s) for s in state])

    def __get_params(self, frame) -> dict:
        args, _, _, values = inspect.getargvalues(frame)
        values.pop("self")
        values.pop("kwargs")
        return values

    @property
    def hparams(self):
        return {"policy_type": self.policy_type, "hparams": self.__hparams}

    def loss(self, batch):
        # Sample replay buffer
        state = batch["state"]
        optimal_action = batch["optimal_action"]

        # Train the action optimizer
        L = F.mse_loss
        predicted_action = self.bc_network(state)
        loss = L(predicted_action, optimal_action)
        return loss

    def train(self, batch):
        """
        Performs one gradient step
        If it is time, performs target update

        """
        self.total_it += 1
        loss = self.loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clipping > 0:
            clip_grad(
                self.bc_network.parameters(), max_norm=self.gradient_clipping
            )
        self.optimizer.step()
        return loss.item()

    def val(self, batch):
        loss = self.loss(batch)
        return loss.item()

    # def fit_fixed(self, replay_buffer, n_steps):
    #     train_losses = [
    #         self.train(replay_buffer.sample(self.batch_size, "train"))
    #         for _ in range(n_steps)
    #     ]
    #     val_losses = [
    #         self.val(replay_buffer.sample(self.batch_size, "val"))
    #         for _ in range(n_steps)
    #     ]

    #     return {
    #         "advice_loss_train": np.mean(train_losses),
    #         "advice_loss_val": np.mean(val_losses),
    #     }

    def fit_restart(self, replay_buffer, n_steps):
        self.bc_network = cpd(self.n_0)
        self.optimizer = self.OPT(
            self.bc_network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        td = Subset(replay_buffer, replay_buffer.train_indexes)
        vd = Subset(replay_buffer, replay_buffer.val_indexes)

        td = DataLoader(td, batch_size=min(self.batch_size, len(td)), shuffle=True)
        vd = DataLoader(vd, batch_size=min(self.batch_size, len(vd)))

        patience_counter, best_val_loss = 0, float("inf")
        best_model = cpd(self.bc_network)

        for idx in range(1000):  # Limit epochs
            self.bc_network.train()
            training_loss = sum(self.train(batch) for batch in td) / len(td)
            self.bc_network.eval()
            val_loss = sum(self.val(batch) for batch in vd) / len(vd)

            if val_loss < best_val_loss:
                best_val_loss, best_train_loss = val_loss, training_loss
                best_model = cpd(self.bc_network)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        self.bc_network = cpd(best_model)
        return {
            "epochs": idx,
            "advice_cloner_train_loss": best_train_loss,
            "advice_cloner_val_loss": best_val_loss,
        }

    def get_state_dict(self):
        state = {
            "action_optimizer": self.bc_network.state_dict(),
            "action_optimizer_optimizer": self.optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state):
        self.bc_network.load_state_dict(state["action_optimizer"])
        self.optimizer.load_state_dict(state["action_optimizer_optimizer"])

    def set_train(self):
        self.bc_network.train()

    def set_eval(self):
        self.bc_network.eval()
