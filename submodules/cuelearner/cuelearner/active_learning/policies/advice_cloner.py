import inspect
import numpy as np
import os
import torch
import torch.nn.functional as F
import yaml


from torch.utils.data import DataLoader, random_split, Subset
from torch.nn.utils import clip_grad_norm_ as clip_grad

from copy import deepcopy as cpd
from cuelearner.common.models import ActionOptimizerNetwork
from cuelearner.common.utils import angle, signed_angle, rotate_vector


import random


class AdviceCloner(object):
    policy_type = "advice_cloner"

    def __init__(
        self,
        state_dim,
        action_dim,
        min_action,
        max_action,
        batch_size,
        hidden_sizes_advice_cloner,
        advice_step_size,
        advice_max_steps,
        advice_termination_threshold,
        advice_search_width,
        # advice_training_sampling,
        residual_scaling_factor,
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
                "The following args were specified but are not used by the advice policy:"
            )
            [print(f"- {k}") for k in kwargs.keys()]
        self.advice_termination_threshold = advice_termination_threshold
        self.advice_step_size = advice_step_size
        self.advice_max_steps = advice_max_steps
        self.advice_search_width = advice_search_width
        self.residual_scaling_factor = residual_scaling_factor

        # --- initialize network given a seed
        torch.manual_seed(seed)
        self.action_optimizer = ActionOptimizerNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes_advice_cloner,
            min_action=min_action,
            max_action=max_action,
        ).to(self.device)

        self.OPT = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)[optimizer]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
        self.optimizer = self.OPT(
            self.action_optimizer.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.n_0 = cpd(self.action_optimizer)

        self.total_it = 0
        self.batch_size = batch_size
        self.fit = dict(fixed=self.fit_fixed, restarts=self.fit_restart)[fit_policy]
        self.patience = patience
        # self.advice_training_sampling = advice_training_sampling
        self.__rng = np.random.default_rng(seed)
        # needed from the trainer to know that this has the same action
        # space as the policy
        self.is_privileged = False

    def select_action(self, state, action):
        max_advice_its = self.advice_max_steps
        max_advice_its = 0 if max_advice_its < 0 else max_advice_its

        left_bound = rotate_vector(action, self.advice_search_width)
        right_bound = rotate_vector(action, -self.advice_search_width)

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        action_0 = action.clone()
        model = self.action_optimizer
        with torch.no_grad():
            for idx in range(max_advice_its):
                residual = model(state, action) / self.residual_scaling_factor
                new_action = action + residual
                new_action = new_action / torch.norm(new_action, 2)
                # walls at bounds
                if signed_angle(action_0, new_action) > self.advice_search_width:
                    new_action = torch.FloatTensor(left_bound.reshape(1, -1)).to(
                        self.device
                    )
                elif signed_angle(action_0, new_action) < -self.advice_search_width:
                    new_action = torch.FloatTensor(right_bound.reshape(1, -1)).to(
                        self.device
                    )

                if angle(new_action, action) < self.advice_termination_threshold:
                    action = new_action
                    break
                else:
                    action = new_action
                # not having early termination seems to work a bit better (or the same)
                # in the adaptation scenario. Could get rid of another magic number/tunable param
                # action = new_action

        return action.cpu().numpy().flatten()
    
    def select_action_batch(self, state, action):
        return np.array([self.select_action(s, a) for (s, a) in zip(state, action)])

    # def select_action_train(self, state, action):
    #     max_advice_its = self.advice_max_steps
    #     max_advice_its = 0 if max_advice_its < 0 else max_advice_its

    #     if self.advice_training_sampling == "dagger":
    #         max_advice_its = self.__rng.integers(0, max_advice_its + 1)

    #     left_bound = rotate_vector(action, self.advice_search_width)
    #     right_bound = rotate_vector(action, -self.advice_search_width)

    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
    #     action_0 = action.clone()
    #     model = self.action_optimizer
    #     with torch.no_grad():
    #         for idx in range(max_advice_its):
    #             residual = model(state, action) / self.residual_scaling_factor
    #             new_action = action + residual
    #             new_action = new_action / torch.norm(new_action, 2)
    #             # walls at bounds
    #             if signed_angle(action_0, new_action) > self.advice_search_width:
    #                 new_action = torch.FloatTensor(left_bound.reshape(1, -1)).to(
    #                     self.device
    #                 )
    #             elif signed_angle(action_0, new_action) < -self.advice_search_width:
    #                 new_action = torch.FloatTensor(right_bound.reshape(1, -1)).to(
    #                     self.device
    #                 )

    #             if angle(new_action, action) < self.advice_termination_threshold:
    #                 action = new_action
    #                 break
    #             else:
    #                 action = new_action

    #     return action.cpu().numpy().flatten()

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
        action = batch["action"]
        next_state = batch["next_state"]
        reward = batch["reward"]
        done = batch["done"]
        improved_action = batch["improved_action"]

        # Train the action optimizer
        L = F.mse_loss
        model = self.action_optimizer

        residuals_gt = (improved_action - action) * self.residual_scaling_factor
        residuals_pred = model(state, action)
        loss = L(residuals_pred, residuals_gt)
        return loss

    def train(self, batch):
        """
        Performs one gradient step
        If it is time, performs target update

        """
        self.total_it += 1
        real_batch_size = batch["state"].shape[0]
        # prevent undue influence on training of small batches
        loss = self.loss(batch) * real_batch_size / self.batch_size
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clipping > 0:
            clip_grad(
                self.action_optimizer.parameters(), max_norm=self.gradient_clipping
            )
        self.optimizer.step()
        return loss.item() * real_batch_size

    def val(self, batch):
        real_batch_size = batch["state"].shape[0]
        loss = self.loss(batch)
        return loss.item() * real_batch_size

    def fit_fixed(self, replay_buffer, n_steps):
        train_losses = [
            self.train(replay_buffer.sample(self.batch_size, "train"))
            for _ in range(n_steps)
        ]
        val_losses = [
            self.val(replay_buffer.sample(self.batch_size, "val"))
            for _ in range(n_steps)
        ]

        return {
            "advice_loss_train": np.mean(train_losses),
            "advice_loss_val": np.mean(val_losses),
        }

    def fit_restart(self, replay_buffer, n_steps):
        self.action_optimizer = cpd(self.n_0)
        self.optimizer = self.OPT(
            self.action_optimizer.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        td = Subset(replay_buffer, replay_buffer.train_indexes)
        vd = Subset(replay_buffer, replay_buffer.val_indexes)
        len_training_data = len(td)
        len_validation_data = len(vd)

        td = DataLoader(td, batch_size=min(self.batch_size, len(td)), shuffle=True)
        vd = DataLoader(vd, batch_size=min(self.batch_size, len(vd)))

        patience_counter, best_val_loss = 0, float("inf")
        best_model = cpd(self.action_optimizer)

        for idx in range(10000):  # Limit epochs was 10K at some point
            self.action_optimizer.train()
            training_loss = sum(self.train(batch) for batch in td) / len_training_data
            self.action_optimizer.eval()
            val_loss = sum(self.val(batch) for batch in vd) / len_validation_data

            if val_loss < best_val_loss:
                best_val_loss, best_train_loss = val_loss, training_loss
                best_model = cpd(self.action_optimizer)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        self.action_optimizer = cpd(best_model)
        return {
            "epochs": idx,
            "advice_cloner_train_loss": best_train_loss,
            "advice_cloner_val_loss": best_val_loss,
        }

    # def fit_restart(self, replay_buffer, n_steps):

    #     self.action_optimizer = cpd(self.n_0)
    #     self.optimizer = self.OPT(
    #         self.action_optimizer.parameters(),
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay,
    #     )

    #     td = Subset(replay_buffer, replay_buffer.train_indexes)
    #     vd = Subset(replay_buffer, replay_buffer.val_indexes)

    #     train_loader = DataLoader(
    #         td, batch_size=min(self.batch_size, len(td)), shuffle=True
    #     )
    #     val_loader = DataLoader(vd, batch_size=min(self.batch_size, len(vd)))

    #     patience, patience_counter = 10, 0
    #     best_val_loss = float("inf")
    #     best_train_loss = float("inf")

    #     n_pt = None
    #     for epoch in range(1000):
    #         self.action_optimizer.train()
    #         training_loss = 0.0
    #         for idx, batch in enumerate(train_loader):
    #             training_loss += self.train(batch)
    #         training_loss = training_loss / (idx + 1)

    #         val_loss = 0.0
    #         self.action_optimizer.eval()
    #         with torch.no_grad():  # No gradients needed
    #             for idx, batch in enumerate(val_loader):
    #                 val_loss += self.val(batch)

    #         val_loss = val_loss / (idx + 1)
    #         # Early stopping check
    #         if val_loss < best_val_loss:
    #             patience_counter = 0
    #             best_val_loss = val_loss
    #             best_train_loss = training_loss

    #             n_pt = cpd(self.action_optimizer)
    #         else:
    #             patience_counter += 1  # Increment patience

    #         if patience_counter >= patience:
    #             break

    #     self.action_optimizer = cpd(n_pt)
    #     res = {
    #         "advice_cloner_train_loss": best_train_loss,
    #         "advice_cloner_val_loss": best_val_loss,
    #     }
    #     return res

    def get_state_dict(self):
        state = {
            "action_optimizer": self.action_optimizer.state_dict(),
            "action_optimizer_optimizer": self.optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state):
        self.action_optimizer.load_state_dict(state["action_optimizer"])
        self.optimizer.load_state_dict(state["action_optimizer_optimizer"])

    def set_train(self):
        self.action_optimizer.train()

    def set_eval(self):
        self.action_optimizer.eval()

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
        self._freeze_layers_util(self.action_optimizer, n_trainable_layers)
        self._freeze_layers_util(self.n_0, n_trainable_layers)
