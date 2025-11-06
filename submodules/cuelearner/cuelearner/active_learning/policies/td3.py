import copy
import itertools
import inspect
import numpy as np
import os
import torch
import torch.nn.functional as F
import yaml

from copy import deepcopy as cpd
from torch.nn.utils import clip_grad_norm_ as clip_grad
from cuelearner.common.models import MLPActorCritic
from cuelearner.common.utils import clip_norm


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    # MIT License

    # Copyright (c) 2020 Scott Fujimoto

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    policy_type = "td3"

    def __init__(
        self,
        state_dim,
        action_dim,
        min_action,
        max_action,
        batch_size,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=1,
        hidden_sizes_actor=[400, 300],
        hidden_sizes_critic=[400, 300],
        optimizer_critic="sgd",
        learning_rate_critic=1e-3,
        weight_decay_critic=0.0,
        gradient_clipping_critic=0.0,
        optimizer_policy="sgd",
        learning_rate_policy=1e-3,
        weight_decay_policy=0.0,
        gradient_clipping_policy=0.0,
        fit_policy="fixed",
        seed=None,
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
        self.ac = MLPActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes_actor=hidden_sizes_actor,
            hidden_sizes_critic=hidden_sizes_critic,
            min_action=min_action,
            max_action=max_action,
        ).to(self.device)
        # ---
        self.ac_target = cpd(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_target.parameters():
            p.requires_grad = False

        q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        p_params = self.ac.pi.parameters()
        AVAILABLE_OPTIMIZERS = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)
        self.OPT_CRITIC = AVAILABLE_OPTIMIZERS[optimizer_critic]
        self.OPT_POLICY = AVAILABLE_OPTIMIZERS[optimizer_policy]

        self.learning_rate_critic = learning_rate_critic
        self.weight_decay_critic = weight_decay_critic
        self.gradient_clipping_critic = gradient_clipping_critic

        self.critic_optimizer = self.OPT_CRITIC(
            q_params,
            lr=self.learning_rate_critic,
            weight_decay=self.weight_decay_critic,
        )

        self.learning_rate_policy = learning_rate_policy
        self.weight_decay_policy = weight_decay_policy
        self.gradient_clipping_policy = gradient_clipping_policy
        self.actor_optimizer = self.OPT_POLICY(
            p_params,
            lr=self.learning_rate_policy,
            weight_decay=self.weight_decay_policy,
        )

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.to_torch = lambda x: torch.tensor(
            x, dtype=torch.float32, device=self.device
        )
        self.batch_size = batch_size
        self.fit = dict(
            fixed=self.fit_fixed,
        )[fit_policy]

    def select_action_train(self, state):
        return self.select_action(state)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.ac.pi(state).cpu().numpy().flatten()
        return action

    def get_q_estimate(self, state, action):
        # numpy arrays
        with torch.no_grad():
            state = self.to_torch(state).unsqueeze(0)
            action = self.to_torch(action).unsqueeze(0)

            q_estimate = self.ac.q1(state, action).squeeze(0)
        return q_estimate.item()

    def __get_params(self, frame) -> dict:
        args, _, _, values = inspect.getargvalues(frame)
        values.pop("self")
        values.pop("kwargs")
        return values

    @property
    def hparams(self):
        return {"policy_type": self.policy_type, "hparams": self.__hparams}

    def critic_loss(self, batch):
        # Sample replay buffer
        state = batch["state"]
        action = batch["action"]
        next_state = batch["next_state"]
        reward = batch["reward"]
        done = batch["done"]

        with torch.no_grad():
            if self.discount > 0.0:
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn(
                        action.size(), dtype=action.dtype, generator=self.__rng
                    ).to(action.device)
                    * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                # Box action space
                # next_action = (self.ac_target.pi(next_state) + noise).clamp(
                #     -self.max_action, self.max_action
                # )
                # spherical action space
                next_action = clip_norm(
                    self.ac_target.pi(next_state) + noise,
                    min_norm=self.min_action,
                    max_norm=self.max_action,
                )
                target_Q1 = self.ac_target.q1(next_state, next_action)
                target_Q2 = self.ac_target.q2(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1.0 - done) * self.discount * target_Q
            else:
                target_Q = reward

        # Get current Q estimates
        current_Q1 = self.ac.q1(state, action)
        current_Q2 = self.ac.q2(state, action)

        # Compute critic loss
        L = F.mse_loss
        critic_loss = 0.5 * (L(current_Q1, target_Q) + L(current_Q2, target_Q))
        return critic_loss

    def actor_loss(self, batch):
        actor_loss = -self.ac.q1(batch["state"], self.ac.pi(batch["state"])).mean()
        return actor_loss

    def train(self, batch):
        """
        Performs one gradient step
        If it is time, performs target update

        """
        self.total_it += 1
        critic_loss = self.critic_loss(batch)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.gradient_clipping_critic > 0:
            clip_grad(self.ac.q1.parameters(), max_norm=self.gradient_clipping_critic)
            clip_grad(self.ac.q2.parameters(), max_norm=self.gradient_clipping_critic)
        self.critic_optimizer.step()

        # Delayed policy updates
        # Compute actor loss ayways for logging purposes
        actor_loss = self.actor_loss(batch)
        if self.total_it % self.policy_freq == 0:
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.gradient_clipping_policy > 0:
                clip_grad(
                    self.ac.pi.parameters(), max_norm=self.gradient_clipping_policy
                )
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.ac.parameters(), self.ac_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        return critic_loss.item(), actor_loss.item()

    def val(self, batch):
        critic_loss = self.critic_loss(batch)
        actor_loss = self.actor_loss(batch)
        return critic_loss, actor_loss

    def fit_fixed(self, replay_buffer, n_steps):
        train_losses_critic = np.zeros(n_steps)
        train_losses_actor = np.zeros(n_steps)
        for i in range(n_steps):
            batch = replay_buffer.sample(self.batch_size, "train")
            critic_loss, actor_loss = self.train(batch)
            train_losses_critic[i] = critic_loss
            train_losses_actor[i] = actor_loss

        val_losses_critic = np.zeros(n_steps)
        val_losses_actor = np.zeros(n_steps)
        with torch.no_grad():
            for i in range(n_steps):
                batch = replay_buffer.sample(self.batch_size, "val")
                critic_loss, actor_loss = self.val(batch)
                val_losses_critic[i] = critic_loss
                val_losses_actor[i] = actor_loss

        res = {
            "critic_loss_train": np.mean(train_losses_critic),
            "critic_loss_val": np.mean(val_losses_critic),
            "actor_loss_train": np.mean(train_losses_actor),
            "actor_loss_val": np.mean(train_losses_critic),
        }

        return res

    def get_state_dict(self):
        state = {
            "ac": self.ac.state_dict(),
            "ac_target": self.ac_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state):
        self.ac.load_state_dict(state["ac"])
        self.ac_target.load_state_dict(state["ac_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def set_train(self):
        self.ac.train()
        self.ac_target.train()

    def set_eval(self):
        self.ac.eval()
        self.ac_target.eval()

    @property
    def q(self):
        return self.ac.q1
