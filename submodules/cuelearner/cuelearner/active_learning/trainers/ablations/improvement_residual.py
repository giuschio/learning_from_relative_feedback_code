import numpy as np
import torch

from copy import deepcopy as dcp

from torch.utils.tensorboard import SummaryWriter

from cuelearner.active_learning.utils import dump_hparams, eval_policy
from cuelearner.active_learning.utils import LinearUpdateScheduler as Scheduler
from cuelearner.common.data_structures import ReplayBuffer
from cuelearner.common.utils import CheckpointManager, LogDir

from cuelearner.common.utils import Logger
from cuelearner.common.utils import Timer
from cuelearner.common.utils import angle

from cuelearner.active_learning.policies import AdviceWrapper


class Trainer:
    def __init__(
        self,
        # training args
        max_steps,
        buffer_size,
        start_steps,
        eval_every,
        eval_steps,
        log_dir,
        # advice params
        advice_noise,
        adv_update_after,
        adv_update_every,
        # checkpointing
        advice_checkpoint_options,
        # logging
        verbose,
        print_every,
        seed,
        **kwargs,
    ) -> None:

        if len(kwargs) > 0:
            print("The following args were specified but are not used by the trainer:")
            [print(f"- {k}") for k in kwargs.keys()]
        self.max_steps = max_steps
        self.buffer_size = buffer_size
        self.start_steps = start_steps
        self.eval_steps = eval_steps
        self.eval_every = eval_every
        self.verbose = verbose
        self.print_every = print_every
        self.__rng = np.random.default_rng(seed)

        # --- ADVICE PARAMS
        self.advice_noise = advice_noise
        self.adv_update_after = adv_update_after
        self.adv_update_every = adv_update_every
        self.adv_grad_steps = adv_update_every
        # END ADVICE PARAMS

        self.log_dir = LogDir(log_dir, seed)
        self.__logger = Logger()
        self.__writer = SummaryWriter(self.log_dir.root_dir)
        self.__advice_ckpt_manager = CheckpointManager(
            self.log_dir.get_path("ckpt_advice"),
            mode="max",
            **advice_checkpoint_options,
        )

    def __log(self, key, value, epoch):
        if isinstance(value, list):
            value = np.array(value)

        if isinstance(value, np.ndarray):
            if len(value) == 0:
                return
            value = np.mean(value)

        self.__logger.add_scalar(key, value, epoch)
        self.__writer.add_scalar(key, value, epoch)

    def __log_dict(self, dictionary, epoch):
        for key, value in dictionary.items():
            self.__log(key, value, epoch)

    def __print_log(self):
        if self.verbose:
            print(self.__logger)

    def fit(self, policy, advice_cloner, train_env, val_env, expert):
        # save network hyperparameters
        dump_hparams(advice_cloner, self.log_dir.get_path("ckpt_advice/hparams.yaml"))

        # initialize replay buffer
        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        # Box action space
        # max_action = float(train_env.action_space.high[0])
        # spherical action space
        max_action = train_env.action_space.max_action
        min_action = train_env.action_space.min_action

        # --- ADVICE BUFFER
        advice_buffer = ReplayBuffer(
            device=advice_cloner.device,
            fields={
                "state": state_dim,
                "action": action_dim,
                "next_state": state_dim,
                "done": 1,
                "reward": 1,  # Assuming reward is a single value
                "improved_action": action_dim,
            },
            seed=self.__rng.integers(low=0, high=1_000_000).item(),
            max_size=self.buffer_size,
        )

        advice_environment = dcp(val_env)

        # END ADVICE BUFFER

        timer = Timer()
        timer.start("total")

        # --- BASELINE
        with torch.no_grad():
            initial_rewards = eval_policy(policy, dcp(val_env), steps=self.eval_steps)
        self.__log("val_reward", initial_rewards, 0)
        self.__log("train_reward", initial_rewards, 0)
        self.__log("step", 0, 0)
        self.__print_log()

        # END TRY TO COMPUTE UPPER BOUND

        scheduler_adv_training = Scheduler(self.adv_update_after, self.adv_update_every)
        scheduler_eval = Scheduler(self.eval_every, self.eval_every)
        scheduler_print = Scheduler(self.print_every, self.print_every)

        # --- TRAINING
        state, info = train_env.reset()
        done = False
        train_reward_p = []

        for step in range(1, self.max_steps):
            policy_action = policy.select_action(state)

            old_info = info
            next_state, reward, done, terminated, info = train_env.step(policy_action)
            train_reward_p.append(reward)

            # get the optimal action from the expert
            optimal_action = expert.select_action(state)

            # needed for numerical reasons (sometimes)
            optimal_action = (
                optimal_action
                if angle(policy_action, optimal_action)
                < advice_cloner.advice_search_width
                else policy_action
            )

            # ---- ADVICE END

            advice_buffer.add(
                state=state,
                action=policy_action,
                next_state=next_state,
                reward=reward,
                done=done,
                improved_action=optimal_action,
            )

            state = next_state

            if scheduler_adv_training(step):
                advice_cloner.set_train()
                res = advice_cloner.fit(advice_buffer, self.adv_grad_steps)
                self.__log_dict(res, step)
                self.__log("step", step, step)
                advice_cloner.set_eval()
                self.__log("train_reward", train_reward_p, step)
                train_reward_p = []

            advice_policy = AdviceWrapper(policy, advice_cloner)
            if scheduler_eval(step):
                with torch.no_grad():
                    final_rewards = eval_policy(
                        advice_policy, dcp(val_env), self.eval_steps
                    )
                self.__log("val_reward", final_rewards, step)
                self.__log("val_improvement", final_rewards - initial_rewards, step)
                self.__log("step", step, step)
                self.__log_dict(timer.get_all(), step)
                self.__advice_ckpt_manager.update(
                    state_dict=advice_cloner.get_state_dict(),
                    epoch=step,
                    metric=np.mean(final_rewards),
                    metric_name="val_reward",
                )

            if scheduler_print(step):
                self.__print_log()

            if done:
                # Reset environment
                state, info = train_env.reset()
                done = False
        self.__writer.close()
