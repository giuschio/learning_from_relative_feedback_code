import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from cuelearner.active_learning.utils import eval_policy, eval_random
from cuelearner.active_learning.utils import dump_hparams
from cuelearner.active_learning.utils import LinearUpdateScheduler as Scheduler
from cuelearner.common.data_structures import ReplayBuffer
from cuelearner.common.utils import CheckpointManager, LogDir

from cuelearner.common.utils import Logger
from cuelearner.common.utils import clip_norm
from cuelearner.common.utils import Timer
from copy import deepcopy as dcp


def combine_unit_vectors(u, v):
    """
    Combines two 2D unit vectors by summing their rotations.

    Parameters:
    - u: Tuple representing the first unit vector (u_x, u_y).
    - v: Tuple representing the second unit vector (v_x, v_y).

    Returns:
    - A tuple representing the resulting unit vector after summing the rotations.
    """
    # Calculate angles of each vector
    theta_u = np.arctan2(u[1], u[0])
    theta_v = np.arctan2(v[1], v[0])
    
    # Sum the angles
    theta_combined = theta_u + theta_v
    
    # Convert the resulting angle back to a unit vector
    combined_vector = (np.cos(theta_combined), np.sin(theta_combined))
    
    return combined_vector

class AdviceWrapper:
    def __init__(self, base_policy, advice_policy) -> None:
        self.base_policy = base_policy
        self.advice_policy = advice_policy

    def select_action(self, state):
        action = self.base_policy.select_action(state)
        action_res = self.advice_policy.select_action(np.concatenate((state, action)))
        return combine_unit_vectors(action, action_res)

class Trainer:
    def __init__(
        self,
        # training args
        max_steps,
        buffer_size,
        start_steps,
        update_after,
        update_every,
        act_noise,
        eval_every,
        eval_steps,
        log_dir,
        grad_steps,
        # checkpointing
        policy_checkpoint_options,
        verbose,
        print_every,
        seed,
    ) -> None:
        self.max_steps = max_steps
        self.buffer_size = buffer_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.grad_steps = grad_steps if grad_steps > 0 else update_every
        self.act_noise = act_noise
        self.eval_steps = eval_steps
        self.eval_every = eval_every
        self.verbose = verbose
        self.print_every = print_every
        self.__rng = np.random.default_rng(seed)

        self.log_dir = LogDir(log_dir, seed)
        self.__logger = Logger()
        self.__writer = SummaryWriter(self.log_dir.root_dir)
        self.__policy_ckpt_manager = CheckpointManager(
            self.log_dir.get_path("ckpt_policy"),
            mode="max",
            **policy_checkpoint_options,
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

    def fit(self, policy, residual_policy, train_env, val_env):
        # save network hyperparameters
        dump_hparams(residual_policy, self.log_dir.get_path("ckpt_policy/hparams.yaml"))

        # initialize replay buffer
        state_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        # Box action space
        # max_action = float(train_env.action_space.high[0])
        # spherical action space
        max_action = train_env.action_space.max_action
        min_action = train_env.action_space.min_action

        replay_buffer = ReplayBuffer(
            device=policy.device,
            fields={
                "state": state_dim + 2,
                "action": action_dim,
                "next_state": state_dim,
                "done": 1,
                "reward": 1,  # Assuming reward is a single value
            },
            seed=self.__rng.integers(low=0, high=1_000_000).item(),
            max_size=self.buffer_size,
        )

        # --- BASELINE
        # random_reward = eval_random(val_env)
        timer = Timer()
        timer.start("total")
        scheduler_training = Scheduler(self.update_after, self.update_every)
        scheduler_eval = Scheduler(self.eval_every, self.eval_every)
        scheduler_print = Scheduler(self.print_every, self.print_every)

        # --- EVALUATE INITIAL POLICY
        policy.set_eval()
        timer.start("evaluation")
        with torch.no_grad():
            rews = eval_policy(policy, dcp(val_env), steps=self.eval_steps)
        rew_avg = np.mean(rews)
        timer.stop("evaluation")
        self.__log("step", 0, 0)
        self.__log("val_reward", rew_avg, 0)
        self.__print_log()
        # --- END EVALUATE INITIAL POLICY

        # --- TRAINING
        state, info = train_env.reset()
        done = False
        rews_train_p = []
        for step in range(1, self.max_steps):
            # Select action randomly or according to policy
            timer.start("action_choice")
            if step < self.start_steps:
                action = train_env.action_space.sample()
            else:
                action = policy.select_action_train(state)
                residual_action = residual_policy.select_action_train(np.concatenate((state, action)))
                action = combine_unit_vectors(action, residual_action)
                noise = self.__rng.normal(0, self.act_noise, size=action_dim)

                # Box action space
                # action = (action + noise).clip(-max_action, max_action)
                # spherical action space
                # TODO: this clipping is a bit weird and does not comply with normal clipping
                #       in the future, I should probably just pretend and clip each element, and then let the
                #       environment figure it out
                action = clip_norm(action + noise, min_action, max_action)
            timer.stop("action_choice")
            timer.start("simulator")
            next_state, reward, done, terminated, info = train_env.step(action)
            timer.stop("simulator")
            replay_buffer.add(
                state=np.concatenate((state, action)),
                action=residual_action,
                next_state=next_state,
                reward=reward,
                done=done,
            )
            state = next_state
            rews_train_p.append(reward)

            if scheduler_training(step):
                timer.start("training")
                residual_policy.set_train()
                res = residual_policy.fit(replay_buffer, self.grad_steps)
                timer.stop("training")
                self.__log_dict(res, step)
                residual_policy.set_eval()
                self.__log("step", step, step)
                self.__log_dict(timer.get_all(), step)

            if scheduler_eval(step):
                policy.set_eval()
                residual_policy.set_eval()
                wrapper_policy = AdviceWrapper(policy, residual_policy)
                timer.start("evaluation")
                with torch.no_grad():
                    rews = eval_policy(wrapper_policy, dcp(val_env), steps=self.eval_steps)
                rew_avg = np.mean(rews)
                timer.stop("evaluation")
                self.__log("step", step, step)
                # self.__log("random_policy_reward", random_reward, step)
                self.__log("val_reward", rew_avg, step)
                self.__log("train_reward", rews_train_p, step)
                self.__log_dict(timer.get_all(), step)
                rews_train_p = []

                # Save models
                self.__policy_ckpt_manager.update(
                    state_dict=residual_policy.get_state_dict(),
                    epoch=step,
                    metric=rew_avg,
                    metric_name="val_reward",
                )
            if scheduler_print(step):
                self.__print_log()

            if done:
                # Reset environment
                state, info = train_env.reset()
                done = False
        self.__writer.close()
