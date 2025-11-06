import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from cuelearner.active_learning.utils import eval_policy, eval_random
from cuelearner.active_learning.utils import dump_hparams
from cuelearner.active_learning.utils import LinearUpdateScheduler as Scheduler
from cuelearner.common.data_structures import ReplayBuffer, NStepReplayBuffer
from cuelearner.common.utils import CheckpointManager, LogDir, rotate_vector

from cuelearner.common.utils import Logger
from cuelearner.common.utils import clip_norm
from cuelearner.common.utils import Timer
from collections import deque

def get_user_choice():
    dictionary = {0: 0.0, 1: 1.0, 2: -1.0}
    while True:
        try:
            choice = int(
                input(
                    "Choose improvement: \n  -(0) Heading is optimal \n  -(1) Counterclockwise \n  -(2) Clockwise\n  -(3) Don't know\nchoice: "
                )
            )
            if choice in [0, 1, 2]:
                return dictionary[choice]
            else:
                return None
        except ValueError:
            print("Invalid input. Please enter a numeric value (1 or 2).")


def eval_nav_policy(env, policy, n_steps):
    step_rewards = []
    episode_success = []
    episode_collision = []

    mean = lambda arr: sum(arr) / (len(arr) + 1e-7)

    state, info = env.reset()
    done = False
    n_envs = len(state)
    for _ in range(0, n_steps, n_envs):
        action = policy.select_action_batch(state)
        state, reward, done, terminated, info = env.step(action)
        [step_rewards.append(float(r)) for r in reward]

        for index in range(n_envs):
            if done[index]:
                success = float(env.unwrapped.termination_manager.get_term('goal_reached')[index])
                collision = float(env.unwrapped.termination_manager.get_term('base_contact')[index])

                episode_success.append(float(success))
                episode_collision.append(float(collision))
    res = dict(
        val_reward=mean(step_rewards),
        val_success=mean(episode_success),
        val_collision=mean(episode_collision)
    )
    return res
    


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
        # advice params
        advice_scheduler,  # the advice scheduler will need to go away and be substituted with a heuristic based on uncertainty
        advice_noise,
        advice_q_threshold,
        advice_probability,
        advice_usage_limit,
        adv_update_after,
        adv_update_every,
        # checkpointing
        policy_checkpoint_options,
        advice_checkpoint_options,
        # logging
        verbose,
        print_every,
        seed,
        save_replay_buffer_every=100_000,
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
        self.save_replay_buffer_every = save_replay_buffer_every

        # --- ADVICE PARAMS
        self.advice_scheduler = get_advice_scheduler(
            advice_scheduler.type, advice_scheduler.args
        )
        self.advice_noise = advice_noise
        self.advice_q_threshold = advice_q_threshold
        self.advice_probability = advice_probability
        self.advice_usage_limit = (
            advice_usage_limit if advice_usage_limit > 0 else self.max_steps
        )
        self.adv_update_after = adv_update_after
        self.adv_update_every = adv_update_every
        self.adv_grad_steps = adv_update_every
        # END ADVICE PARAMS

        self.log_dir = LogDir(log_dir, seed)
        self.__logger = Logger()
        self.__writer = SummaryWriter(self.log_dir.root_dir)
        self.__policy_ckpt_manager = CheckpointManager(
            self.log_dir.get_path("ckpt_policy"),
            mode="max",
            **policy_checkpoint_options,
        )
        self.__advice_ckpt_manager = CheckpointManager(
            self.log_dir.get_path("ckpt_advice"),
            mode="none",
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

    def fit(self, policy, advice_agent, train_env, val_env, expert, replay_buffer_checkpoint=None):
        # save network hyperparameters
        dump_hparams(policy, self.log_dir.get_path("ckpt_policy/hparams.yaml"))
        advice_agent.set_eval()

        # initialize replay buffer
        # state_dim = train_env.observation_space.shape[0]
        state_dim = train_env.state_dim
        action_dim = train_env.action_space.shape[0]
        # Box action space
        # max_action = float(train_env.action_space.high[0])
        # spherical action space
        max_action = train_env.action_space.max_action
        min_action = train_env.action_space.min_action

        if replay_buffer_checkpoint is None:
            replay_buffer = ReplayBuffer(
                device=policy.device,
                fields={
                    "state": state_dim,
                    "action": action_dim,
                    "next_state": state_dim,
                    "done": 1,
                    "reward": 1,  # Assuming reward is a single value
                    "done_n": 1,
                    "next_state_n": state_dim,
                    "reward_n": 1,
                    "lookahead": 1,
                },
                seed=self.__rng.integers(low=0, high=1_000_000).item(),
                max_size=self.buffer_size,
            )
        else:
            replay_buffer = replay_buffer_checkpoint

        # --- BASELINE
        # random_reward = eval_random(val_env)
        random_reward = 0.
        timer = Timer()
        timer.start("total")
        scheduler_training = Scheduler(self.update_after, self.update_every)
        scheduler_eval = Scheduler(self.eval_every, self.eval_every)
        scheduler_print = Scheduler(self.print_every, self.print_every)
        scheduler_save_replay_buffer = Scheduler(self.save_replay_buffer_every, self.save_replay_buffer_every)

        print("Start training")
        # --- TRAINING
        advices_used = 0
        adv_step = 0
        # advice_is_trained = False
        state, info = train_env.reset()
        # done = False
        rews_train_p = []
        # use_advice = True
        success_train_p = []
        collision_train_p = []

        num_envs = train_env.num_envs
        done = [False] * num_envs
        use_advice = [False] * num_envs
        for index in range(num_envs):
            ua = self.__rng.uniform(0, 1) < self.advice_probability
            use_advice[index] = ua

        nstep_wrapper = NStepReplayBuffer(replay_buffer, num_envs=num_envs, n_step=30, gamma=policy.discount)
        lookahead = np.array([30.] * num_envs)
        for step in range(1, self.max_steps, num_envs):
            # Select action randomly or according to policy
            timer.start("action_choice")
            if step < self.start_steps:
                # get action at random
                action = train_env.action_space.sample(num_envs=num_envs)
            else:
                # get action from policy
                action = policy.select_action_train_batch(state)
                timer.start("action_improvement")
                for index in range(num_envs):
                    if use_advice[index]:
                        if advice_agent.is_privileged:
                            advised_action = advice_agent.select_action(train_env.get_navigation_observation(index))
                        else:
                            advised_action = advice_agent.select_action(state[index], action[index])
                        action[index] = advised_action
                        advices_used += 1
                timer.stop("action_improvement")
            timer.stop("action_choice")
            timer.start("simulator")
            old_info = info

            next_state, reward, done, terminated, info = train_env.step(action)
            timer.stop("simulator")

            transition = {'state': state,
                          'next_state': next_state,
                          'action': action,
                          'reward': reward,
                          'done': done,
                          'lookahead': lookahead,
                          'env_id': list(range(num_envs))}
            nstep_wrapper.add_batch(transition)

            state = next_state
            [rews_train_p.append(float(r)) for r in reward]

            if scheduler_training(step):
                print("training the policy")
                timer.start("training")
                policy.set_train()
                res = policy.fit(replay_buffer, self.grad_steps)
                timer.stop("training")
                self.__log_dict(res, step)
                policy.set_eval()
                self.__log("step", step, step)
                self.__log("advices_given", adv_step, step)
                self.__log("advices_used", advices_used, step)
                self.__log_dict(timer.get_all(), step)

            if scheduler_eval(step):
                self.__log("step", step, step)
                self.__log("random_policy_reward", random_reward, step)
                avg_train_reward = np.mean(np.array(rews_train_p))
                self.__log("train_reward", avg_train_reward, step)
                self.__log("train_success", success_train_p, step)
                self.__log("train_collision", collision_train_p, step)
                val_res = eval_nav_policy(train_env, policy, self.eval_steps)
                self.__log_dict(val_res, step)
                self.__log_dict(timer.get_all(), step)
                rews_train_p = []
                success_train_p = []
                collision_train_p = []

                # Save models
                self.__policy_ckpt_manager.update(
                    state_dict=policy.get_state_dict(),
                    epoch=step,
                    metric=avg_train_reward,
                    metric_name="train_reward",
                )
            if scheduler_print(step):
                self.__print_log()

            if scheduler_save_replay_buffer(step):
                print("Saving replay buffer...")
                replay_buffer.to_file(self.log_dir.get_path("ckpt_buffer/buffer.ckpt"))
                print("Finished saving replay buffer")

            for index in range(num_envs):
                if done[index]:
                    ua = (
                    (self.__rng.uniform(0, 1) < self.advice_probability)
                    and (step < self.advice_usage_limit))
                    use_advice[index] = ua

                    success = float(train_env.unwrapped.termination_manager.get_term('goal_reached')[index])
                    success_train_p.append(success)

                    collision = float(train_env.unwrapped.termination_manager.get_term('base_contact')[index])
                    collision_train_p.append(collision)
        self.__writer.close()


class AdviceScheduler:
    def __init__(self) -> None:
        self._rng = None

    def __call__(self, current_step) -> bool:
        return True

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)


class LinearFixedBudgetAdviceScheduler(AdviceScheduler):
    def __init__(self, budget, advice_after, alpha_init, decay_rate) -> None:
        super().__init__()
        self.advice_after = advice_after
        self.alpha_init = alpha_init
        self.decay_rate = decay_rate

        self.budget = budget
        self.n_given = 0

    def __call__(self, current_step) -> bool:
        if (current_step < self.advice_after) or (self.n_given >= self.budget):
            return False
        else:
            adj_step = current_step - self.advice_after
            alpha = self.alpha_init - self.decay_rate * adj_step
            res = self._rng.uniform(0, 1) < alpha
            self.n_given += 1 if res else 0
            return res


class StepBudgetAdviceScheduler(AdviceScheduler):
    def __init__(self, budget, advice_after, advice_every) -> None:
        super().__init__()
        self._budget = 0
        self._budget_increment = budget
        self._advice_after = advice_after
        self._advice_every = advice_every
        self._next_update = advice_after

        self.n_given = 0

    def __call__(self, current_step) -> bool:
        if current_step >= self._next_update:
            self._budget += self._budget_increment
            self._next_update += self._advice_every

        if self.n_given < self._budget:
            self.n_given += 1
            return True
        else:
            return False


def get_advice_scheduler(scheduler_type, kwargs):
    C = __adv_schedulers[scheduler_type]
    return C(**kwargs)


__adv_schedulers = dict(
    linear_fixed_budget=LinearFixedBudgetAdviceScheduler,
    step_budget=StepBudgetAdviceScheduler,
)
