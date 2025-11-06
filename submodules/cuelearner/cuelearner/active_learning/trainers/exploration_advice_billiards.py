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

    def fit(self, policy, advice_cloner, train_env, val_env, expert):
        # save network hyperparameters
        dump_hparams(policy, self.log_dir.get_path("ckpt_policy/hparams.yaml"))
        advice_cloner.set_eval()
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
                "state": state_dim,
                "action": action_dim,
                "next_state": state_dim,
                "done": 1,
                "reward": 1,  # Assuming reward is a single value
            },
            seed=self.__rng.integers(low=0, high=1_000_000).item(),
            max_size=self.buffer_size,
        )

        self.advice_scheduler.seed(self.__rng.integers(low=0, high=1_000_000).item())
        # END ADVICE BUFFER

        # --- BASELINE
        random_reward = eval_random(val_env)
        timer = Timer()
        timer.start("total")
        scheduler_training = Scheduler(self.update_after, self.update_every)
        scheduler_adv_training = Scheduler(self.adv_update_after, self.adv_update_every)
        scheduler_eval = Scheduler(self.eval_every, self.eval_every)
        scheduler_print = Scheduler(self.print_every, self.print_every)

        # --- TRAINING
        advices_used = 0
        adv_step = 0
        advice_is_trained = False
        state, info = train_env.reset()
        done = False
        rews_train_p = []
        use_advice = False
        for step in range(1, self.max_steps):
            # Select action randomly or according to policy
            timer.start("action_choice")
            if step < self.start_steps:
                # get action at random
                action = train_env.action_space.sample()
            else:
                # get action from policy
                action = policy.select_action_train(state)
                # try to evaluate action quality
                # TODO: try to see it we can get rid of this criterion
                # since it would eliminate one magic number
                # expected_reward = policy.get_q_estimate(state, action)
                # decide whether to use advice or not
                if use_advice:
                    timer.start("action_improvement")
                    # improve action
                    # giulio: this is an ablation to see if using the random base makes a difference
                    action = train_env.action_space.sample()
                    action = advice_cloner.select_action(state=state, action=action)
                    timer.stop("action_improvement")
                    advices_used += 1
                # apply noise
                std_dev = self.advice_noise if use_advice else self.act_noise
                noise = self.__rng.normal(0, std_dev, size=action_dim)
                # Box action space
                # action = (action + noise).clip(-max_action, max_action)
                # spherical action space
                # TODO: this clipping is a bit weird and does not comply with normal clipping
                #       in the future, I should probably just pretend and clip each element, and then let the
                #       environment figure it out
                action = clip_norm(action + noise, min_action, max_action)
            timer.stop("action_choice")
            timer.start("simulator")
            old_info = info
            next_state, reward, done, terminated, info = train_env.step(action)
            timer.stop("simulator")
            replay_buffer.add(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
            )

            state = next_state
            rews_train_p.append(reward)

            if scheduler_training(step):
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
                policy.set_eval()
                advice_cloner.set_eval()
                timer.start("evaluation")
                with torch.no_grad():
                    rews = eval_policy(policy, val_env, steps=self.eval_steps)
                rew_avg = np.mean(rews)
                timer.stop("evaluation")
                self.__log("step", step, step)
                self.__log("random_policy_reward", random_reward, step)
                self.__log("val_reward", rew_avg, step)
                self.__log("train_reward", rews_train_p, step)
                self.__log_dict(timer.get_all(), step)
                rews_train_p = []

                # Save models
                self.__policy_ckpt_manager.update(
                    state_dict=policy.get_state_dict(),
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
                # for each episode, decide whether to use advice or not
                use_advice = (
                    (self.__rng.uniform(0, 1) < self.advice_probability)
                    and (step < self.advice_usage_limit)
                )
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
