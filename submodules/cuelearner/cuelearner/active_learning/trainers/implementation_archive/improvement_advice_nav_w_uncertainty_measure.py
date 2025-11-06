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


def angle(v1, v2):
    """
    Calculate the positive angle in degrees between two vectors.
    """
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Calculate the angle in degrees
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))


def rotate_vec(vector, degrees):
    angle_rad = np.radians(degrees)
    # Create the rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return np.dot(rotation_matrix, np.asarray(vector))

def compute_advice(action, optimal_action, advice_step_size):
    next_actions = [
        rotate_vec(vector=action, degrees=d)
        for d in [-1.0 * advice_step_size, 0.0, advice_step_size]
    ]
    distances = np.array([angle(optimal_action, a) for a in next_actions])
    return next_actions[np.argmin(np.abs(distances))]

def compute_advice_direction(action, optimal_action, advice_step_size):
    directions = [-1.0 * advice_step_size, 0.0, advice_step_size]
    next_actions = [
        rotate_vec(vector=action, degrees=d)
        for d in directions
    ]
    distances = np.array([angle(optimal_action, a) for a in next_actions])
    return directions[np.argmin(np.abs(distances))]

# def get_user_choice(hint):
#     dictionary = {0: 0.0, 1: -1.0, 2: 1.0}
#     while True:
#         try:
#             # red is negative rotation
#             choice = int(
#                 input(
#                     f"Choose improvement: \n  -(0) Heading is optimal {'*' if hint == 0. else ''} \n  -(1) Red {'*' if hint < 0. else ''} \n  -(2) Green {'*' if hint > 0. else ''}\n -(3) Don't know\nchoice: "
#                 )
#             )
#             if choice in [0, 1, 2]:
#                 return dictionary[choice]
#             else:
#                 return None
#         except ValueError:
#             print("Invalid input. Please enter a numeric value (1 or 2).")


def get_user_choice():
    dictionary = {0: 0.0, 1: -1.0, 2: 1.0}
    while True:
        try:
            # red is negative rotation
            choice = int(
                input(
                    "Choose improvement: \n  -(0) Heading is optimal  \n  -(1) Red \n  -(2) Green \n -(3) Don't know\nchoice: "
                )
            )
            if choice in [0, 1, 2]:
                return dictionary[choice]
            else:
                return None
        except ValueError:
            print("Invalid input. Please enter a numeric value (1 or 2).")



class HeadingUncertainty:
    def __init__(self, buffer_size) -> None:
        self.buffer_size = buffer_size
        self.q = deque(maxlen=buffer_size)

    @staticmethod
    def _angle(v1, v2):
        """
        Calculate the positive angle in degrees between two vectors.
        """
        v1, v2 = np.array(v1).flatten(), np.array(v2).flatten()
        dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # Calculate the angle in degrees
        return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    @staticmethod
    def _mean_oscillation(v: deque):
        if len(v) == v.maxlen:
            mean_uncertainty_angle = sum([HeadingUncertainty._angle(v[i][:2], v[i+1][:2]) for i in range(len(v) - 1)])/(len(v) - 1)
            return mean_uncertainty_angle / 90.
        else:
            return 0.

    def append(self, action):
        self.q.append(action)

    def reset(self):
        self.q = deque(maxlen=self.buffer_size)

    def get_uncertainty(self):
        return self._mean_oscillation(self.q)



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
        try:
            dump_hparams(policy, self.log_dir.get_path("ckpt_policy/hparams.yaml"))
        except Exception as e:
            print("Could not dump hyperparams of the base policy")
        dump_hparams(advice_cloner, self.log_dir.get_path("ckpt_advice/hparams.yaml"))

        # initialize replay buffer
        # state_dim = train_env.observation_space.shape[0]
        state_dim = train_env.state_dim
        action_dim = train_env.action_space.shape[0]
        # Box action space
        # max_action = float(train_env.action_space.high[0])
        # spherical action space
        max_action = train_env.action_space.max_action
        min_action = train_env.action_space.min_action

        # --- ADVICE BUFFER
        advice_buffer = ReplayBuffer(
            device=policy.device,
            fields={
                "state": state_dim,
                "action": action_dim,
                "next_state": state_dim,
                "done": 1,
                "reward": 1,  # Assuming reward is a single value
                "improved_action": action_dim,
                "closest_expert_action": action_dim,
            },
            seed=self.__rng.integers(low=0, high=1_000_000).item(),
            max_size=self.advice_scheduler.budget,
        )

        self.advice_scheduler.seed(self.__rng.integers(low=0, high=1_000_000).item())
        # END ADVICE BUFFER

        # --- BASELINE
        # random_reward = eval_random(val_env)
        random_reward = 0.
        timer = Timer()
        timer.start("total")
        scheduler_adv_training = Scheduler(self.adv_update_after, self.adv_update_every)
        scheduler_eval = Scheduler(self.eval_every, self.eval_every)
        scheduler_print = Scheduler(self.print_every, self.print_every)

        print("Start training")
        # --- TRAINING
        advices_used = 0
        adv_step = 0
        state, info = train_env.reset()
        rews_train_p = []
        success_train_p = []
        collision_train_p = []

        num_envs = 1
        done = [False]
        uncertainty_measure = HeadingUncertainty(buffer_size=20)

        for step in range(1, self.max_steps, num_envs):
            # Select action randomly or according to policy
            timer.start("action_choice")
            action = policy.select_action_batch(state)
            timer.start("action_improvement")

            advised_action = advice_cloner.select_action(state[0], action[0])
            use_advice = uncertainty_measure.get_uncertainty() > 0.15
            uncertainty_measure.append(action[0])
            if use_advice:
                action[0] = advised_action
            timer.stop("action_improvement")

            timer.stop("action_choice")
            timer.start("simulator")
            old_info = info
            next_state, reward, done, terminated, info = train_env.step(action)
            timer.stop("simulator")

            if use_advice and self.advice_scheduler(step):
                if expert == 'human':
                    # uncomment for debugging
                    # this calculates a hint (i.e. go towards the goal)
                    # since orbit seems to have some issues with rendering the arrows, you
                    # can check this hint to figure out if there is an issue with rendering
                    # timer.start("a_star")
                    # optimal_action = train_env.get_navigation_observation()['goal_embedded'][:, :2]
                    # timer.stop("a_star")

                    # direction_hint = compute_advice_direction(
                    #     action[0],
                    #     optimal_action[0],
                    #     advice_cloner.advice_step_size
                    # )

                    # human_feedback = get_user_choice(direction_hint)
                    human_feedback = get_user_choice()
                    if human_feedback is not None:
                        action_rotation = advice_cloner.advice_step_size * human_feedback
                        improved_action = rotate_vec(action[0], action_rotation)
                        advice_buffer.add(
                            state=state[0],
                            action=action[0],
                            next_state=next_state[0],
                            reward=reward[0],
                            done=done[0],
                            improved_action=improved_action,
                        )
                        adv_step += 1
                    else:
                        pass
                else:
                    timer.start("a_star")
                    optimal_action = expert.select_action(train_env.get_navigation_observation())
                    timer.stop("a_star")
                    improved_action = compute_advice(
                        action[0],
                        optimal_action[0],
                        advice_cloner.advice_step_size
                    )

                    advice_buffer.add(
                        state=state[0],
                        action=action[0],
                        next_state=next_state[0],
                        reward=reward[0],
                        done=done[0],
                        improved_action=improved_action,
                    )
                    adv_step += 1

            state = next_state
            [rews_train_p.append(float(r)) for r in reward]

            if adv_step > 2000:
                scheduler_adv_training.update_every = 100

            if scheduler_adv_training(adv_step):
                print(f"training advice at advice step {adv_step}")
                timer.start("advice_training")
                advice_cloner.set_train()
                res = advice_cloner.fit(advice_buffer, self.adv_grad_steps)
                timer.stop("advice_training")
                self.__log_dict(res, step)
                self.__log("step", step, step)
                self.__advice_ckpt_manager.update(
                    state_dict=advice_cloner.get_state_dict(),
                    epoch=adv_step,
                )
                advice_cloner.set_eval()
                advice_is_trained = True
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
                self.__log_dict(timer.get_all(), step)
                rews_train_p = []
                success_train_p = []
                collision_train_p = []

                # # Save models
                # self.__policy_ckpt_manager.update(
                #     state_dict=policy.get_state_dict(),
                #     epoch=step,
                #     metric=avg_train_reward,
                #     metric_name="train_reward",
                # )
            if scheduler_print(step):
                self.__print_log()

            for index in range(num_envs):
                if done[index]:
                    success = float(train_env.env.termination_manager.get_term('goal_reached')[index])
                    success_train_p.append(success)

                    collision = float(train_env.env.termination_manager.get_term('base_contact')[index])
                    collision_train_p.append(collision)
                    uncertainty_measure.reset()

            if adv_step >= self.advice_scheduler.budget:
                break
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
