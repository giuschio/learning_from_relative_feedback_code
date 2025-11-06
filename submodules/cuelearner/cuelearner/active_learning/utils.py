import numpy as np

from omegaconf import OmegaConf


def eval_policy(policy, env, steps, **kwargs):
    """
    Assumes the policy has a function policy.select_action(state, **kwargs)
    """
    step_rewards = np.zeros(steps)

    state, info = env.reset()
    done = False
    for i in range(steps):

        action = policy.select_action(state, **kwargs)
        state, reward, done, terminated, info = env.step(action)
        step_rewards[i] = info["eval_reward"]

        if done:
            state, info = env.reset()
            done = False
    return step_rewards


def eval_random(env):
    class RPolicy:
        @staticmethod
        def select_action(state):
            return env.action_space.sample()

    random_reward = np.mean(eval_policy(RPolicy(), env, 500))
    return random_reward


def dump_hparams(net, fpath):
    conf = OmegaConf.create(net.hparams)
    with open(fpath, "w") as f:
        OmegaConf.save(config=conf, f=f.name)


class UpdateScheduler:
    def __init__(self) -> None:
        self.next_update = None

    def get_next(self):
        raise NotImplementedError()

    def __call__(self, current_step) -> bool:
        if current_step >= self.next_update:
            self.next_update = self.get_next()
            return True
        else:
            return False


class LinearUpdateScheduler(UpdateScheduler):
    def __init__(self, update_after, update_every) -> None:
        super().__init__()
        self.update_after = update_after
        self.update_every = update_every

        self.next_update = update_after

    def get_next(self):
        return self.next_update + self.update_every


class ExponentialUpdateScheduler(UpdateScheduler):
    def __init__(self, update_after, update_ratio) -> None:
        super().__init__()
        self.update_after = update_after
        self.update_ratio = update_ratio

        self.next_update = update_after

    def get_next(self):
        a = self.next_update
        while a - self.next_update < self.update_after:
            a = a * self.update_ratio
        return a


__update_schedulers = dict(
    linear=LinearUpdateScheduler,
    exponential=ExponentialUpdateScheduler,
)


def get_update_scheduler(type, **kwargs):
    MType = __update_schedulers[type]
    return MType(**kwargs)
