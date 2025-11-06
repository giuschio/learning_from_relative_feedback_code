import torch
import numpy as np
import os
import random


from torch.utils.data import Dataset


def _get_train_val_indexes(size, val_split):
    val_size = int(size * val_split)
    # Generate all indices
    all_indices = np.arange(size)
    step = size // val_size
    val_indices = all_indices[::step][:val_size]
    train_indices = np.setdiff1d(all_indices, val_indices)
    return train_indices, val_indices


class ReplayBuffer(Dataset):
    def __init__(self, fields, seed, max_size=int(1e6), device=None, validation_share=0.2):
        super(ReplayBuffer, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.seed = seed
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.__rng_numpy = np.random.default_rng(seed)
        self.__rng_python = random.Random(seed)
        fields["index"] = 1
        # Initialize storage for each field as a PyTorch tensor on the specified device
        self.storage = {
            field: torch.zeros((max_size, dim), device=self.device)
            for field, dim in fields.items()
        }

        self.fields = fields
        self.n_adds = 0
        self.n_adds_old = 0

        # compute training and validation indexes
        td, vd = _get_train_val_indexes(self.max_size, validation_share)
        self.__val = torch.zeros(self.max_size, dtype=torch.bool)
        self.__val[vd] = True
        self.__train = ~self.__val

        self.train_indexes = list()
        self.val_indexes = list()

    def _validate_input(self, **kwargs):
        # Utility function to validate input
        for key, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device)
                kwargs[key] = value  # Update kwargs with the tensor

        return kwargs

    def add(self, **kwargs):
        kwargs = self._validate_input(**kwargs)
        for key, value in kwargs.items():
            if key in self.storage:
                self.storage[key][self.ptr] = value
            else:
                raise ValueError(f"Field {key} not in buffer storage.")
        self.storage["index"][self.ptr] = self.n_adds
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.n_adds += 1

        if self.size < self.max_size:
            idxs = self.train_indexes if self.__train[self.ptr] else self.val_indexes
            idxs.append(self.ptr)

    def add_batch(self, **kwargs):
        num_envs = next(iter(kwargs.values())).shape[0]
        for i in range(num_envs):
            data = {k: kwargs[k][i] for k in kwargs.keys()}
            self.add(**data)

    @property
    def has_new_data(self):
        res = self.n_adds > self.n_adds_old
        self.n_adds_old = self.n_adds
        return res

    def __check(self):
        if self.size == 0:
            return RuntimeError("Trying to sample from an emtpy replay buffer")

    def sample(self, batch_size, subset=None):
        self.__check()
        if subset is None:
            ind = self.__rng_numpy.integers(0, self.size, size=batch_size)
        elif subset == "train":
            indexes = self.train_indexes
            ind = self.__rng_python.choices(indexes, k=batch_size)
        elif subset == "val":
            indexes = self.val_indexes
            ind = self.__rng_python.choices(indexes, k=batch_size)
        return {key: values[ind] for key, values in self.storage.items()}

    def get_all(self):
        self.__check()
        return {key: values[0 : len(self)] for key, values in self.storage.items()}

    def get_all_np(self):
        self.__check()
        return {
            key: values[0 : len(self)].cpu().numpy()
            for key, values in self.storage.items()
        }

    def __getitem__(self, idx):
        return {key: values[idx] for key, values in self.storage.items()}

    def __len__(self):
        return self.size

    def get_size(self, key):
        if key in self.storage:
            return self.storage[key].shape[1]
        else:
            raise ValueError(f"Field {key} not in buffer storage.")

    def to_file(self, fname):
        torch.save(
            {
                "max_size": self.max_size,
                "seed": self.seed,
                "ptr": self.ptr,
                "size": self.size,
                "fields": self.fields,
                "storage": self.storage,
                "train_indexes": self.train_indexes,
                "val_indexes": self.val_indexes,
                "n_adds": self.n_adds,
                "n_adds_old": self.n_adds_old,
                "rng_numpy_state": self.__rng_numpy.bit_generator.state,
                "rng_python_state": self.__rng_python.getstate(),
            },
            fname,
        )

    @classmethod
    def from_file(cls, fname, device=None):
        # Ensure the file exists
        if not os.path.exists(fname):
            raise FileNotFoundError(f"The file '{fname}' does not exist.")
        
        # Attempt to load the data
        try:
            data = torch.load(fname, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the replay buffer from '{fname}'. Error: {str(e)}")
        
        # Create the buffer using fields and max_size
        buffer = cls(data["fields"], seed=data["seed"], max_size=data["max_size"], device=device)
        
        # Restore buffer properties
        buffer.ptr = data["ptr"]
        buffer.size = data["size"]

        # Restore the storage to the correct device
        buffer.storage = {
            key: value.to(device) if device else value for key, value in data["storage"].items()
        }

        # Restore training and validation indexes
        buffer.train_indexes = data.get("train_indexes", [])
        buffer.val_indexes = data.get("val_indexes", [])

        # Restore additional internal state variables
        buffer.n_adds = data.get("n_adds", 0)
        buffer.n_adds_old = data.get("n_adds_old", 0)

        # Restore RNG states if available
        rng_numpy_state = data.get("rng_numpy_state")
        if rng_numpy_state is not None:
            buffer.__rng_numpy.bit_generator.state = rng_numpy_state

        rng_python_state = data.get("rng_python_state")
        if rng_python_state is not None:
            buffer.__rng_python.setstate(rng_python_state)

        return buffer

    def to(self, device):
        if device != self.device:
            self.device = device
            for key in self.storage:
                self.storage[key] = self.storage[key].to(device)


class NStepReplayBuffer:
    def __init__(self, replay_buffer, num_envs, n_step, gamma):
        self.replay_buffer = replay_buffer  # The underlying ReplayBuffer instance
        self.num_envs = num_envs  # Number of parallel environments
        self.n_step = n_step  # The number of steps to accumulate rewards over
        self.gamma = gamma  # Discount factor for future rewards
        self.n_step_buffers = {env_id: [] for env_id in range(num_envs)}  # Buffers per environment

    def add(self, env_id, **kwargs):
        # Add the current transition to the appropriate environment's n-step buffer
        self.n_step_buffers[env_id].append(kwargs)

        # If the buffer contains enough steps, or the episode has ended, process the buffer
        if len(self.n_step_buffers[env_id]) == self.n_step or kwargs.get('done', False):
            self._commit_to_replay_buffer(env_id)

        # If done=True, flush the remaining transitions
        if kwargs.get('done', False):
            self.flush(env_id)

    def add_batch(self, batch_data):
        # Iterate over the batch, adding each transition to the appropriate environment's buffer
        env_ids = batch_data['env_id']
        for i, env_id in enumerate(env_ids):
            transition = {key: batch_data[key][i] for key in batch_data if key != 'env_id'}
            self.add(env_id, **transition)

    def _commit_to_replay_buffer(self, env_id):
        # Calculate n-step reward, final state, and done_n for the given environment
        n_step_reward = 0.0
        gamma_power = 1.0
        done_n = False
        for i, transition in enumerate(self.n_step_buffers[env_id]):
            n_step_reward += gamma_power * transition['reward']
            gamma_power *= self.gamma
            if transition.get('done', False):
                done_n = True
                break  # Stop accumulating if the episode has ended

        # Determine the n-th next state and done_n
        if len(self.n_step_buffers[env_id]) >= self.n_step:
            next_state_n = self.n_step_buffers[env_id][-1]['next_state']
        else:
            next_state_n = self.n_step_buffers[env_id][-1]['next_state']
            done_n = done_n or self.n_step_buffers[env_id][-1]['done']

        # Prepare the data to be added to the replay buffer
        transition = {
            **self.n_step_buffers[env_id][0],  # Use the first transition's data as the base
            'reward_n': n_step_reward,
            'next_state_n': next_state_n,
            'done_n': done_n,
        }

        # Add the processed transition to the replay buffer
        self.replay_buffer.add(**transition)

        # Remove the first transition (as it has now been processed)
        self.n_step_buffers[env_id].pop(0)

    def flush(self, env_id):
        # Flush the specific environment's buffer
        while len(self.n_step_buffers[env_id]) > 0:
            self._commit_to_replay_buffer(env_id)