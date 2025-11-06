import glob
import numpy as np
import os
import pandas as pd
import torch

from omegaconf import OmegaConf
from time import time as get_secs


def print_reward_summary(dataset):
    print(f"Size of the dataset: {len(dataset)}")
    rewards = np.array([dataset[i]["reward"] for i in range(len(dataset))])

    unique_elements, counts = np.unique(rewards, return_counts=True)
    total_elements = rewards.size
    percentages = (counts / total_elements) * 100

    # Print results
    print(f"Average reward: {np.mean(rewards):.3f}")
    print("Reward distribution")
    for element, percentage in zip(unique_elements, percentages):
        print(f"  * {element}: {percentage:.2f}%")


def print_average_step_reward(dataset):
    print(f"Size of the dataset: {len(dataset)}")
    rewards = np.array([float(dataset[i]["reward"]) for i in range(len(dataset))])

    # Print results
    print(f"Average reward: {np.mean(rewards):.3f}")


def omega_to_yaml(config, fname):
    # Convert the OmegaConf object to a YAML formatted string
    yaml_str = OmegaConf.to_yaml(config)

    # Write the YAML string to a file
    with open(fname, "w") as file:
        file.write(yaml_str)


def is_pytorch(tensor):
    return isinstance(tensor, torch.Tensor)


def clip_norm_pytorch(tensor, min_norm, max_norm):
    norms = torch.norm(tensor, p=2, dim=1, keepdim=True)
    # Clamp the norms to be within the range [min_norm, max_norm]
    clamped_norms = torch.clamp(norms, min=min_norm, max=max_norm)
    scale = clamped_norms / norms
    # Avoid division by zero in case of zero norms
    scale[norms == 0] = 1
    return tensor * scale


def clip_norm_numpy(array, min_norm, max_norm):
    norm = np.linalg.norm(array)
    if norm > 0:
        # Compute scale factor within [min_norm / norm, max_norm / norm]
        clipped_norm = np.clip(norm, min_norm, max_norm)
        scale_factor = clipped_norm / norm
    else:
        scale_factor = 1  # Avoid division by zero for zero norm
    return array * scale_factor


def clip_norm(vector, min_norm, max_norm):
    if is_pytorch(vector):
        return clip_norm_pytorch(vector, min_norm, max_norm)
    else:
        return clip_norm_numpy(vector, min_norm, max_norm)


def rotate_vector(vector, degrees):
    if is_pytorch(vector):
        radian_f = torch.deg2rad
        cos_f = torch.cos
        sin_f = torch.sin
        matmul_f = torch.matmul
        tensor_f = torch.tensor
        dtype = vector.dtype
    else:
        radian_f = np.deg2rad
        cos_f = np.cos
        sin_f = np.sin
        matmul_f = np.dot
        tensor_f = np.array
        dtype = None

    rads = radian_f(degrees)
    R = tensor_f([[cos_f(rads), -sin_f(rads)], [sin_f(rads), cos_f(rads)]], dtype=dtype)
    return matmul_f(R, vector)


def angle(v1, v2):
    if is_pytorch(v1) and is_pytorch(v2):
        norm_f = torch.norm
        dot_f = torch.dot
        clamp_f = torch.clamp
        arccos_f = torch.arccos
        degrees_f = torch.rad2deg
    else:
        norm_f = np.linalg.norm
        dot_f = np.dot
        clamp_f = np.clip
        arccos_f = np.arccos
        degrees_f = np.degrees
    v1, v2 = v1.flatten(), v2.flatten()
    v1_normalized = v1 / norm_f(v1)
    v2_normalized = v2 / norm_f(v2)
    cosine_similarity = clamp_f(dot_f(v1_normalized, v2_normalized), -1, 1)
    angle = arccos_f(cosine_similarity)
    return degrees_f(angle)


def signed_angle(v1, v2):
    # Positive if v2 is counterclockwise to v1
    if is_pytorch(v1) and is_pytorch(v2):
        norm_f = torch.norm
        dot_f = torch.dot
        atan2_f = torch.atan2
        rad2deg_f = torch.rad2deg
    else:
        norm_f = np.linalg.norm
        dot_f = np.dot
        atan2_f = np.arctan2
        rad2deg_f = np.degrees

    v1, v2 = v1.flatten(), v2.flatten()
    v1_normalized = v1 / norm_f(v1)
    v2_normalized = v2 / norm_f(v2)

    # Compute the cosine similarity
    cosine_similarity = dot_f(v1_normalized, v2_normalized)

    # Compute the sine similarity using the 2D cross product manually
    sine_similarity = (
        v1_normalized[0] * v2_normalized[1] - v1_normalized[1] * v2_normalized[0]
    )

    # Compute the signed angle using atan2
    angle_rad = atan2_f(sine_similarity, cosine_similarity)

    # Convert the angle from radians to degrees
    angle_deg = rad2deg_f(angle_rad)

    return angle_deg


class LogDir:
    def __init__(self, root_folder: str, seed: int):
        self.root_dir = None
        self.checkpoints_dir = None
        self.seed = seed
        self._setup_directories(root_folder)

    def _setup_directories(self, root_folder):
        # Create root folder if it doesn't exist
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)

        basename = f"version_{self.seed}_"

        # Find the next version_X folder
        v_folders = glob.glob(os.path.join(root_folder, f"{basename}*"))
        v_indexes = [int(os.path.basename(f).replace(basename, "")) for f in v_folders]
        current_version = 0 if len(v_indexes) == 0 else max(v_indexes) + 1
        version_folder = os.path.join(root_folder, f"{basename}{current_version}")
        self.root_dir = version_folder

    def get_path(self, path):
        full_path = os.path.join(self.root_dir, path)
        dpath = (
            full_path
            if path.endswith(os.sep) or not os.path.splitext(path)[1]
            else os.path.dirname(full_path)
        )
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        return full_path


class CheckpointManager:
    def __init__(
        self,
        directory,
        mode="min",
        save_last=True,
        save_best=True,
        save_period=-1,
    ):
        self.directory = directory
        self.save_last = save_last
        self.save_best = save_best
        self.save_periodic = save_period > 0
        self.period = save_period
        self.next_save = save_period
        self.best_metric = float("inf")
        self.last_checkpoint_path = None
        self.best_checkpoint_path = None

        self.k = -1 if mode == "max" else 1

    def update(self, state_dict, epoch, metric=0, metric_name=None):
        if self.directory is None:
            raise ValueError("CheckpointManager.directory is None")
        # Save the last checkpoint if enabled
        if self.save_last:
            # Remove the previous checkpoint
            if self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path):
                os.remove(self.last_checkpoint_path)

            # Save the current checkpoint
            current_checkpoint_path = os.path.join(self.directory, f"last.ckpt")
            torch.save(state_dict, current_checkpoint_path)
            self.last_checkpoint_path = current_checkpoint_path

        # Update and save the best checkpoint if enabled
        metric_u = self.k * metric
        if self.save_best and (metric_u < self.best_metric):
            self.best_metric = metric_u
            checkpoint_name = f"model-epoch={epoch}-{metric_name}={metric:.2f}.ckpt"
            best_checkpoint_path = os.path.join(self.directory, checkpoint_name)
            torch.save(state_dict, best_checkpoint_path)

            # Remove the old best checkpoint
            if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                os.remove(self.best_checkpoint_path)
            self.best_checkpoint_path = best_checkpoint_path

        # Save periodic checkpoints if enabled
        if self.save_periodic and epoch >= self.next_save:
            self.next_save += self.period
            periodic_checkpoint_path = os.path.join(
                self.directory, f"epoch={epoch}.ckpt"
            )
            torch.save(state_dict, periodic_checkpoint_path)


class Logger:
    def __init__(self):
        self.data = pd.DataFrame()
        self.new_data = False

    def add_scalar(self, key, value, epoch):
        # Check if the key is a new column and add it if necessary
        if key not in self.data.columns:
            self.data[key] = pd.NA

        # Set the value for the given epoch and key
        self.data.at[epoch, key] = value
        self.new_data = True

    def __str__(self):
        if not self.new_data:
            return "No new data to display."

        ff = lambda value, sig_digits=2: (
            f"{value:.{max(-int(np.floor(np.log10(abs(value)))) + sig_digits - 1, 0)}f}"
            if value != 0
            else f"{0:.{sig_digits}f}"
        )

        self.new_data = False
        output = ["----"]
        for key in self.data.columns:
            # Get the last valid (non-NaN) value in the column
            last_valid_value = self.data[key].dropna().last_valid_index()
            value = (
                self.data.at[last_valid_value, key]
                if last_valid_value is not None
                else pd.NA
            )

            if pd.isna(value):
                output.append(f"{key:20}: {value}")
            else:
                formatted_value = ff(value)
                output.append(f"{key:20}: {formatted_value}")
        return "\n".join(output)

    def to_csv(self, fname, overwrite=True):
        if os.path.exists(fname) and not overwrite:
            raise FileExistsError(
                f"The file '{fname}' already exists. Set overwrite=True to overwrite it."
            )
        self.data.to_csv(fname)

    @staticmethod
    def from_csv(fname):
        logger = Logger()
        logger.data = pd.read_csv(fname, index_col=0)
        return logger


def get_checkpoint(path):
    return glob.glob(os.path.join(path, "checkpoints/model*.ckpt"))[0]


class Timer:
    def __init__(self):
        # Initializes a dictionary to store timer start times and elapsed times
        self.timers = {}

    def start(self, key):
        # Start or restart a timer by the given key
        if key not in self.timers:
            self.timers[key] = {"start_time": get_secs(), "elapsed": 0}
        else:
            # If the timer is already running, we reset its start time
            self.timers[key]["start_time"] = get_secs()

    def stop(self, key):
        # Stop the timer and calculate the elapsed time
        if key in self.timers and "start_time" in self.timers[key]:
            elapsed_time = get_secs() - self.timers[key]["start_time"]
            self.timers[key]["elapsed"] += elapsed_time
            # Remove the start_time to indicate the timer is not currently running
            del self.timers[key]["start_time"]

    def _current_elapsed(self, key):
        # Helper method to get current elapsed time, including running time if not stopped
        if key in self.timers:
            elapsed = self.timers[key].get("elapsed", 0)
            if "start_time" in self.timers[key]:
                elapsed += get_secs() - self.timers[key]["start_time"]
            return elapsed / 60.0
        return 0

    def get(self, key):
        # Return the total elapsed time for the timer, including any currently running time
        return self._current_elapsed(key)

    def get_all(self):
        # Returns a dictionary with keys modified to include the total elapsed time
        return {f"time_{key}": self._current_elapsed(key) for key in self.timers}

    def reset(self):
        # Reset all timers
        self.timers = {}
