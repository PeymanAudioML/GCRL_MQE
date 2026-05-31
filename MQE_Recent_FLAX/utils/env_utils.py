import collections
import os
import platform
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics.

    Tracks per-episode return, length, and wall-clock duration, and
    accumulates a global timestep counter across episodes.

    Args:
        env: The gymnasium environment to wrap.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self) -> None:
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray) -> tuple:
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> tuple:
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack the last ``num_stack`` observations.

    Concatenates frames along the last axis so that downstream agents
    receive temporal context without any recurrence.

    Args:
        env: The gymnasium environment to wrap.
        num_stack: Number of consecutive frames to stack.
    """

    def __init__(self, env: gymnasium.Env, num_stack: int) -> None:
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self) -> np.ndarray:
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs) -> tuple:
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action: np.ndarray) -> tuple:
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(
    dataset_name: str,
    dataset_path: str | None = None,
    frame_stack: int | None = None,
) -> tuple:
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        dataset_path: Optional path to a local dataset directory. When
            ``None``, the dataset is fetched from the default remote
            source.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name, dataset_path=dataset_path, compact_dataset=True)
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset
