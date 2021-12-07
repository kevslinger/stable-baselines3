import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, unwrap_vec_normalize
from scipy.special import softmax


def get_time_limit(env: VecEnv, current_max_episode_length: Optional[int]) -> int:
    """
    Get time limit from environment.

    :param env: Environment from which we want to get the time limit.
    :param current_max_episode_length: Current value for max_episode_length.
    :return: max episode length
    """
    # try to get the attribute from environment
    if current_max_episode_length is None:
        try:
            current_max_episode_length = env.get_attr("spec")[0].max_episode_steps
            # Raise the error because the attribute is present but is None
            if current_max_episode_length is None:
                raise AttributeError
        # if not available check if a valid value was passed as an argument
        except AttributeError:
            raise ValueError(
                "The max episode length could not be inferred.\n"
                "You must specify a `max_episode_steps` when registering the environment,\n"
                "use a `gym.wrappers.TimeLimit` wrapper "
                "or pass `max_episode_length` to the model constructor"
            )
    return current_max_episode_length


class ReplayBuffer(DictReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer.
    Paper: https://arxiv.org/abs/1707.01495

    .. warning::

      For performance reasons, the maximum number of steps per episodes must be specified.
      In most cases, it will be inferred if you specify ``max_episode_steps`` when registering the environment
      or if you use a ``gym.wrappers.TimeLimit`` (and ``env.spec`` is not None).
      Otherwise, you can directly pass ``max_episode_length`` to the replay buffer constructor.


    Replay buffer for sampling HER (Hindsight Experience Replay) transitions.
    In the online sampling case, these new transitions will not be saved in the replay buffer
    and will only be created at sampling time.

    :param env: The training environment
    :param buffer_size: The size of the buffer measured in transitions.
    :param max_episode_length: The maximum length of an episode. If not specified,
        it will be automatically inferred if the environment uses a ``gym.wrappers.TimeLimit`` wrapper.
    :param goal_selection_strategy: Strategy for sampling goals for replay.
        One of ['episode', 'final', 'future']
    :param device: PyTorch device
    :param n_sampled_goal: Number of virtual transitions to create per real transition,
        by sampling new goals.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        device: Union[th.device, str] = "cpu",
        max_episode_length: Optional[int] = None,
        online_sampling: bool = True,
        handle_timeout_termination: bool = True,
        prioritize_occlusions: int = 0, # -1 is False, 0 is None, 1 is True,
        run_name: str= ""
    ):

        super(ReplayBuffer, self).__init__(buffer_size, env.observation_space, env.action_space, device, env.num_envs)

        # if we sample her transitions online use custom replay buffer
        self.online_sampling = online_sampling

        # maximum steps in episode
        self.max_episode_length = get_time_limit(env, max_episode_length)
        # storage for transitions of current episode for offline sampling
        # for online sampling, it replaces the "classic" replay buffer completely
        her_buffer_size = buffer_size

        self.env = env
        self._vec_normalize_env = unwrap_vec_normalize(env)
        self.buffer_size = her_buffer_size

        self.replay_buffer = None
        self.online_sampling = True

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination

        # buffer with episodes
        # number of episodes which can be stored until buffer size is reached
        self.max_episode_stored = self.buffer_size // self.max_episode_length
        self.current_idx = 0
        # Counter to prevent overflow
        self.episode_steps = 0

        # Get shape of observation and goal (usually the same)
        self.obs_shape = get_obs_shape(self.env.observation_space.spaces["observation"])
        self.goal_shape = get_obs_shape(self.env.observation_space.spaces["achieved_goal"])

        # input dimensions for buffer initialization
        input_shape = {
            "observation": (self.env.num_envs,) + self.obs_shape,
            "achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "desired_goal": (self.env.num_envs,) + self.goal_shape,
            "action": (self.action_dim,),
            "reward": (1,),
            "next_obs": (self.env.num_envs,) + self.obs_shape,
            "next_achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "next_desired_goal": (self.env.num_envs,) + self.goal_shape,
            "done": (1,),
        }
        self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }
        # Store info dicts are it can be used to compute the reward (e.g. continuity cost)
        self.info_buffer = [deque(maxlen=self.max_episode_length) for _ in range(self.max_episode_stored)]
        # episode length storage, needed for episodes which has less steps than the maximum length
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

        self.occluded_goal_frac = []

        self.H_T = np.zeros((100, 100))
        self.run_name = run_name

        # OCCLUSION-BASED PRIORITY
        self._buffer["occlusions"] = np.zeros((self.max_episode_stored, 1, 1), dtype=np.float32)
        self.prioritize_occlusions = prioritize_occlusions

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize],
    ) -> DictReplayBufferSamples:
        """
        Sample function for online sampling of HER transition,
        this replaces the "regular" replay buffer ``sample()``
        method in the ``train()`` function.

        :param batch_size: Number of element to sample
        :param env: Associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: Samples.
        """
        minibatch = self._sample_transitions(batch_size, maybe_vec_env=env)  # pytype: disable=bad-return-type
        reward_fraction = 100 * (len(minibatch.rewards) - th.count_nonzero(minibatch.rewards)) / len(minibatch.rewards)
        self.reward_frac.append(reward_fraction.item())
        occluded_goal_fraction = 100 * (len(minibatch.observations['desired_goal']) - th.count_nonzero(th.count_nonzero(minibatch.observations['desired_goal'], axis=1))) / len(minibatch.observations['desired_goal'])
        self.occluded_goal_frac.append(occluded_goal_fraction.item())
        return minibatch

    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
    ) -> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
        if self.prioritize_occlusions == 0:
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
        else:
            # episode_indices = np.random.choice(range(self.n_episodes_stored), size=batch_size, replace=True,
            #                                    p=softmax(self._buffer["occlusions"][:self.n_episodes_stored]).flatten())
            episode_indices = np.random.choice(range(self.n_episodes_stored), size=batch_size, replace=True,
                                               p=((1 + self._buffer['occlusions'][:self.n_episodes_stored]) / (
                                                       self.n_episodes_stored + sum(
                                                   self._buffer['occlusions'][:self.n_episodes_stored]))).flatten())

        ep_lengths = self.episode_lengths[episode_indices]


        # Select which transitions to use
        transitions_indices = np.random.randint(ep_lengths)

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys() if key != "occlusions"}


        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # concatenate observation with (desired) goal
        observations = self._normalize_obs(transitions, maybe_vec_env)

        # HACK to make normalize obs and `add()` work with the next observation
        next_observations = {
            "observation": transitions["next_obs"],
            "achieved_goal": transitions["next_achieved_goal"],
            # The desired goal for the next observation must be the same as the previous one
            "desired_goal": transitions["desired_goal"],
        }
        next_observations = self._normalize_obs(next_observations, maybe_vec_env)

        next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

        normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

        return DictReplayBufferSamples(
            observations=normalized_obs,
            actions=self.to_torch(transitions["action"]),
            next_observations=next_obs,
            dones=self.to_torch(transitions["done"]),
            rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
        )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

        # Remove termination signals due to timeout
        if self.handle_timeout_termination:
            done_ = done * (1 - np.array([info.get("TimeLimit.truncated", False) for info in infos]))
        else:
            done_ = done

        self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]
        if self.prioritize_occlusions == 1:  # Occlusion based priority
            self._buffer["occlusions"][self.pos] += np.count_nonzero(obs['achieved_goal'] == 0.0)
        elif self.prioritize_occlusions == -1:  # NON-Occlusion based priority
            self._buffer["occlusions"][self.pos] += np.count_nonzero(obs['achieved_goal'])

        # When doing offline sampling
        # Add real transition to normal replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(
                obs,
                next_obs,
                action,
                reward,
                done,
                infos,
            )

        self.info_buffer[self.pos].append(infos)

        # update current pointer
        self.current_idx += 1

        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode()
            self.episode_steps = 0

    def store_episode(self) -> None:
        """
        Increment episode counter
        and reset transition pointer.
        """
        # add episode length to length storage
        self.episode_lengths[self.pos] = self.current_idx

        # update current episode pointer
        # Note: in the OpenAI implementation
        # when the buffer is full, the episode replaced
        # is randomly chosen
        self.pos += 1
        if self.pos == self.max_episode_stored:
            self.full = True
            self.pos = 0
        # reset transition pointer
        self.current_idx = 0

    @property
    def n_episodes_stored(self) -> int:
        if self.full:
            return self.max_episode_stored
        return self.pos

    def size(self) -> int:
        """
        :return: The current number of transitions in the buffer.
        """
        return int(np.sum(self.episode_lengths))

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.current_idx = 0
        self.full = False
        self.episode_lengths = np.zeros(self.max_episode_stored, dtype=np.int64)

    def truncate_last_trajectory(self) -> None:
        """
        Only for online sampling, called when loading the replay buffer.
        If called, we assume that the last trajectory in the replay buffer was finished
        (and truncate it).
        If not called, we assume that we continue the same trajectory (same episode).
        """
        # If we are at the start of an episode, no need to truncate
        current_idx = self.current_idx

        # truncate interrupted episode
        if current_idx > 0:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated.\n"
                "If you are in the same episode as when the replay buffer was saved,\n"
                "you should use `truncate_last_trajectory=False` to avoid that issue."
            )
            # get current episode and transition index
            pos = self.pos
            # set episode length for current episode
            self.episode_lengths[pos] = current_idx
            # set done = True for current episode
            # current_idx was already incremented
            self._buffer["done"][pos][current_idx - 1] = np.array([True], dtype=np.float32)
            # reset current transition index
            self.current_idx = 0
            # increment episode counter
            self.pos = (self.pos + 1) % self.max_episode_stored
            # update "full" indicator
            self.full = self.full or self.pos == 0

    def get_heatmap(self, n_calls, bin_range=0.1, plot=False):
        """Get a heatmap of the goals in the buffer"""
        #x_min, x_max = self.env.envs[0].x_left_limit, self.env.envs[0].x_right_limit
        #y_min, y_max = self.env.envs[0].y_down_limit, self.env.envs[0].y_up_limit
        x_min, x_max = self.env.envs[0].min_x, self.env.envs[0].max_x
        y_min, y_max = self.env.envs[0].min_y, self.env.envs[0].max_y
        x_bins = np.arange(x_min, x_max + bin_range, bin_range)
        y_bins = np.arange(y_min, y_max + bin_range, bin_range)

        sample = self._sample_transitions(256, self._vec_normalize_env)

        goals_x, goals_y = np.hsplit(sample.observations['desired_goal'][:, -2:], 2)
        goals_x = goals_x.flatten().cpu().numpy()
        goals_y = goals_y.flatten().cpu().numpy()
        H, xedges, yedges = np.histogram2d(goals_x, goals_y, bins=(x_bins, y_bins))

        self.H_T += H.T

        heatmap_dir = 'testHeatmap'
        if not os.path.exists(os.path.join(os.getcwd(), heatmap_dir)):
            os.mkdir(os.path.join(os.getcwd(), heatmap_dir))
        if not os.path.exists(os.path.join(os.getcwd(), heatmap_dir, self.run_name)):
            os.mkdir(os.path.join(os.getcwd(), heatmap_dir, self.run_name))
        np.save(os.path.join(os.getcwd(), heatmap_dir, self.run_name, f'{n_calls}.npy'), H.T)
        np.save(os.path.join(os.getcwd(), heatmap_dir, self.run_name, f'{n_calls}_cumulative.npy'), self.H_T)


        if plot:
            xcenters = (xedges[:-1] + xedges[1:]) / 2
            ycenters = (yedges[:-1] + yedges[1:]) / 2
            fig, ax = plt.subplots()
            plt.title(f'Heatmap of Goals, {self.run_name}, iteration {n_calls}')
            plt.xlim([xedges[0], xedges[-1]])
            plt.ylim([yedges[0], yedges[-1]])
            im = NonUniformImage(ax, interpolation='bilinear')
            im.set_data(xcenters, ycenters, self.H_T)
            ax.images.append(im)
            plt.savefig(os.path.join(os.getcwd(), heatmap_dir, self.run_name, f'{n_calls}.png'))
