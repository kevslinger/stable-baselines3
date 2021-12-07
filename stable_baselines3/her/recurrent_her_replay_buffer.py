from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from scipy.special import softmax

class RecurrentHerReplayBuffer(HerReplayBuffer):
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
        replay_buffer: Optional[DictReplayBuffer] = None,
        max_episode_length: Optional[int] = None,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
        handle_timeout_termination: bool = True,
        prioritize_occlusions: int = 0,  # -1 is False, 0 is None, 1 is True,
        run_name: str = ""
    ):

        super(RecurrentHerReplayBuffer, self).__init__(env, buffer_size, device, replay_buffer, max_episode_length, n_sampled_goal,
                                                             goal_selection_strategy, online_sampling, handle_timeout_termination,
                                                       prioritize_occlusions, run_name)

    def sample_goals(
        self,
        episode_indices: np.ndarray,
        her_indices: np.ndarray,
        transitions_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Sample goals based on goal_selection_strategy.
        This is a vectorized (fast) version.

        :param episode_indices: Episode indices to use.
        :param her_indices: HER indices.
        :param transitions_indices: Transition indices to use.
        :return: Return sampled goals.
        """
        her_episode_indices = episode_indices[her_indices]

        if self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # replay with final state of current episode
            transitions_indices = self.episode_lengths[her_episode_indices] - 1

        elif self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # replay with random state which comes from the same episode and was observed after current transition
            transitions_indices = np.random.randint(
                transitions_indices[her_indices] + 1, self.episode_lengths[her_episode_indices]
            )

        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # replay with random state which comes from the same episode as current transition
            transitions_indices = np.random.randint(self.episode_lengths[her_episode_indices])

        elif self.goal_selection_strategy == GoalSelectionStrategy.OCCLUDED:
            # I want to take the trajectory and prioritize those transitions which are occluded.
            # This is no longer a vectorized (fast) version
            goals_to_assign = []
            for i, her_ep_idx in enumerate(her_episode_indices):
                potential_goals = self._buffer["achieved_goal"][her_ep_idx, transitions_indices[her_indices[i]] + 1:].squeeze()
                if len(potential_goals) == 2 * self.env.envs[0].hist_len and softmax(np.count_nonzero(potential_goals == 0.0, axis=-1)).flatten()[0] == 1:
                    goals_to_assign.append(potential_goals)
                else:
                    #print(potential_goals)
                    goals_to_assign.append(potential_goals[np.random.choice(range(len(potential_goals)), p=softmax(np.count_nonzero(potential_goals == 0.0, axis=-1)).flatten())])
            return np.expand_dims(np.array(goals_to_assign), axis=1)

        elif self.goal_selection_strategy == GoalSelectionStrategy.UNOCCLUDED:
            goals_to_assign = []
            for i, her_ep_idx in enumerate(her_episode_indices):
                potential_goals = self._buffer["achieved_goal"][her_ep_idx,
                                  transitions_indices[her_indices[i]] + 1:].squeeze()
                if len(potential_goals) == 2 * self.env.envs[0].hist_len and softmax(np.count_nonzero(potential_goals, axis=-1)).flatten()[0] == 1:
                    goals_to_assign.append(potential_goals)
                else:
                    goals_to_assign.append(potential_goals[np.random.choice(range(len(potential_goals)), p=softmax(
                        np.count_nonzero(potential_goals, axis=-1)).flatten())])
            return np.expand_dims(np.array(goals_to_assign), axis=1)

        else:
            raise ValueError(f"Strategy {self.goal_selection_strategy} for sampling goals not supported!")

        return np.tile(self._buffer["achieved_goal"][her_episode_indices, transitions_indices][:, :, -int(self.goal_shape[0]/self.env.envs[0].hist_len):], self.env.envs[0].hist_len)
        #return np.tile(self._buffer["achieved_goal"][her_episode_indices, transitions_indices], self.env.envs[0].hist_len)

    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
        n_sampled_goal: Optional[int] = None,
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
        # A subset of the transitions will be relabeled using HER algorithm
        her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]

        ep_lengths = self.episode_lengths[episode_indices]

        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE or \
                self.goal_selection_strategy == GoalSelectionStrategy.UNOCCLUDED or \
                self.goal_selection_strategy == GoalSelectionStrategy.OCCLUDED:
            # restrict the sampling domain when ep_lengths > 1
            # otherwise filter out the indices
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        # Select which transitions to use
        transitions_indices = np.random.randint(ep_lengths)

        # get selected transitions
        transitions = {key: self._buffer[key][episode_indices, transitions_indices].copy() for key in self._buffer.keys() if key != "occlusions"}

        # sample new desired goals and relabel the transitions
        new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices)
        transitions["desired_goal"][her_indices] = new_goals

        # Convert info buffer to numpy array
        transitions["info"] = np.array(
            [
                self.info_buffer[episode_idx][transition_idx]
                for episode_idx, transition_idx in zip(episode_indices, transitions_indices)
            ]
        )

        # Edge case: episode of one timesteps with the future strategy
        # no virtual transition can be created
        if len(her_indices) > 0:
            # Vectorized computation of the new reward
            transitions["reward"][her_indices, 0] = self.env.env_method(
                "compute_reward",
                # the new state depends on the previous state and action
                # s_{t+1} = f(s_t, a_t)
                # so the next_achieved_goal depends also on the previous state and action
                # because we are in a GoalEnv:
                # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                transitions["next_achieved_goal"][her_indices, 0],
                # here we use the new desired goal
                transitions["desired_goal"][her_indices, 0],
                transitions["info"][her_indices, 0],
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
