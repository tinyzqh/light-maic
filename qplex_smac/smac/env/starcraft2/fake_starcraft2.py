from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from qplex_smac.smac.env.multiagentenv import MultiAgentEnv
# from qplex_smac.smac.env.starcraft2.maps import get_map_params

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class FakeStarCraft2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
        self,
        map_name="fake",
        seed=None
    ):

        self.n_agents = 10  # 智能体的数量
        self.episode_limit = 150

        # Other
        self._seed = seed

        # Actions
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = 18

        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0

        # Qatten
        self.unit_dim = 7

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """A single environment step. Returns reward, terminated, info."""
        reward = np.random.randn()
        terminated = False if np.random.randn() < 0.9 else True
        info = {}
        return reward, terminated, info

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """

        agent_obs = np.array([np.random.randint(0, 2) for _ in range(176)], dtype=np.float32)
        return agent_obs

    def get_obs(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        """Returns the global state.
        NOTE: This functon should not be used during decentralised execution.
        """

        state = np.array([np.random.randn() for _ in range(322)])
        return state

    def get_obs_size(self):
        """Returns the size of the observation."""
        # 单个智能体的观测 176维
        return 176

    def get_state_size(self):
        """Returns the size of the global state."""
        # 全局状态
        return 322

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = [np.random.randint(0, 2) for _ in range(self.n_actions)]
            avail_actions.append(avail_agent)
        return avail_actions

    def close(self):
        pass

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def render(self):
        """Not implemented."""
        pass

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "battles_draw": self.timeouts,
            "win_rate": self.battles_won / self.battles_game,
            "timeouts": self.timeouts,
            "restarts": self.force_restarts,
        }
        return stats
