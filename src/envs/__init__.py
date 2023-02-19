from functools import partial
from qplex_smac.smac.env import MultiAgentEnv, FakeStarCraft2Env
import sys
import os

from .lbforaging import ForagingEnv
from .join1 import Join1Env

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    "sc2fake": partial(env_fn, env=FakeStarCraft2Env),
    # "sc2": partial(env_fn, env=StarCraft2Env),
    "foraging": partial(env_fn, env=ForagingEnv),
    "join1": partial(env_fn, env=Join1Env),
}
