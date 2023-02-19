import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run

logger = get_logger()

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


def my_main(_config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(config, _log)


def _get_comment(params, arg_name):
    comment = ''
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            comment = _v.split("=")[1]
            del params[_i]
            break
    return comment


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.Loader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    # params = deepcopy(sys.argv)

    params = ['src/main.py', '--config=maic_qplex', '--env-config=sc2fake', 'with', 'env_args.map_name=MMM2', 'seed=42']
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    comment = _get_comment(params, '--comment')
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict['comment'] = comment

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")

    config_dict.update({'seed': 42})
    my_main(config_dict, logger)
