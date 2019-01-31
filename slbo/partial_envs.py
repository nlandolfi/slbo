# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from slbo.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.walker2d_env import Walker2DEnv
from slbo.envs.mujoco.humanoid_env import HumanoidEnv
from slbo.envs.mujoco.ant_env import AntEnv
from slbo.envs.mujoco.hopper_env import HopperEnv
from slbo.envs.mujoco.swimmer_env import SwimmerEnv
from slbo.envs.mujoco.half_cheetah_task_env import HalfCheetahTaskEnv
from slbo.envs.mujoco.swimmer_task_env import SwimmerTaskEnv


def make_env(id: str, task_config=None):
    envs = {
        'HalfCheetah-v2': HalfCheetahEnv,
        'Walker2D-v2': Walker2DEnv,
        'Humanoid-v2': HumanoidEnv,
        'Ant-v2': AntEnv,
        'Hopper-v2': HopperEnv,
        'Swimmer-v2': SwimmerEnv,
    }
    task_envs = {
        'HalfCheetahTask-v2': HalfCheetahTaskEnv,
        'SwimmerTask-v2': SwimmerTaskEnv,
    }
    if id in envs:
        env = envs[id]()
    elif id in task_envs:
        env = envs[id](task_config=task_config)
    else:
        raise Exception(f"env {id} not recognized")
        
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    return env
