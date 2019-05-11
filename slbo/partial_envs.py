# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from slbo.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.walker2d_env import Walker2DEnv
from slbo.envs.mujoco.humanoid_env import HumanoidEnv
from slbo.envs.mujoco.ant_env import AntEnv
from slbo.envs.mujoco.hopper_env import HopperEnv
from slbo.envs.mujoco.swimmer_env import SwimmerEnv
from slbo.envs.mujoco.ant_task_env import AntTaskEnv, AntTaskConfig
from slbo.envs.mujoco.ant2d_task_env import Ant2DTaskEnv, Ant2DTaskConfig
from slbo.envs.mujoco.half_cheetah_task_env import HalfCheetahTaskEnv, HalfCheetahTaskConfig
from slbo.envs.mujoco.swimmer_task_env import SwimmerTaskEnv, SwimmerTaskConfig

envs = {
    'HalfCheetah-v2': HalfCheetahEnv,
    'Walker2D-v2': Walker2DEnv,
    'Humanoid-v2': HumanoidEnv,
    'Ant-v2': AntEnv,
    'Hopper-v2': HopperEnv,
    'Swimmer-v2': SwimmerEnv,
}
task_envs = {
    'AntTask-v2': AntTaskEnv,
    'AntTask2D-v2': Ant2DTaskEnv,
    'HalfCheetahTask-v2': HalfCheetahTaskEnv,
    'SwimmerTask-v2': SwimmerTaskEnv,
}
task_configs = {
    'AntTask-v2': AntTaskConfig,
    'AntTask2D-v2': Ant2DTaskConfig,
    'HalfCheetahTask-v2': HalfCheetahTaskConfig,
    'SwimmerTask-v2': SwimmerTaskConfig,
}

for k in task_envs:
    assert k in task_configs

def make_env(id: str, task_config=None):
    if id in envs:
        env = envs[id]()
    elif id in task_envs:
        env = task_envs[id](task_config=task_config)
    else:
        raise Exception(f"env {id} not recognized")
        
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    return env

def make_task(id: str):
    if id not in task_envs:
        raise Exception(f"env {id} not recognized")

    return task_configs[id]()