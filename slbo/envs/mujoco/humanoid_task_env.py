# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import numpy as np
from rllab.envs.mujoco import humanoid_task_env
from rllab.envs.base import Step
from slbo.envs import BaseModelBasedEnv

HumanoidTaskConfig = humanoid_task_env.HumanoidTaskConfig

class HumanoidTaskEnv(humanoid_task_env.HumanoidTaskEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        data = self.model.data
        return np.concatenate([
            data.qpos.flat,  # 17
            data.qvel.flat,  # 16
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def step(self, action):
        self.forward_dynamics(action)
        alive_bonus = self.alive_bonus
        data = self.model.data

        comvel = self.get_body_comvel("torso")
        if self._task_config.goal_velocity == -math.inf:
            lin_vel_reward = -comvel[0]
        elif self._task_config.goal_velocity == math.inf:
            lin_vel_reward = comvel[0]
        else:
            lin_vel_reward = -np.abs(comvel[0] - self._task_config.goal_velocity)        
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = 0.
        vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))
        reward = lin_vel_reward + #alive_bonus - ctrl_cost - \
            #impact_cost - vel_deviation_cost
        pos = data.qpos.flat[2]
        done = pos < 0.8 or pos > 2.0

        next_obs = self.get_current_obs()
        return Step(next_obs, reward, done)

    def mb_step(self, states, actions, next_states):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5

        alive_bonus = 0.2
        if self._task_config.goal_velocity == -math.inf:
            lin_vel_reward = -next_states[:, 36]
        elif self._task_config.goal_velocity == math.inf:
            lin_vel_reward = next_states[:, 36]
        else:
            lin_vel_reward = -np.abs(next_states[:, 36] - self._task_config.goal_velocity)  
        ctrl_cost = 5.e-4 * np.square(actions / scaling).sum(axis=1)
        impact_cost = 0.
        vel_deviation_cost = 5.e-3 * np.square(next_states[:, 37:39]).sum(axis=1)
        reward = lin_vel_reward + #alive_bonus - ctrl_cost - impact_cost - vel_deviation_cost

        dones = (next_states[:, 2] < 0.8) | (next_states[:, 2] > 2.0)
        return reward, dones