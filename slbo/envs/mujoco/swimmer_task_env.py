# Copy of swimmer_env.py with different mb_step.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from rllab.envs.mujoco import swimmer_task_env
from slbo.envs import BaseModelBasedEnv


class SwimmerTaskEnv(swimmer_task_env.SwimmerTaskEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 5
            self.model.data.qvel.flat,  # 5
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso"),  # 3
        ]).reshape(-1)

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(actions / scaling), axis=-1)
        forward_reward = -1.5 * np.abs(next_states[:, -3] - self._task_config.goal_velocity)
        reward = forward_reward - ctrl_cost
        return reward, np.zeros_like(reward, dtype=np.bool)
