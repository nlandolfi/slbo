# Copy of half_cheetah_env.py with different mb_step.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from rllab.envs.mujoco import half_cheetah_task_env
from slbo.envs import BaseModelBasedEnv

HalfCheetahTaskConfig = half_cheetah_task_env.HalfCheetahTaskConfig

class HalfCheetahTaskEnv(half_cheetah_task_env.HalfCheetahTaskEnv, BaseModelBasedEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 9
            self.model.data.qvel.flat,  # 9
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def mb_step(self, states, actions, next_states):
        actions = np.clip(actions, *self.action_bounds)
        reward_ctrl = -0.05 * np.sum(np.square(actions), axis=-1)
        if self._task_config.goal_velocity == -math.inf:
            reward_fwd = -1 * next_states[..., 21]
        elif self._task_config.goal_velocity == math.inf:
            reward_fwd = next_states[..., 21]
        else:
            reward_fwd = -1. * np.abs(next_states[..., 21] - self._task_config.goal_velocity)
        return reward_ctrl + reward_fwd, np.zeros_like(reward_fwd, dtype=np.bool)
