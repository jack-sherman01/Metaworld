from __future__ import annotations

from typing import Any
import numpy as np
import numpy.typing as npt
import gymnasium as gym
import os

from metaworld.envs.sawyer_push_v3 import SawyerPushEnvV3
from metaworld.sawyer_xyz_env import RenderMode

# 导入 omni_reward
import sys
sys.path.insert(0, "/home/hzhang/heng/omniR/omni_reward")
from omni_reward.reward.interface import OmniRewardInterface
from omni_reward.vision.captioner import VLMCaptioner
from omni_reward.vision.text_encoder import TextEncoder


class SawyerPushOmniRewardEnv(gym.Wrapper):
    """
    SawyerPushEnvV3 的包装器，使用 omni_reward 替换原始奖励函数。
    """
    
    def __init__(
        self,
        render_mode: RenderMode | None = "rgb_array",
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        task_description: str = "Push the puck to the red goal position",
        use_original_reward: bool = False,
        omni_reward_weight: float = 1.0,
        original_reward_weight: float = 0.0,
        # omni_reward 相关参数
        alpha: float = 0.6,
        lambda_: float = 1.0,
        # API 密钥
        openai_api_key: str | None = None,
    ) -> None:
        # 设置 API 密钥
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 创建基础环境
        env = SawyerPushEnvV3(
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
            reward_function_version="v2",
        )
        super().__init__(env)
        
        self.task_description = task_description
        self.use_original_reward = use_original_reward
        self.omni_reward_weight = omni_reward_weight
        self.original_reward_weight = original_reward_weight
        
        # 初始化 captioner 和 text_encoder
        self.captioner = VLMCaptioner()
        self.text_encoder = TextEncoder()
        
        # 初始化 omni_reward 接口
        self.omni_reward = OmniRewardInterface(
            captioner=self.captioner,
            text_encoder=self.text_encoder,
            alpha=alpha,
            lambda_=lambda_,
            store_history=True,
        )
        
        self._prev_image: npt.NDArray[np.uint8] | None = None
        self._timestep = 0
        
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_image = self._get_current_image()
        self._timestep = 0
        
        # 使用 start_episode 初始化目标和基线
        self.omni_reward.start_episode(
            goal_text=self.task_description,
            initial_image=self._prev_image,
        )
        
        if self._prev_image is not None:
            info["image"] = self._prev_image
        return obs, info
    
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        current_image = self._get_current_image()
        self._timestep += 1
        
        omni_reward_value = 0.0
        if current_image is not None:
            try:
                # 不再传递 goal_text，因为已经在 start_episode 中设置
                omni_reward_value = self.omni_reward.compute_reward(
                    scene_image=current_image,
                    timestep=self._timestep,
                )
            except Exception as e:
                print(f"Warning: omni_reward computation failed: {e}")
                omni_reward_value = 0.0
        
        if self.use_original_reward:
            final_reward = (
                self.omni_reward_weight * omni_reward_value +
                self.original_reward_weight * original_reward
            )
        else:
            final_reward = self.omni_reward_weight * omni_reward_value
        
        info["original_reward"] = original_reward
        info["omni_reward"] = omni_reward_value
        info["combined_reward"] = final_reward
        if current_image is not None:
            info["image"] = current_image
        
        self._prev_image = current_image
        return obs, final_reward, terminated, truncated, info
    
    def _get_current_image(self) -> npt.NDArray[np.uint8] | None:
        if self.env.render_mode == "rgb_array":
            return self.env.render()
        return None
    
    def set_task(self, task) -> None:
        self.env.set_task(task)