"""
使用 omni_reward 训练 SawyerPushEnvV3 的示例脚本。
"""
import sys
sys.path.insert(0, "/home/hzhang/heng/omniR/omni_reward")
sys.path.insert(0, "/home/hzhang/heng/omniR/Metaworld")

import gymnasium as gym
import numpy as np
from metaworld.envs.sawyer_push_omni_reward import SawyerPushOmniRewardEnv
import metaworld

# 首先创建 MT10 benchmark 来获取任务
mt10 = metaworld.MT10()

# 获取 push-v3 任务
push_tasks = [task for task in mt10.train_tasks if task.env_name == 'push-v3']
print(f"Number of push tasks: {len(push_tasks)}")

if len(push_tasks) == 0:
    # 如果没有找到，打印所有可用的任务名称
    all_task_names = set(task.env_name for task in mt10.train_tasks)
    print(f"Available task names: {all_task_names}")
    # 尝试匹配包含 'push' 的任务
    push_tasks = [task for task in mt10.train_tasks if 'push' in task.env_name.lower()]
    print(f"Tasks containing 'push': {len(push_tasks)}")

# 创建使用 omni_reward 的环境
env = SawyerPushOmniRewardEnv(
    render_mode="rgb_array",
    task_description="Push the puck to the red goal position. "
                     "The robot arm should approach the puck and push it towards the goal.",
    use_original_reward=False,  # 只使用 omni_reward
    omni_reward_weight=1.0,
)

# 必须先设置任务！
if push_tasks:
    env.set_task(push_tasks[0])
else:
    # 如果还是找不到，使用第一个任务
    print("No push task found, using first available task")
    env.set_task(mt10.train_tasks[0])

# 测试环境
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Image available: {'image' in info}")

for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: reward={reward:.4f}, "
          f"omni_reward={info.get('omni_reward', 0):.4f}, "
          f"original_reward={info.get('original_reward', 0):.4f}")
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()