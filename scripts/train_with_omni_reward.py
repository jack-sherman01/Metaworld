"""
Example script for training SawyerPushEnvV3 with omni_reward.
"""
import sys
sys.path.insert(0, "/home/hzhang/heng/omniR/omni_reward")
sys.path.insert(0, "/home/hzhang/heng/omniR/Metaworld")

import os
import argparse
import gymnasium as gym
import numpy as np
from metaworld.envs.sawyer_push_v3 import SawyerPushEnvV3
import metaworld

from examples.env_wrapper import OmniRewardWrapper
from omni_reward.vision.captioner import VLMCaptioner
from omni_reward.vision.text_encoder import TextEncoder

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train with OmniReward")
parser.add_argument(
    "--provider",
    type=str,
    default="openai",
    choices=["openai", "gemini", "claude", "qwen"],
    help="VLM provider to use"
)
args = parser.parse_args()

# Get API key based on provider
if args.provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    env_var_name = "OPENAI_API_KEY"
elif args.provider == "gemini":
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    env_var_name = "GEMINI_API_KEY or GOOGLE_API_KEY"
elif args.provider == "claude":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    env_var_name = "ANTHROPIC_API_KEY"
elif args.provider == "qwen":
    api_key = os.getenv("DASHSCOPE_API_KEY")
    env_var_name = "DASHSCOPE_API_KEY"

if not api_key:
    raise ValueError(f"Please set the {env_var_name} environment variable")

print(f"Using VLM provider: {args.provider}")

# First create MT10 benchmark to get tasks
mt10 = metaworld.MT10()

# Get push-v3 task
push_tasks = [task for task in mt10.train_tasks if task.env_name == 'push-v3']
print(f"Number of push task variants: {len(push_tasks)}")  # Will be 50 variants

# Use only the first variant
if push_tasks:
    push_task = push_tasks[0]
    print(f"Using push task: {push_task.env_name}")

# Create base environment
base_env = SawyerPushEnvV3(
    render_mode="rgb_array",
    height=480,
    width=480,
)

# Must set task first!
if push_tasks:
    base_env.set_task(push_tasks[0])
else:
    print("No push task found, using first available task")
    base_env.set_task(mt10.train_tasks[0])

# Initialize omni_reward components with detailed captions
captioner = VLMCaptioner(
    provider=args.provider,  # Use selected provider
    template="detailed_state",  # Use detailed template
    max_tokens=1024,  # Allow longer responses
    api_key=api_key,
)
text_encoder = TextEncoder()

# Wrap with OmniRewardWrapper
goal_text = ("Push the puck to the red goal position. "
             "The robot arm should approach the puck and push it towards the goal.")

env = OmniRewardWrapper(
    env=base_env,
    captioner=captioner,
    text_encoder=text_encoder,
    goal_text=goal_text,
    use_subgoals=False,  # No subgoals for this test
    openai_api_key=api_key if args.provider == "openai" else None,
    task_name="push-v3",
    camera_name="corner2",  # Side view
)

# Test the environment
obs, info = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Image available: {'image' in info}")
print("="*80)

for step in range(500):  # Run 500 steps to test
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Print reward and subgoal info
    print(f"\n--- Step {step} ---")
    print(f"Reward: {reward:.4f}")
    
    # If omni_reward_info exists, print detailed info
    if 'omni_reward_info' in info:
        omni_info = info['omni_reward_info']
        print(f"Omni Reward Info:")
        print(f"  - Current subgoal index: {omni_info.get('current_subgoal_index', 'N/A')}")
        print(f"  - Current subgoal: {omni_info.get('current_subgoal', 'N/A')}")
        print(f"  - Subgoal reward: {omni_info.get('reward', 0.0):.4f}")
        print(f"  - Subgoal completed: {omni_info.get('subgoal_completed', False)}")
        print(f"  - All completed: {omni_info.get('all_completed', False)}")
        if 'all_subgoals' in omni_info:
            print(f"  - Total subgoals: {len(omni_info['all_subgoals'])}")
            print(f"  - All subgoals: {omni_info['all_subgoals']}")
    
    if terminated or truncated:
        print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
        obs, info = env.reset()
        print("="*80)

env.close()