import os
import json
import datetime
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import metaworld
from metaworld.policies import ENV_POLICY_MAP

# Create results directory
RESULTS_DIR = "results"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(RESULTS_DIR, TIMESTAMP)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "videos"), exist_ok=True)


def evaluate_single_task(
    task_name, env_cls, task, num_episodes=10, render_video=False, reward_function_version="v2"
):
    """Evaluate a single task and return results"""
    # Create environment with render mode if recording video
    if render_video:
        env = env_cls(render_mode="rgb_array", reward_function_version=reward_function_version)
    else:
        env = env_cls(reward_function_version=reward_function_version)
    
    env.set_task(task)
    
    # Get scripted policy (if available)
    policy = None
    # if task_name in ENV_POLICY_MAP:
    #     policy = ENV_POLICY_MAP[task_name]()
    
    episode_rewards = []
    episode_successes = []
    episode_lengths = []
    frames = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        success = False
        steps = 0
        
        for step in range(500):
            # Use scripted policy or random action
            if policy is not None:
                action = policy.get_action(obs)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Record video for the first episode only
            if render_video and episode == 0:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    print(f"Warning: Failed to render frame: {e}")
                    render_video = False  # Disable further rendering
            
            if info.get("success", False):
                success = True
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_lengths.append(steps)
    
    env.close()
    
    return {
        "task_name": task_name,
        "rewards": episode_rewards,
        "successes": episode_successes,
        "lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "success_rate": np.mean(episode_successes),
        "mean_length": np.mean(episode_lengths),
        "frames": frames,
    }


def save_video(frames, filepath, fps=30):
    """Save video from frames"""
    if len(frames) == 0:
        print(f"No frames to save: {filepath}")
        return
    
    try:
        import imageio
        imageio.mimsave(filepath, frames, fps=fps)
        print(f"Video saved: {filepath}")
    except ImportError:
        print("Please install imageio: pip install imageio imageio-ffmpeg")
    except Exception as e:
        print(f"Failed to save video: {e}")


def save_results_text(results, filepath):
    """Save results as text file"""
    with open(filepath, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MetaWorld Benchmark Evaluation Results\n")
        f.write(f"Timestamp: {TIMESTAMP}\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Task: {result['task_name']}\n")
            f.write(f"  Mean Reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}\n")
            f.write(f"  Success Rate: {result['success_rate']:.2%}\n")
            f.write(f"  Mean Episode Length: {result['mean_length']:.1f}\n")
            f.write("-" * 40 + "\n")
        
        # Summary
        f.write("\n" + "=" * 60 + "\n")
        f.write("Summary\n")
        f.write("=" * 60 + "\n")
        mean_success = np.mean([r["success_rate"] for r in results])
        mean_reward = np.mean([r["mean_reward"] for r in results])
        f.write(f"Overall Success Rate: {mean_success:.2%}\n")
        f.write(f"Overall Mean Reward: {mean_reward:.2f}\n")
    
    print(f"Text results saved: {filepath}")


def save_results_json(results, filepath):
    """Save results as JSON file"""
    # Convert numpy types to Python native types
    results_clean = []
    for r in results:
        r_clean = {
            "task_name": r["task_name"],
            "rewards": [float(x) for x in r["rewards"]],
            "successes": [bool(x) for x in r["successes"]],
            "lengths": [int(x) for x in r["lengths"]],
            "mean_reward": float(r["mean_reward"]),
            "std_reward": float(r["std_reward"]),
            "success_rate": float(r["success_rate"]),
            "mean_length": float(r["mean_length"]),
        }
        results_clean.append(r_clean)
    
    with open(filepath, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"JSON results saved: {filepath}")


def plot_results(results, filepath):
    """Plot and save results charts"""
    task_names = [r["task_name"] for r in results]
    success_rates = [r["success_rate"] for r in results]
    mean_rewards = [r["mean_reward"] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Success rate bar chart
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(task_names)), success_rates, color="steelblue")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate per Task")
    ax1.set_xticks(range(len(task_names)))
    ax1.set_xticklabels(task_names, rotation=45, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.8, color="r", linestyle="--", label="80% Threshold")
    ax1.legend()
    
    # Add value labels
    for bar, rate in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f"{rate:.0%}", ha="center", va="bottom", fontsize=8)
    
    # Mean reward bar chart
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(task_names)), mean_rewards, color="forestgreen")
    ax2.set_xlabel("Task")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Mean Reward per Task")
    ax2.set_xticks(range(len(task_names)))
    ax2.set_xticklabels(task_names, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Chart saved: {filepath}")


def plot_reward_curves(results, filepath):
    """Plot reward curves for all tasks"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for result in results:
        ax.plot(result["rewards"], label=result["task_name"], marker="o", markersize=4)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Curves per Task")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Reward curves saved: {filepath}")


def main():
    print("=" * 60)
    print("MetaWorld Benchmark Evaluation")
    print(f"Results will be saved to: {SAVE_DIR}")
    print("=" * 60)
    
    # Select benchmark: MT10, MT50, ML10, ML45
    BENCHMARK = "MT10"
    NUM_EPISODES = 10
    RECORD_VIDEO = True  # Whether to record videos
    
    print(f"\nUsing benchmark: {BENCHMARK}")
    print(f"Evaluating {NUM_EPISODES} episodes per task")
    print(f"Record video: {RECORD_VIDEO}")
    
    # Install video dependencies
    if RECORD_VIDEO:
        try:
            import imageio
        except ImportError:
            print("\nInstalling video dependencies...")
            os.system("pip install imageio imageio-ffmpeg")
    
    # Load benchmark
    if BENCHMARK == "MT10":
        benchmark = metaworld.MT10()
        classes = benchmark.train_classes
        tasks = benchmark.train_tasks
    elif BENCHMARK == "MT50":
        benchmark = metaworld.MT50()
        classes = benchmark.train_classes
        tasks = benchmark.train_tasks
    elif BENCHMARK == "ML10":
        benchmark = metaworld.ML10()
        classes = benchmark.test_classes
        tasks = benchmark.test_tasks
    elif BENCHMARK == "ML45":
        benchmark = metaworld.ML45()
        classes = benchmark.test_classes
        tasks = benchmark.test_tasks
    else:
        raise ValueError(f"Unknown benchmark: {BENCHMARK}")
    
    results = []
    
    REWARD_VERSION = "v1"  # any string != "v2" will trigger the else-branch

    for task_name, env_cls in classes.items():
        print(f"\nEvaluating task: {task_name}")
        
        # Get corresponding task
        task_matches = [t for t in tasks if t.env_name == task_name]
        if not task_matches:
            raise ValueError(f"No task found for env_name={task_name}. tasks size={len(tasks)}")
        task = task_matches[0]
        
        # Evaluate task
        result = evaluate_single_task(
            task_name,
            env_cls,
            task,  # <-- use the matched task object (NOT tasks[task_name])
            num_episodes=NUM_EPISODES,
            render_video=RECORD_VIDEO,
            reward_function_version=REWARD_VERSION,
        )
        results.append(result)

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    
    save_results_text(results, os.path.join(SAVE_DIR, "results.txt"))
    save_results_json(results, os.path.join(SAVE_DIR, "results.json"))
    plot_results(results, os.path.join(SAVE_DIR, "success_rate_plot.png"))
    plot_reward_curves(results, os.path.join(SAVE_DIR, "reward_curves.png"))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    mean_success = np.mean([r["success_rate"] for r in results])
    mean_reward = np.mean([r["mean_reward"] for r in results])
    print(f"Overall Success Rate: {mean_success:.2%}")
    print(f"Overall Mean Reward: {mean_reward:.2f}")
    print(f"\nResults saved to: {SAVE_DIR}")
    print(f"  - results.txt (Text report)")
    print(f"  - results.json (JSON data)")
    print(f"  - success_rate_plot.png (Success rate chart)")
    print(f"  - reward_curves.png (Reward curves)")
    if RECORD_VIDEO:
        print(f"  - videos/ (Task videos)")


if __name__ == "__main__":
    main()