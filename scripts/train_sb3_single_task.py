import argparse
import os
import datetime
import json

import metaworld
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym


class SuccessInfoWrapper(gym.Wrapper):
    """
    MetaWorld 用 info["success"]。
    SB3 的 EvalCallback/Robotics 习惯用 info["is_success"] 来统计成功率。
    这里做一个兼容映射。
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "is_success" not in info:
            info["is_success"] = bool(info.get("success", False))
        return obs, reward, terminated, truncated, info


def make_env(env_cls, task, seed: int, reward_version: str):
    def _thunk():
        env = env_cls(reward_function_version=reward_version)
        env.set_task(task)
        env.reset(seed=seed)
        env = SuccessInfoWrapper(env)
        # 让 Monitor 把 success/is_success 也记录进 csv（每个 episode 结束时）
        env = Monitor(env, info_keywords=("success", "is_success"))
        return env
    return _thunk


def evaluate_success_rate(env_cls, task, model, reward_version: str, episodes: int = 20, seed: int = 0) -> float:
    env = env_cls(reward_function_version=reward_version)
    env.set_task(task)
    env = SuccessInfoWrapper(env)

    successes = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        success = False
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            success = success or bool(info.get("is_success", False))
            if terminated or truncated:
                break
        successes.append(success)

    env.close()
    return float(sum(successes) / len(successes))


def train_one_task(classes, tasks, args, task_name: str) -> str:
    if task_name not in classes:
        raise ValueError(f"Unknown task: {task_name}")

    env_cls = classes[task_name]
    task_matches = [t for t in tasks if t.env_name == task_name]
    if not task_matches:
        raise ValueError(f"No task object for env_name={task_name}")
    task = task_matches[0]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.save_root, args.benchmark, task_name, f"{args.algo}_{args.reward_version}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "run_args.json"), "w") as f:
        d = vars(args).copy()
        d["resolved_task"] = task_name
        json.dump(d, f, indent=2)

    env_fns = [make_env(env_cls, task, seed=args.seed + i, reward_version=args.reward_version) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, filename=os.path.join(out_dir, "monitor.csv"))

    eval_env = DummyVecEnv([make_env(env_cls, task, seed=args.seed + 10_000, reward_version=args.reward_version)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(out_dir, "eval_monitor.csv"))

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // max(args.n_envs, 1),
        save_path=os.path.join(out_dir, "checkpoints"),
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(out_dir, "best_model"),
        log_path=os.path.join(out_dir, "eval_logs"),
        eval_freq=50_000 // max(args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            vec_env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=os.path.join(out_dir, "tb"),
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=os.path.join(out_dir, "tb"),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    model_path = os.path.join(out_dir, "model.zip")
    model.save(model_path)

    # 训练完保存“最终 success rate”（建议用 best_model 更有代表性）
    final_success_rate = evaluate_success_rate(
        env_cls=env_cls,
        task=task,
        model=model,
        reward_version=args.reward_version,
        episodes=20,
        seed=args.seed + 20_000,
    )
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(
            {
                "task_name": task_name,
                "algo": args.algo,
                "reward_version": args.reward_version,
                "final_success_rate_20ep": final_success_rate,
            },
            f,
            indent=2,
        )

    vec_env.close()
    eval_env.close()
    return model_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="MT10", choices=["MT10", "MT50", "ML10", "ML45"])
    p.add_argument("--task", default="reach-v3")
    p.add_argument("--all-tasks", action="store_true", help="Train all tasks in the benchmark (one model per task).")
    p.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reward-version", default="v2")
    p.add_argument("--save-root", default="trained_models")
    args = p.parse_args()

    if args.benchmark == "MT10":
        bench = metaworld.MT10()
        classes, tasks = bench.train_classes, bench.train_tasks
    elif args.benchmark == "MT50":
        bench = metaworld.MT50()
        classes, tasks = bench.train_classes, bench.train_tasks
    elif args.benchmark == "ML10":
        bench = metaworld.ML10()
        classes, tasks = bench.test_classes, bench.test_tasks
    else:
        bench = metaworld.ML45()
        classes, tasks = bench.test_classes, bench.test_tasks

    if args.all_tasks:
        for task_name in classes.keys():
            model_path = train_one_task(classes, tasks, args, task_name)
            print(f"[DONE] {task_name} -> {model_path}")
    else:
        model_path = train_one_task(classes, tasks, args, args.task)
        print(model_path)


if __name__ == "__main__":
    main()