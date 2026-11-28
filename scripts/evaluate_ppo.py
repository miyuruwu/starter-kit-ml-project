
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from smart_buildings.team.rl.gym_wrapper import HVACEnv


def evaluate(config_path, steps, model=None, *, occupancy_scale=None):
    if occupancy_scale is None:
        occupancy_scale = getattr(model, "occupancy_scale", 1.0)

    env = HVACEnv(
        config_path=config_path,
        max_episode_steps=steps,
        reward_scale=1.0,
        occupancy_scale=occupancy_scale,
    )

    records = []
    total_reward = 0.0

    try:
        obs, info = env.reset(seed=42)

        start_time = time.perf_counter()

        for step in range(steps):

            if model is None:
                # Zero residual = original PID
                action = np.zeros(
                    env.action_space.shape,
                    dtype=np.float32,
                )
            else:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            raw_reward = float(info["raw_reward"])
            total_reward += raw_reward

            records.append({
                "step": step + 1,
                "reward": raw_reward,
                "cumulative_reward": total_reward,
                **{
                    f"action_{i}": float(value)
                    for i, value in enumerate(action)
                },
                **{
                    name: float(value)
                    for name, value in zip(
                        env.encoder.FEATURE_NAMES, obs
                    )
                },
            })

            if (step + 1) % 288 == 0:
                print(
                    f"Day {(step + 1) // 288}: "
                    f"reward={total_reward:.6f}"
                )

            if terminated or truncated:
                break

        elapsed = time.perf_counter() - start_time

        summary = {
            "total_reward": total_reward,
            "mean_reward": (
                total_reward / len(records)
            ),
            "total_steps": len(records),
            "elapsed_seconds": elapsed,
            "fps": len(records) / max(elapsed, 1e-9),
        }

        return summary, pd.DataFrame(records)

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--building",
        default="building_train_1",
    )

    parser.add_argument(
        "--weather",
        default="fairbanks_1",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=288,
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/smoke_test/final_model.zip",
    )

    parser.add_argument(
        "--occupancy-scale",
        type=float,
        help="Require the checkpoint to use this occupancy scale.",
    )

    parser.add_argument(
        "--name",
        default="smoke_test",
    )

    args = parser.parse_args()

    config_path = (
        Path("smart_buildings/training_scenarios")
        / args.building
        / args.weather
        / "config.gin"
    )

    output_dir = (
        Path("results/rl")
        / args.name
        / args.building
        / args.weather
    )

    model = PPO.load(
        args.checkpoint,
        device="cpu",
    )
    occupancy_scale = getattr(model, "occupancy_scale", 1.0)
    if args.occupancy_scale is not None and args.occupancy_scale != occupancy_scale:
        parser.error(
            f"--occupancy-scale must match the checkpoint value {occupancy_scale:g}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n=== EVALUATING PID ===")

    pid_summary, pid_df = evaluate(
        config_path,
        args.steps,
        occupancy_scale=occupancy_scale,
    )

    print("\n=== EVALUATING PPO ===")

    ppo_summary, ppo_df = evaluate(
        config_path,
        args.steps,
        model=model,
    )

    pid_df.to_csv(
        output_dir / "pid_timeseries.csv",
        index=False,
    )

    ppo_df.to_csv(
        output_dir / "ppo_timeseries.csv",
        index=False,
    )

    improvement = (
        ppo_summary["total_reward"]
        - pid_summary["total_reward"]
    )

    comparison = {
        "building": args.building,
        "weather": args.weather,
        "checkpoint": args.checkpoint,
        "occupancy_scale": occupancy_scale,
        "pid": pid_summary,
        "ppo": ppo_summary,
        "reward_improvement": improvement,
    }

    with open(
        output_dir / "comparison.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(comparison, f, indent=2)

    print("\n=== FINAL COMPARISON ===")

    print(
        f"PID reward: "
        f"{pid_summary['total_reward']:.6f}"
    )

    print(
        f"PPO reward: "
        f"{ppo_summary['total_reward']:.6f}"
    )

    print(
        f"Reward improvement: "
        f"{improvement:+.6f}"
    )

    if (
        pid_summary["total_steps"]
        != ppo_summary["total_steps"]
    ):
        print("WARNING: Rollout lengths differ.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
