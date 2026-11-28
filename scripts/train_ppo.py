
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)

from smart_buildings.team.rl.gym_wrapper import HVACEnv


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
        "--timesteps",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--episode-steps",
        type=int,
        default=8640,
    )

    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--occupancy-scale",
        type=float,
        default=1.0,
        help="Divide the occupancy feature by this positive value.",
    )

    parser.add_argument(
        "--name",
        default="ppo_v1",
    )

    args = parser.parse_args()

    config_path = (
        Path("smart_buildings/training_scenarios")
        / args.building
        / args.weather
        / "config.gin"
    )

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    run_dir = Path("runs") / args.name
    checkpoint_dir = Path("checkpoints") / args.name

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        HVACEnv(
            config_path=config_path,
            max_episode_steps=args.episode_steps,
            reward_scale=args.reward_scale,
            occupancy_scale=args.occupancy_scale,
        ),
        filename=str(run_dir / "monitor.csv"),
    )

    callback = CheckpointCallback(
        save_freq=2048,
        save_path=str(checkpoint_dir),
        name_prefix="ppo",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,

        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,

        policy_kwargs={
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 64],
            },
            "log_std_init": -1.0,
        },

        seed=42,
        device="cpu",
        verbose=1,
    )
    model.occupancy_scale = args.occupancy_scale

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            log_interval=1,
        )

        model.save(
            str(checkpoint_dir / "final_model")
        )

        print("\nTraining completed.")

        print(
            "Model saved to:",
            checkpoint_dir / "final_model.zip",
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
