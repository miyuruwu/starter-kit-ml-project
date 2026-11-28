
import numpy as np

from stable_baselines3.common.env_checker import check_env

from smart_buildings.team.rl.gym_wrapper import HVACEnv


CONFIG_PATH = (
    "smart_buildings/training_scenarios/"
    "building_train_1/fairbanks_1/config.gin"
)


def main():
    env = HVACEnv(
        config_path=CONFIG_PATH,
        max_episode_steps=288,
    )

    try:
        obs, info = env.reset()

        print("\n=== ENVIRONMENT ===")
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Observation shape:", obs.shape)

        print("\n=== ENCODED FEATURES ===")

        for name, value in zip(
            env.encoder.FEATURE_NAMES,
            obs,
        ):
            print(f"{name:30s}: {value:.6f}")

        print("\n=== ZERO CORRECTION TEST ===")

        base_action = env.pid.get_action(env.raw_obs)

        zero_action = np.zeros(
            5,
            dtype=np.float32,
        )

        final_action = env.adapter.combine(
            base_action,
            zero_action,
        )

        np.testing.assert_allclose(
            base_action["continuous_action"],
            final_action["continuous_action"],
            atol=1e-6,
        )

        np.testing.assert_array_equal(
            base_action["discrete_action"],
            final_action["discrete_action"],
        )

        print("PASS: zero correction preserves PID action")

        print("\n=== CHECK_ENV ===")

        check_env(env, warn=True)

        print("PASS: Gymnasium environment is valid")

        print("\n=== PID EQUIVALENCE ROLLOUT ===")

        obs, info = env.reset()

        total_reward = 0.0

        for step in range(288):
            obs, reward, terminated, truncated, info = (
                env.step(zero_action)
            )

            total_reward += info["raw_reward"]

            if (step + 1) % 48 == 0:
                print(
                    f"Step {step + 1:3d}: "
                    f"total_reward={total_reward:.6f}"
                )

            if terminated or truncated:
                break

        print("\n=== RESULT ===")
        print("Total steps:", step + 1)
        print("Total raw reward:", total_reward)

    finally:
        env.close()


if __name__ == "__main__":
    main()
