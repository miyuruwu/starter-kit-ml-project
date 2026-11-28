
from pathlib import Path
from pprint import pprint

import numpy as np

from smart_buildings.evaluate import (
    create_building_meta_data,
    load_hybrid_environment,
)


CONFIG_PATH = Path(
    "smart_buildings/training_scenarios/"
    "building_train_1/fairbanks_1/config.gin"
)


def main():
    print("Loading environment...")

    env = load_hybrid_environment(CONFIG_PATH)

    env._metrics_path = None
    env._occupancy_normalization_constant = 125.0

    metadata = create_building_meta_data(
        env,
        CONFIG_PATH.parent,
    )

    # 1. Inspect metadata
    print("\n=== METADATA KEYS ===")

    for key, value in metadata.items():
        print(f"{key}: {type(value).__name__}")

    # 2. Inspect action indices
    print("\n=== ACTION INDICES ===")

    keys = [
        "temp_indices",
        "pressure_indices",
        "recirculation_indices",
        "water_indices",
    ]

    for key in keys:
        print(f"\n{key}:")
        pprint(metadata.get(key))

    # 3. Reset environment
    result = env.reset()

    print("\n=== RESET RETURN TYPE ===")
    print(type(result))

    if isinstance(result, tuple):
        obs = result[0]
        print("Tuple length:", len(result))
    else:
        obs = result

    obs = np.asarray(obs)

    print("\n=== OBSERVATION SUMMARY ===")
    print("Shape:", obs.shape)
    print("Dtype:", obs.dtype)
    print("Min:", np.min(obs))
    print("Max:", np.max(obs))
    print("Mean:", np.mean(obs))
    print("All finite:", np.isfinite(obs).all())

    # 4. Inspect every observation field
    names = metadata["observation_field_names"]

    print("\n=== OBSERVATION FIELDS ===")

    for idx, (name, value) in enumerate(zip(names, obs)):
        print(f"[{idx:03d}] {str(name):60s} {value}")

    # 5. Test a valid action
    spec = env.action_spec()

    action = {
        "continuous_action": np.zeros(
            spec["continuous_action"].shape,
            dtype=np.float32,
        ),
        "discrete_action": np.ones(
            spec["discrete_action"].shape,
            dtype=np.int32,
        ),
    }

    next_obs, reward, terminated, truncated, info = (
        env.step(action)
    )

    print("\n=== STEP RESULT ===")
    print("Observation shape:", np.asarray(next_obs).shape)
    print("Reward:", reward)
    print("Reward type:", type(reward))
    print("Terminated:", terminated)
    print("Truncated:", truncated)

    print("\n=== INFO ===")
    pprint(info)

    # 6. Inspect zone temperature indices
    print("\n=== ZONE TEMPERATURE INDICES ===")
    pprint(metadata.get("zone_temp_indices"))

    if callable(getattr(env, "close", None)):
        env.close()


if __name__ == "__main__":
    main()
