
import numpy as np


class ActionAdapter:
    ACTION_DIM = 5

    def __init__(self, metadata):
        self.metadata = metadata

        self.temp_indices = np.asarray(
            metadata["temp_indices"], dtype=int
        )

        self.recirc_indices = np.asarray(
            metadata["recirculation_indices"], dtype=int
        )

        self.water_temp_indices = np.asarray(
            metadata["water_indices"], dtype=int
        )

        pressure_indices = np.asarray(
            metadata["pressure_indices"], dtype=int
        )

        n_ahu = len(self.temp_indices)

        # Metadata layout observed in the starter kit:
        # AHU pressure indices followed by HWS pressure.
        if len(pressure_indices) != n_ahu + 1:
            raise ValueError(
                "Unexpected pressure index layout"
            )

        self.ahu_pressure_indices = pressure_indices[:n_ahu]
        self.water_pressure_indices = pressure_indices[n_ahu:]

        self.groups = [
            self.temp_indices,
            self.ahu_pressure_indices,
            self.recirc_indices,
            self.water_temp_indices,
            self.water_pressure_indices,
        ]

        self.scales = np.array([
            0.15,
            0.20,
            0.20,
            0.15,
            0.20,
        ], dtype=np.float32)

    def combine(self, base_action, rl_action):
        rl_action = np.asarray(
            rl_action,
            dtype=np.float32,
        )

        if rl_action.shape != (self.ACTION_DIM,):
            raise ValueError(
                f"Invalid RL action shape: {rl_action.shape}"
            )

        if not np.isfinite(rl_action).all():
            raise ValueError("Non-finite RL action")

        rl_action = np.clip(rl_action, -1.0, 1.0)

        continuous = np.array(
            base_action["continuous_action"],
            dtype=np.float32,
            copy=True,
        )

        discrete = np.array(
            base_action["discrete_action"],
            dtype=np.int32,
            copy=True,
        )

        # Apply one correction to each actuator group.
        for i, indices in enumerate(self.groups):
            continuous[indices] += (
                self.scales[i] * rl_action[i]
            )

        continuous = np.clip(
            continuous, -1.0, 1.0
        ).astype(np.float32)

        return {
            "continuous_action": continuous,
            "discrete_action": discrete,
        }
