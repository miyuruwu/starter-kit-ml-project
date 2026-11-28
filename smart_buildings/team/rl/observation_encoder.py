
import numpy as np


class ObservationEncoder:
    OBS_DIM = 24

    FEATURE_NAMES = (
        "zone_temp_mean",
        "zone_temp_min",
        "zone_temp_max",
        "zone_temp_std",
        "heating_error_mean",
        "heating_error_max",
        "cooling_error_mean",
        "cooling_error_max",
        "cold_zone_ratio",
        "hot_zone_ratio",
        "outside_temperature",
        "occupancy",
        "comfort_now",
        "comfort_soon",
        "hod_sin",
        "hod_cos",
        "dow_sin",
        "dow_cos",
        "mean_damper",
        "mean_reheat",
        "water_temperature",
        "mean_static_pressure",
        "ahu_running_ratio",
        "mean_recirculation",
    )

    def __init__(self, metadata, pid_policy, *, occupancy_scale=1.0):
        if not np.isfinite(occupancy_scale) or occupancy_scale <= 0:
            raise ValueError("occupancy_scale must be finite and positive")

        self.occupancy_scale = occupancy_scale
        self.metadata = metadata

        self.names = metadata["observation_field_names"]
        self.index = {
            name: i
            for i, name in enumerate(self.names)
        }

        # Reuse the verified PID normalization constants.
        self.temp_mean = (
            pid_policy.zone_air_temperature_sensor_mean
        )

        self.temp_std = (
            pid_policy.zone_air_temperature_sensor_std
        )

        self.zone_indices = []
        self.heating_indices = []
        self.cooling_indices = []

        # Match all three temperature fields by VAV ID.
        for name, idx in self.index.items():
            if not name.endswith(
                "_zone_air_temperature_sensor"
            ):
                continue

            prefix = name.removesuffix(
                "_zone_air_temperature_sensor"
            )

            self.zone_indices.append(idx)

            self.heating_indices.append(
                self.index[
                    prefix
                    + "_zone_air_heating_temperature_setpoint_sensor"
                ]
            )

            self.cooling_indices.append(
                self.index[
                    prefix
                    + "_zone_air_cooling_temperature_setpoint_sensor"
                ]
            )

        if not self.zone_indices:
            raise ValueError("No zone temperature sensors found")

        self.outside_indices = self._find(
            "outside_air_temperature_sensor"
        )

        self.damper_indices = self._find(
            "supply_air_damper_percentage_command"
        )

        self.reheat_indices = self._find(
            "reheat_valve_percentage_command"
        )

        self.pressure_indices = self._find(
            "supply_air_static_pressure_sensor"
        )

        self.recirc_indices = self._find(
            "recirculation_percentage_command"
        )

        self.ahu_status_indices = [
            i for name, i in self.index.items()
            if name.startswith("ahs_ahu_")
            and name.endswith("supervisor_run_command")
        ]

    def _find(self, suffix):
        indices = [
            i for name, i in self.index.items()
            if name.endswith(suffix)
        ]

        if not indices:
            raise ValueError(f"Missing observation field: {suffix}")

        return indices

    def encode(self, obs):
        obs = np.asarray(obs, dtype=np.float32)

        # Convert normalized zone temperatures to Kelvin.
        zone_temp = (
            obs[self.zone_indices] * self.temp_std
            + self.temp_mean
        )

        heating_sp = obs[self.heating_indices]
        cooling_sp = obs[self.cooling_indices]

        heating_error = np.maximum(
            heating_sp - zone_temp, 0.0
        )

        cooling_error = np.maximum(
            zone_temp - cooling_sp, 0.0
        )

        outdoor = np.mean(obs[self.outside_indices])

        features = np.array([
            # Zone temperature statistics
            (np.mean(zone_temp) - 295.0) / 10.0,
            (np.min(zone_temp) - 295.0) / 10.0,
            (np.max(zone_temp) - 295.0) / 10.0,
            np.std(zone_temp) / 10.0,

            # Heating and cooling demand
            np.mean(heating_error) / 10.0,
            np.max(heating_error) / 10.0,
            np.mean(cooling_error) / 10.0,
            np.max(cooling_error) / 10.0,

            np.mean(heating_error > 0),
            np.mean(cooling_error > 0),

            # Outdoor temperature
            (outdoor - 293.15) / 30.0,

            # Occupancy and comfort mode
            obs[self.index["num_occupants"]] / self.occupancy_scale,
            obs[self.index["comfort_mode_now"]],
            obs[self.index["comfort_mode_soon"]],

            # Cyclic time
            obs[self.index["hod_sin_000"]],
            obs[self.index["hod_cos_000"]],
            obs[self.index["dow_sin_000"]],
            obs[self.index["dow_cos_000"]],

            # Zone actuator states
            np.mean(obs[self.damper_indices]),
            np.mean(obs[self.reheat_indices]),

            # Plant states
            (
                obs[
                    self.index[
                        "hws_supply_water_temperature_sensor"
                    ]
                ] - 330.0
            ) / 20.0,

            np.mean(obs[self.pressure_indices]) / 2000.0,
            np.mean(obs[self.ahu_status_indices]),
            np.mean(obs[self.recirc_indices]),

        ], dtype=np.float32)

        if features.shape != (self.OBS_DIM,):
            raise ValueError(
                f"Unexpected feature shape: {features.shape}"
            )

        if not np.isfinite(features).all():
            raise ValueError("Non-finite observation detected")

        return features
