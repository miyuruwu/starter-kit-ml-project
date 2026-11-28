"""Run the exported deterministic PPO actor with NumPy."""

import numpy as np


class NumpyPPOActor:
    @classmethod
    def load(cls, source):
        with np.load(source, allow_pickle=False) as archive:
            layers = tuple(
                (archive[f"weight_{i}"], archive[f"bias_{i}"])
                for i in range(3)
            )
            return cls(layers, float(archive["occupancy_scale"].item()))

    def __init__(self, layers, occupancy_scale):
        self.layers = layers
        self.occupancy_scale = occupancy_scale

    def predict(self, observation):
        features = np.asarray(observation, dtype=np.float32)
        for weight, bias in self.layers[:-1]:
            features = np.tanh(features @ weight.T + bias)
        weight, bias = self.layers[-1]
        return np.clip(features @ weight.T + bias, -1.0, 1.0)
