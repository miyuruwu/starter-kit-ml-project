"""Export an SB3 residual PPO checkpoint into a self-contained Kaggle notebook."""

import argparse
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from textwrap import wrap

from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn

from smart_buildings.team.rl.action_adapter import ActionAdapter
from smart_buildings.team.rl.inference import NumpyPPOActor
from smart_buildings.team.rl.observation_encoder import ObservationEncoder


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CELLS = (
    ("original-pid", "smart_buildings/baselines/pid.py"),
    ("observation-encoder", "smart_buildings/team/rl/observation_encoder.py"),
    ("action-adapter", "smart_buildings/team/rl/action_adapter.py"),
    ("numpy-actor", "smart_buildings/team/rl/inference.py"),
)


def export_actor(model):
    policy = model.policy
    if (
        not isinstance(policy.features_extractor, FlattenExtractor)
        or not isinstance(policy.action_dist, DiagGaussianDistribution)
        or policy.squash_output
        or not isinstance(model.observation_space, spaces.Box)
        or model.observation_space.shape != (ObservationEncoder.OBS_DIM,)
        or not isinstance(model.action_space, spaces.Box)
        or model.action_space.shape != (ActionAdapter.ACTION_DIM,)
        or not np.all(model.action_space.low == -1)
        or not np.all(model.action_space.high == 1)
    ):
        raise ValueError("Expected a flat observation and a clipped Gaussian PPO actor")

    layers = list(policy.mlp_extractor.policy_net)
    if (
        [type(layer) for layer in layers] != [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh]
        or type(policy.action_net) is not nn.Linear
    ):
        raise ValueError("Expected a PPO actor with two tanh hidden layers")

    occupancy_scale = float(model.occupancy_scale)
    if not np.isfinite(occupancy_scale) or occupancy_scale <= 0:
        raise ValueError("Checkpoint occupancy_scale must be finite and positive")

    arrays = {"occupancy_scale": np.asarray(occupancy_scale)}
    for i, layer in enumerate((layers[0], layers[2], policy.action_net)):
        arrays[f"weight_{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"bias_{i}"] = layer.bias.detach().cpu().numpy()

    archive = BytesIO()
    np.savez_compressed(archive, **arrays)
    actor = NumpyPPOActor.load(BytesIO(archive.getvalue()))
    probe = np.random.default_rng(0).normal(size=(256, ObservationEncoder.OBS_DIM))
    expected, _ = model.predict(probe.astype(np.float32), deterministic=True)
    np.testing.assert_allclose(actor.predict(probe), expected, atol=2e-6, rtol=0)
    return archive.getvalue()


def cell(cell_id, source, *, kind="code"):
    result = {"cell_type": kind, "id": cell_id, "metadata": {},
              "source": source.splitlines(keepends=True)}
    if kind == "code":
        compile(source, cell_id, "exec")
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(checkpoint, output):
    checkpoint_bytes = checkpoint.read_bytes()
    digest = sha256(checkpoint_bytes).hexdigest()
    model = PPO.load(BytesIO(checkpoint_bytes), device="cpu")
    weights = base64.b64encode(export_actor(model)).decode("ascii")
    weight_literal = "\n".join(f"    {line!r}" for line in wrap(weights, 100))
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "cells": [
            cell(
                "overview",
                "# Smart Buildings Residual PPO\n\n"
                f"Deterministic PID + PPO from `{checkpoint.name}` "
                f"({model.num_timesteps:,} training steps, occupancy scale "
                f"{model.occupancy_scale:g}).\n\nCheckpoint SHA-256 `{digest}`.\n\n"
                "Attach the Smart Buildings competition input and run with CPU and Internet off.\n\n"
                "The PID implementation comes from the [official starter kit]"
                "(https://github.com/smart-buildings-2026/starter-kit) under Apache-2.0.\n",
                kind="markdown",
            ),
            *[
                cell(cell_id, (ROOT / path).read_text(encoding="utf-8"))
                for cell_id, path in SOURCE_CELLS
            ],
            cell(
                "checkpoint",
                "import base64\nfrom io import BytesIO\n\n"
                "# Compressed NumPy actor weights exported from the checkpoint above.\n"
                f"ACTOR = NumpyPPOActor.load(BytesIO(base64.b64decode(\n{weight_literal}\n)))\n",
            ),
            cell(
                "competition-policy",
                '''import os
from pathlib import Path
import sys

COMPETITION_DATA_FOLDER = Path(os.environ.get(
    "SMART_BUILDINGS_COMPETITION_DATA",
    "/kaggle/input/competitions/neurips-2026-smart-buildings",
))
sys.path.append(str(COMPETITION_DATA_FOLDER / "kaggle_evaluation"))
from kaggle_evaluation import smart_buildings_inference_server


class TransferPolicy(smart_buildings_inference_server.Policy):
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.pid = PIDPolicy(building_meta_data, action_spec)
        self.encoder = ObservationEncoder(
            building_meta_data, self.pid, occupancy_scale=ACTOR.occupancy_scale
        )
        self.adapter = ActionAdapter(building_meta_data)

    def get_action(self, obs):
        residual = ACTOR.predict(self.encoder.encode(obs))
        return self.adapter.combine(self.pid.get_action(obs), residual)
''',
            ),
            cell(
                "run-gateway",
                "smart_buildings_inference_server.SmartBuildingsInferenceServer(TransferPolicy).run(\n"
                "    competition_data_folder=str(COMPETITION_DATA_FOLDER),\n)\n",
            ),
            cell(
                "license",
                "## Starter kit license and notices\n\n```text\n"
                + (ROOT / "NOTICE").read_text(encoding="utf-8")
                + "\n"
                + (ROOT / "LICENSE").read_text(encoding="utf-8")
                + "```\n",
                kind="markdown",
            ),
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {output} from {model.num_timesteps:,} steps (SHA-256 {digest})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trusted SB3 checkpoint to export")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/kaggle/submission_ppo/submit_ppo_kaggle.ipynb")
    args = parser.parse_args()
    build_notebook(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
