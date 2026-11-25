"""Package the local RL source and shipped training scenarios for Kaggle."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ("LICENSE", "NOTICE", "README.md", "pyproject.toml", "uv.lock")
SIMULATOR_DIRS = {
    "baselines", "buildings", "smart_control", "training_scenarios", "weather"
}
TRAINING_SCRIPTS = (
    "scripts/__init__.py",
    "scripts/train_ppo.py",
    "scripts/evaluate_ppo.py",
    "scripts/test_rl_env.py",
)


def source_paths():
    """Include tracked simulator files and the current local RL implementation."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "smart_buildings"], cwd=ROOT
    ).decode("utf-8").split("\0")
    paths = {Path(name) for name in (*ROOT_FILES, *TRAINING_SCRIPTS)}
    for name in filter(None, tracked):
        path = Path(name)
        if path.parts[1] in SIMULATOR_DIRS or len(path.parts) == 2:
            paths.add(path)

    paths.add(Path("smart_buildings/team/__init__.py"))
    paths.update(
        path.relative_to(ROOT)
        for path in (ROOT / "smart_buildings/team/rl").rglob("*.py")
    )
    return sorted(paths)


def build_bundle(output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files": {},
    }

    bundle = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as archive:
            for path in source_paths():
                name = path.as_posix()
                payload = (ROOT / path).read_bytes()
                manifest["files"][name] = sha256(payload).hexdigest()
                archive.writestr(name, payload)
            archive.writestr(
                "KAGGLE_SOURCE.json", json.dumps(manifest, indent=2) + "\n"
            )
        bundle.replace(output)
    finally:
        bundle.unlink(missing_ok=True)
    print(f"Bundled {len(manifest['files'])} files into {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/kaggle/rl_kaggle_source.zip",
    )
    args = parser.parse_args()
    build_bundle(args.output)


if __name__ == "__main__":
    main()
