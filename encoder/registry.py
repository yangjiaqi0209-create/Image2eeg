"""
Central dataset registry for UBP cross-dataset experiments.

Each dataset profile (data/profiles/*.yaml) declares:
  - model I/O dimensions (c_num, timesteps, brain_input_dim)
  - trial semantics (repeated vs per_image, averaging flags)
  - paths, subjects, class counts
  - links to preprocess scripts and training configs

Usage:
  from encoder.registry import get_dataset, list_datasets

  ds = get_dataset("alljoined_eeg")
  print(ds.c_num, ds.timesteps, ds.paths["data_dir"])
  cfg_patch = ds.encoder_data_config()  # merge into OmegaConf for encoder.train

CLI:
  python -m encoder.registry list
  python -m encoder.registry show alljoined_eeg
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from encoder.paths import (
    eeg_data_root as default_eeg_data_root,
    repo_root as default_repo_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "data" / "registry.yaml"


@dataclass(frozen=True)
class DatasetProfile:
    """Resolved dataset specification."""

    id: str
    display_name: str
    raw: Dict[str, Any]
    paths: Dict[str, str]
    channels: List[str]

    @property
    def c_num(self) -> int:
        return int(self.raw["c_num"])

    @property
    def timesteps(self) -> List[int]:
        return list(self.raw["timesteps"])

    @property
    def seq_len(self) -> int:
        t = self.timesteps
        return int(self.raw.get("seq_len", t[1] - t[0]))

    @property
    def brain_input_dim(self) -> int:
        return int(self.raw.get("brain_input_dim", self.c_num * self.seq_len))

    @property
    def trial_format(self) -> str:
        return str(self.raw.get("trial_format", "repeated"))

    @property
    def subjects(self) -> List[str]:
        return list(self.raw.get("subjects", []))

    def path(self, key: str) -> str:
        if key not in self.paths:
            raise KeyError(f"Path {key!r} not in dataset {self.id}; have {list(self.paths)}")
        return self.paths[key]

    def encoder_data_config(self) -> Dict[str, Any]:
        """Fields to merge under config['data'] for encoder training."""
        data: Dict[str, Any] = {
            "data_dir": self.path("data_dir"),
            "timesteps": self.timesteps,
            "train_avg": bool(self.raw.get("train_avg", True)),
            "test_avg": bool(self.raw.get("test_avg", True)),
        }
        if "val_avg" in self.raw:
            data["val_avg"] = bool(self.raw["val_avg"])
        if self.raw.get("per_trials") is not None:
            data["per_trials"] = int(self.raw["per_trials"])
        for key in ("n_train_classes", "n_val_classes", "n_test_classes"):
            if key in self.raw:
                data[key] = int(self.raw[key])
        if "image_root" in self.paths:
            data["image_root"] = self.path("image_root")
        if "feature_dir" in self.paths:
            data["img_feature_dir"] = self.path("feature_dir")
        if self.subjects:
            data["subjects"] = self.subjects
        if self.channels:
            data["channels"] = self.channels
        return data

    def has_separate_val(self) -> bool:
        return bool((self.raw.get("train") or {}).get("separate_val", False))

    def summary_lines(self) -> List[str]:
        lines = [
            f"{self.display_name} ({self.id})",
            f"  brain input: {self.c_num} ch × {self.seq_len} t = {self.brain_input_dim}",
            f"  timesteps: {self.timesteps}  trial_format: {self.trial_format}",
            f"  subjects: {len(self.subjects)}  train_avg: {self.raw.get('train_avg')}",
            f"  data_dir: {self.path('data_dir')}",
        ]
        if "image_root" in self.paths:
            lines.append(f"  image_root: {self.path('image_root')}")
        train = self.raw.get("train") or {}
        if train.get("encoder_config"):
            lines.append(f"  encoder_config: {train['encoder_config']}")
        return lines


def _resolve_channels(ref: Optional[str]) -> List[str]:
    if not ref:
        return []
    module_path, _, attr = ref.rpartition(".")
    mod = importlib.import_module(module_path)
    return list(getattr(mod, attr))


def _format_path(
    value: str,
    repo_root: Path,
    eeg_data_root: str,
) -> str:
    return value.format(
        repo_root=str(repo_root),
        eeg_data_root=eeg_data_root,
    )


def _load_registry_index() -> Dict[str, Any]:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def list_datasets() -> List[str]:
    index = _load_registry_index()
    return list(index["datasets"].keys())


def get_dataset(
    dataset_id: Optional[str] = None,
    *,
    repo_root: Optional[Path] = None,
    eeg_data_root: Optional[str] = None,
) -> DatasetProfile:
    index = _load_registry_index()
    dataset_id = dataset_id or index.get("default", "things_eeg")
    if dataset_id not in index["datasets"]:
        raise KeyError(f"Unknown dataset {dataset_id!r}; choose from {list_datasets()}")

    repo_root = Path(repo_root or default_repo_root())
    eeg_root = eeg_data_root or default_eeg_data_root()

    entry = index["datasets"][dataset_id]
    profile_path = REGISTRY_PATH.parent / entry["profile"]
    with open(profile_path) as f:
        raw = yaml.safe_load(f)

    paths = {
        k: _format_path(v, repo_root, eeg_root)
        for k, v in (raw.get("paths") or {}).items()
    }
    channels = _resolve_channels(raw.get("channels_ref"))

    return DatasetProfile(
        id=raw["id"],
        display_name=raw.get("display_name", raw["id"]),
        raw=raw,
        paths=paths,
        channels=channels,
    )


def apply_profile_to_config(config, profile_id: str):
    """Merge a dataset profile into an OmegaConf encoder config.

    Updates data paths, trial semantics, brain ``c_num`` / ``timesteps``.
    THINGS training omits ``--dataset-profile`` and is unchanged.
    """
    from omegaconf import OmegaConf

    prof = get_dataset(profile_id)
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)

    data = config.get("data")
    if data is None:
        config.data = OmegaConf.create({})
        data = config.data

    for key, val in prof.encoder_data_config().items():
        data[key] = val

    config.c_num = prof.c_num
    config.timesteps = list(prof.timesteps)
    config.dataset_profile = profile_id
    exp_name = (prof.raw.get("train") or {}).get("encoder_exp")
    if exp_name:
        config.name = exp_name

    if prof.raw.get("trial_format"):
        data.trial_format = prof.raw["trial_format"]

    data.separate_val = prof.has_separate_val()

    if config.get("models") and config.models.get("brain"):
        params = config.models.brain.get("params")
        if params is None:
            config.models.brain.params = OmegaConf.create({})
            params = config.models.brain.params
        params.c_num = prof.c_num
        params.timesteps = list(prof.timesteps)

    return config


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="UBP dataset registry")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registered dataset ids")

    show_p = sub.add_parser("show", help="Print resolved profile")
    show_p.add_argument("dataset_id", nargs="?", default=None)

    args = p.parse_args()
    if args.cmd == "list":
        for ds_id in list_datasets():
            index = _load_registry_index()
            desc = index["datasets"][ds_id].get("description", "")
            default = " (default)" if ds_id == index.get("default") else ""
            print(f"  {ds_id}{default}: {desc}")
    elif args.cmd == "show":
        prof = get_dataset(args.dataset_id)
        for line in prof.summary_lines():
            print(line)


if __name__ == "__main__":
    _cli()
