from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


CONFIG_PATH = Path("configs/patchcore.toml")


@dataclass(frozen=True)
class PathsConfig:
    data_root: Path
    model_dir: Path
    public_maps_dir: Path
    private_maps_dir: Path


@dataclass(frozen=True)
class DataConfig:
    categories: list[str]
    include_validation_good: bool
    private_splits: list[str]


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    pretrained: bool
    layers: list[str]
    image_size: int
    batch_size: int


@dataclass(frozen=True)
class PatchCoreConfig:
    coreset_ratio: float
    max_memory_patches: int
    search_chunk_size: int
    gaussian_sigma: float


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    num_workers: int
    device: str
    log_level: str


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool
    limit_train_images: int
    limit_eval_images: int
    limit_predict_images: int


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    patchcore: PatchCoreConfig
    runtime: RuntimeConfig
    debug: DebugConfig


def load_config(path: Path = CONFIG_PATH) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    return Config(
        paths=PathsConfig(
            data_root=Path(raw["paths"]["data_root"]),
            model_dir=Path(raw["paths"]["model_dir"]),
            public_maps_dir=Path(raw["paths"]["public_maps_dir"]),
            private_maps_dir=Path(raw["paths"]["private_maps_dir"]),
        ),
        data=DataConfig(
            categories=list(raw["data"]["categories"]),
            include_validation_good=bool(raw["data"]["include_validation_good"]),
            private_splits=list(raw["data"]["private_splits"]),
        ),
        model=ModelConfig(
            backbone=str(raw["model"]["backbone"]),
            pretrained=bool(raw["model"]["pretrained"]),
            layers=list(raw["model"]["layers"]),
            image_size=int(raw["model"]["image_size"]),
            batch_size=int(raw["model"]["batch_size"]),
        ),
        patchcore=PatchCoreConfig(
            coreset_ratio=float(raw["patchcore"]["coreset_ratio"]),
            max_memory_patches=int(raw["patchcore"]["max_memory_patches"]),
            search_chunk_size=int(raw["patchcore"]["search_chunk_size"]),
            gaussian_sigma=float(raw["patchcore"]["gaussian_sigma"]),
        ),
        runtime=RuntimeConfig(
            seed=int(raw["runtime"]["seed"]),
            num_workers=int(raw["runtime"]["num_workers"]),
            device=str(raw["runtime"]["device"]),
            log_level=str(raw["runtime"]["log_level"]),
        ),
        debug=DebugConfig(
            enabled=bool(raw["debug"]["enabled"]),
            limit_train_images=int(raw["debug"]["limit_train_images"]),
            limit_eval_images=int(raw["debug"]["limit_eval_images"]),
            limit_predict_images=int(raw["debug"]["limit_predict_images"]),
        ),
    )
