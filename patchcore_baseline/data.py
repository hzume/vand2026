from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CategorySummary:
    name: str
    image_size: tuple[int, int]
    image_mode: str
    train_good: int
    validation_good: int
    public_good: int
    public_bad: int
    public_masks: int
    private: int
    private_mixed: int


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    mask_path: Path | None


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def discover_categories(data_root: Path) -> list[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    categories = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if not categories:
        raise ValueError(f"No category directories found under {data_root}")
    return categories


def summarize_category(data_root: Path, category: str) -> CategorySummary:
    root = data_root / category
    train = list_images(root / "train" / "good")
    if not train:
        raise ValueError(f"No training images found at {root / 'train' / 'good'}")

    with Image.open(train[0]) as image:
        image_size = image.size
        image_mode = image.mode

    public_bad = list_images(root / "test_public" / "bad")
    masks = list_images(root / "test_public" / "ground_truth" / "bad")
    missing = [
        p.name
        for p in public_bad
        if not (root / "test_public" / "ground_truth" / "bad" / f"{p.stem}_mask.png").exists()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{category}: missing ground-truth masks for {len(missing)} bad images: {preview}")

    return CategorySummary(
        name=category,
        image_size=image_size,
        image_mode=image_mode,
        train_good=len(train),
        validation_good=len(list_images(root / "validation" / "good")),
        public_good=len(list_images(root / "test_public" / "good")),
        public_bad=len(public_bad),
        public_masks=len(masks),
        private=len(list_images(root / "test_private")),
        private_mixed=len(list_images(root / "test_private_mixed")),
    )


def training_paths(data_root: Path, category: str, include_validation_good: bool) -> list[Path]:
    root = data_root / category
    paths = list_images(root / "train" / "good")
    if include_validation_good:
        paths.extend(list_images(root / "validation" / "good"))
    if not paths:
        raise ValueError(f"{category}: no normal training images found.")
    return sorted(paths)


def public_samples(data_root: Path, category: str) -> list[Sample]:
    root = data_root / category
    samples = [Sample(path, 0, None) for path in list_images(root / "test_public" / "good")]
    for path in list_images(root / "test_public" / "bad"):
        mask = root / "test_public" / "ground_truth" / "bad" / f"{path.stem}_mask.png"
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask for {path}: expected {mask}")
        samples.append(Sample(path, 1, mask))
    if not samples:
        raise ValueError(f"{category}: no public test images found.")
    return samples


def private_paths(data_root: Path, category: str, split: str) -> list[Path]:
    if split not in {"test_private", "test_private_mixed"}:
        raise ValueError(f"Unsupported private split: {split}")
    paths = list_images(data_root / category / split)
    if not paths:
        raise ValueError(f"{category}: no images found for {split}")
    return paths


class ImagePathDataset(Dataset[tuple[torch.Tensor, str, tuple[int, int]]]):
    def __init__(self, paths: list[Path], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, tuple[int, int]]:
        path = self.paths[index]
        with Image.open(path) as image:
            original_size = image.size
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path), original_size
