from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from .config import Config
from .data import ImagePathDataset
from .model import PatchFeatureExtractor, flatten_patch_features
from .transforms import build_transform


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def limit_items(items: list[T], limit: int) -> list[T]:
    return items if limit <= 0 else items[:limit]


def debug_limit_items(items: list[T], limit: int, enabled: bool) -> list[T]:
    return limit_items(items, limit) if enabled else items


def build_loader(paths: list[Path], config: Config, shuffle: bool = False) -> DataLoader:
    dataset = ImagePathDataset(paths, build_transform(config.model.image_size))
    return DataLoader(
        dataset,
        batch_size=config.model.batch_size,
        shuffle=shuffle,
        num_workers=config.runtime.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_extractor(config: Config, device: torch.device) -> PatchFeatureExtractor:
    model = PatchFeatureExtractor(
        backbone=config.model.backbone,
        layers=config.model.layers,
        pretrained=config.model.pretrained,
    )
    model.eval().to(device)
    return model


def extract_memory_bank(
    image_paths: list[Path],
    config: Config,
    device: torch.device,
    category: str,
) -> torch.Tensor:
    loader = build_loader(image_paths, config)
    extractor = make_extractor(config, device)
    chunks: list[torch.Tensor] = []
    grid_size: tuple[int, int] | None = None

    for images, _, _ in tqdm(loader, desc=f"{category}: extracting normal features"):
        features = extractor(images.to(device, non_blocking=True))
        grid_size = tuple(features.shape[-2:])
        chunks.append(flatten_patch_features(features).cpu())

    if not chunks:
        raise RuntimeError(f"{category}: feature extraction produced an empty memory bank.")

    memory = torch.cat(chunks, dim=0).float()
    LOGGER.info("%s: raw memory bank shape=%s grid=%s", category, tuple(memory.shape), grid_size)
    return subsample_memory_bank(memory, config.patchcore.coreset_ratio, config.patchcore.max_memory_patches)


def subsample_memory_bank(memory: torch.Tensor, ratio: float, max_patches: int) -> torch.Tensor:
    if not 0 < ratio <= 1:
        raise ValueError(f"coreset_ratio must be in (0, 1], got {ratio}")
    target = min(len(memory), max(1, int(len(memory) * ratio)), max_patches)
    if target >= len(memory):
        return memory.contiguous()
    indices = torch.randperm(len(memory))[:target]
    sampled = memory[indices].contiguous()
    LOGGER.info("Subsampled memory bank from %d to %d patches", len(memory), len(sampled))
    return sampled


def save_model_artifacts(memory: torch.Tensor, output_dir: Path, category: str, config: Config) -> None:
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"memory_bank": memory}, category_dir / "memory_bank.pt")
    metadata = {
        "category": category,
        "backbone": config.model.backbone,
        "pretrained": config.model.pretrained,
        "layers": config.model.layers,
        "image_size": config.model.image_size,
        "memory_bank_shape": list(memory.shape),
        "coreset_ratio": config.patchcore.coreset_ratio,
        "max_memory_patches": config.patchcore.max_memory_patches,
    }
    (category_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_memory_bank(model_dir: Path, category: str) -> torch.Tensor:
    path = model_dir / category / "memory_bank.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing memory bank for {category}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    memory = payload["memory_bank"].float().contiguous()
    if memory.ndim != 2:
        raise ValueError(f"Invalid memory bank shape for {category}: {tuple(memory.shape)}")
    return memory


@torch.inference_mode()
def nearest_neighbor_scores(
    query: torch.Tensor,
    memory: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    memory = memory.to(device)
    query = query.to(device)
    scores: list[torch.Tensor] = []
    for chunk in query.split(chunk_size):
        distances = torch.cdist(chunk, memory, p=2)
        scores.append(distances.min(dim=1).values.cpu())
    return torch.cat(scores, dim=0)


def gaussian_smooth(map_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return map_tensor
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=map_tensor.dtype, device=map_tensor.device)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)
    padded = F.pad(map_tensor, (radius, radius, 0, 0), mode="reflect")
    smoothed = F.conv2d(padded, kernel_x)
    padded = F.pad(smoothed, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel_y)


@torch.inference_mode()
def predict_maps(
    image_paths: list[Path],
    memory_bank: torch.Tensor,
    config: Config,
    device: torch.device,
    category: str,
) -> Iterable[tuple[Path, np.ndarray]]:
    loader = build_loader(image_paths, config)
    extractor = make_extractor(config, device)

    for images, paths, original_sizes in tqdm(loader, desc=f"{category}: scoring"):
        features = extractor(images.to(device, non_blocking=True))
        batch_size, _, grid_h, grid_w = features.shape
        flat = flatten_patch_features(features)
        patch_scores = nearest_neighbor_scores(
            flat,
            memory_bank,
            chunk_size=config.patchcore.search_chunk_size,
            device=device,
        )
        patch_scores = patch_scores.view(batch_size, grid_h, grid_w)

        widths = original_sizes[0].tolist() if torch.is_tensor(original_sizes[0]) else list(original_sizes[0])
        heights = original_sizes[1].tolist() if torch.is_tensor(original_sizes[1]) else list(original_sizes[1])

        for i in range(batch_size):
            score = patch_scores[i].view(1, 1, grid_h, grid_w).to(device)
            score = gaussian_smooth(score, config.patchcore.gaussian_sigma)
            score = F.interpolate(score, size=(int(heights[i]), int(widths[i])), mode="bilinear", align_corners=False)
            yield Path(paths[i]), score.squeeze().cpu().numpy().astype(np.float32)


def save_tiff_map(map_array: np.ndarray, output_path: Path) -> None:
    if not np.isfinite(map_array).all():
        raise ValueError(f"Anomaly map contains non-finite values: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, map_array.astype(np.float16))


def zero_mask_for_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        width, height = image.size
    return np.zeros((height, width), dtype=np.uint8)


def read_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        return (np.asarray(image.convert("L")) > 0).astype(np.uint8)
