from __future__ import annotations

import argparse
import json
import logging
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, Wide_ResNet50_2_Weights, resnet50, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LOGGER = logging.getLogger(__name__)


class Config:
    data_root = Path("input/mvtec_ad_2")
    model_dir = Path("artifacts/patchcore")
    public_maps_dir = Path("outputs/patchcore/test_public")
    private_maps_dir = Path("submissions/patchcore")

    categories = [
        "can",
        "fabric",
        "fruit_jelly",
        "rice",
        "sheet_metal",
        "vial",
        "wallplugs",
        "walnuts",
    ]
    include_validation_good = False
    private_splits = ["test_private", "test_private_mixed"]

    backbone = "wide_resnet50_2"
    pretrained = True
    layers = ["layer2", "layer3"]
    image_size = 256
    batch_size = 8

    coreset_ratio = 0.1
    max_memory_patches = 50000
    search_chunk_size = 1024
    gaussian_sigma = 4.0

    seed = 42
    num_workers = 2
    device = "auto"
    log_level = "INFO"

    debug_enabled = False
    limit_train_images = 2
    limit_eval_images = 2
    limit_predict_images = 2


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            original_size = image.size
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path), original_size


class PatchFeatureExtractor(nn.Module):
    def __init__(self, backbone, layers, pretrained):
        super().__init__()
        if layers != ["layer2", "layer3"]:
            raise ValueError("This baseline currently expects layers = ['layer2', 'layer3'].")

        if backbone == "wide_resnet50_2":
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            model = wide_resnet50_2(weights=weights)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.extractor = create_feature_extractor(model, return_nodes={layer: layer for layer in layers})

    @torch.inference_mode()
    def forward(self, x):
        features: OrderedDict[str, torch.Tensor] = self.extractor(x)
        layer2 = features["layer2"]
        layer3 = F.interpolate(features["layer3"], size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        patches = torch.cat([layer2, layer3], dim=1)
        return F.normalize(patches, p=2, dim=1)


parser = argparse.ArgumentParser(description="Single-file PatchCore baseline.")
parser.add_argument(
    "mode",
    nargs="?",
    default="all",
    choices=["train", "evaluate", "predict", "all"],
    help="Run one stage or all stages in order.",
)
args = parser.parse_args()

data_root = Config.data_root
model_dir = Config.model_dir
public_maps_dir = Config.public_maps_dir
private_maps_dir = Config.private_maps_dir
categories = list(Config.categories)
include_validation_good = Config.include_validation_good
private_splits = list(Config.private_splits)
backbone = Config.backbone
pretrained = Config.pretrained
layers = list(Config.layers)
image_size = Config.image_size
batch_size = Config.batch_size
coreset_ratio = Config.coreset_ratio
max_memory_patches = Config.max_memory_patches
search_chunk_size = Config.search_chunk_size
gaussian_sigma = Config.gaussian_sigma
seed = Config.seed
num_workers = Config.num_workers
device_name = Config.device
log_level = Config.log_level
debug_enabled = Config.debug_enabled
limit_train_images = Config.limit_train_images
limit_eval_images = Config.limit_eval_images
limit_predict_images = Config.limit_predict_images

logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if device_name == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false.")
LOGGER.info("Using device: %s", device)

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

if args.mode in {"train", "all"}:
    if not 0 < coreset_ratio <= 1:
        raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}")

    for category in categories:
        root = data_root / category
        train_dir = root / "train" / "good"
        paths = sorted(p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if train_dir.exists() else []
        if include_validation_good:
            val_dir = root / "validation" / "good"
            paths.extend(sorted(p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if val_dir.exists() else [])
        paths = sorted(paths)
        if not paths:
            raise ValueError(f"{category}: no normal training images found.")
        if debug_enabled and limit_train_images > 0:
            paths = paths[:limit_train_images]

        LOGGER.info("%s: training with %d normal images", category, len(paths))
        loader = DataLoader(ImagePathDataset(paths, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        extractor = PatchFeatureExtractor(backbone, layers, pretrained)
        extractor.eval().to(device)
        chunks = []
        grid_size = None
        for images, _, _ in tqdm(loader, desc=f"{category}: extracting normal features"):
            features = extractor(images.to(device, non_blocking=True))
            grid_size = tuple(features.shape[-2:])
            chunks.append(features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).cpu())
        if not chunks:
            raise RuntimeError(f"{category}: feature extraction produced an empty memory bank.")

        memory = torch.cat(chunks, dim=0).float()
        LOGGER.info("%s: raw memory bank shape=%s grid=%s", category, tuple(memory.shape), grid_size)
        target = min(len(memory), max(1, int(len(memory) * coreset_ratio)), max_memory_patches)
        if target < len(memory):
            memory = memory[torch.randperm(len(memory))[:target]].contiguous()
            LOGGER.info("Subsampled memory bank to %d patches", len(memory))
        else:
            memory = memory.contiguous()

        category_dir = model_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"memory_bank": memory}, category_dir / "memory_bank.pt")
        metadata = {
            "category": category,
            "backbone": backbone,
            "pretrained": pretrained,
            "layers": layers,
            "image_size": image_size,
            "memory_bank_shape": list(memory.shape),
            "coreset_ratio": coreset_ratio,
            "max_memory_patches": max_memory_patches,
        }
        (category_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info("%s: saved memory bank with shape=%s", category, tuple(memory.shape))

if args.mode in {"evaluate", "all"}:
    all_results = {}
    for category in categories:
        root = data_root / category
        good_dir = root / "test_public" / "good"
        bad_dir = root / "test_public" / "bad"
        mask_dir = root / "test_public" / "ground_truth" / "bad"
        samples = []
        good_paths = sorted(p for p in good_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if good_dir.exists() else []
        bad_paths = sorted(p for p in bad_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if bad_dir.exists() else []
        samples.extend((p, 0, None) for p in good_paths)
        for path in bad_paths:
            mask_path = mask_dir / f"{path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {path}: expected {mask_path}")
            samples.append((path, 1, mask_path))
        if not samples:
            raise ValueError(f"{category}: no public test images found.")
        if debug_enabled:
            good_samples = [s for s in samples if s[1] == 0]
            bad_samples = [s for s in samples if s[1] == 1]
            per_class_limit = max(1, limit_eval_images)
            samples = good_samples[:per_class_limit] + bad_samples[:per_class_limit]

        memory_path = model_dir / category / "memory_bank.pt"
        if not memory_path.exists():
            raise FileNotFoundError(f"Missing memory bank for {category}: {memory_path}")
        memory_bank = torch.load(memory_path, map_location="cpu", weights_only=True)["memory_bank"].float().contiguous()
        if memory_bank.ndim != 2:
            raise ValueError(f"Invalid memory bank shape for {category}: {tuple(memory_bank.shape)}")

        paths = [sample[0] for sample in samples]
        path_to_sample = {sample[0]: sample for sample in samples}
        loader = DataLoader(ImagePathDataset(paths, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        extractor = PatchFeatureExtractor(backbone, layers, pretrained)
        extractor.eval().to(device)
        scores = []
        masks_for_metrics = []
        output_dir = public_maps_dir / category

        for images, batch_paths, original_sizes in tqdm(loader, desc=f"{category}: scoring"):
            features = extractor(images.to(device, non_blocking=True))
            b, _, grid_h, grid_w = features.shape
            flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
            memory_on_device = memory_bank.to(device)
            patch_chunks = []
            for chunk in flat.to(device).split(search_chunk_size):
                patch_chunks.append(torch.cdist(chunk, memory_on_device, p=2).min(dim=1).values.cpu())
            patch_scores = torch.cat(patch_chunks, dim=0).view(b, grid_h, grid_w)
            widths = original_sizes[0].tolist() if torch.is_tensor(original_sizes[0]) else list(original_sizes[0])
            heights = original_sizes[1].tolist() if torch.is_tensor(original_sizes[1]) else list(original_sizes[1])

            for i in range(b):
                score = patch_scores[i].view(1, 1, grid_h, grid_w).to(device)
                if gaussian_sigma > 0:
                    radius = max(1, int(3 * gaussian_sigma))
                    coords = torch.arange(-radius, radius + 1, dtype=score.dtype, device=score.device)
                    kernel = torch.exp(-(coords**2) / (2 * gaussian_sigma**2))
                    kernel = kernel / kernel.sum()
                    score = F.conv2d(F.pad(score, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
                    score = F.conv2d(F.pad(score, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))
                score = F.interpolate(score, size=(int(heights[i]), int(widths[i])), mode="bilinear", align_corners=False)
                anomaly_map = score.squeeze().cpu().numpy().astype(np.float32)
                path = Path(batch_paths[i])
                output_path = output_dir / f"{path.stem}.tiff"
                if not np.isfinite(anomaly_map).all():
                    raise ValueError(f"Anomaly map contains non-finite values: {output_path}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(output_path, anomaly_map.astype(np.float16))

                sample = path_to_sample[path]
                if sample[2] is None:
                    with Image.open(path) as image:
                        width, height = image.size
                    mask = np.zeros((height, width), dtype=np.uint8)
                else:
                    with Image.open(sample[2]) as mask_image:
                        mask = (np.asarray(mask_image.convert("L")) > 0).astype(np.uint8)
                scores.append(anomaly_map)
                masks_for_metrics.append(mask)

        y_score = np.concatenate([s.reshape(-1).astype(np.float32) for s in scores])
        y_true = np.concatenate([m.reshape(-1).astype(np.uint8) for m in masks_for_metrics])
        if int(y_true.sum()) == 0 or int(len(y_true) - y_true.sum()) == 0:
            raise ValueError("Pixel AUROC requires both positive and negative pixels.")
        auroc = float(roc_auc_score(y_true, y_score))
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        best_index = int(np.nanargmax(f1))
        threshold = float(thresholds[min(best_index, len(thresholds) - 1)]) if len(thresholds) else 0.0
        pred = y_score >= threshold
        intersection = float(np.logical_and(pred, y_true == 1).sum())
        union = float(np.logical_or(pred, y_true == 1).sum())
        all_results[category] = {
            "pixel_auroc": auroc,
            "best_f1": float(f1[best_index]),
            "best_iou": intersection / max(union, 1.0),
            "threshold": threshold,
        }
        LOGGER.info("%s: AUROC=%.5f F1=%.5f IoU=%.5f threshold=%.6f", category, auroc, all_results[category]["best_f1"], all_results[category]["best_iou"], threshold)

    public_maps_dir.mkdir(parents=True, exist_ok=True)
    (public_maps_dir / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", public_maps_dir / "metrics.json")

if args.mode in {"predict", "all"}:
    for split in private_splits:
        if split not in {"test_private", "test_private_mixed"}:
            raise ValueError(f"Unsupported private split: {split}")
        for category in categories:
            root = data_root / category
            split_dir = root / split
            paths = sorted(p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if split_dir.exists() else []
            if not paths:
                raise ValueError(f"{category}: no images found for {split}")
            if debug_enabled and limit_predict_images > 0:
                paths = paths[:limit_predict_images]

            memory_path = model_dir / category / "memory_bank.pt"
            if not memory_path.exists():
                raise FileNotFoundError(f"Missing memory bank for {category}: {memory_path}")
            memory_bank = torch.load(memory_path, map_location="cpu", weights_only=True)["memory_bank"].float().contiguous()
            if memory_bank.ndim != 2:
                raise ValueError(f"Invalid memory bank shape for {category}: {tuple(memory_bank.shape)}")

            loader = DataLoader(ImagePathDataset(paths, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
            extractor = PatchFeatureExtractor(backbone, layers, pretrained)
            extractor.eval().to(device)
            output_dir = private_maps_dir / split / category

            for images, batch_paths, original_sizes in tqdm(loader, desc=f"{category}: scoring"):
                features = extractor(images.to(device, non_blocking=True))
                b, _, grid_h, grid_w = features.shape
                flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
                memory_on_device = memory_bank.to(device)
                patch_chunks = []
                for chunk in flat.to(device).split(search_chunk_size):
                    patch_chunks.append(torch.cdist(chunk, memory_on_device, p=2).min(dim=1).values.cpu())
                patch_scores = torch.cat(patch_chunks, dim=0).view(b, grid_h, grid_w)
                widths = original_sizes[0].tolist() if torch.is_tensor(original_sizes[0]) else list(original_sizes[0])
                heights = original_sizes[1].tolist() if torch.is_tensor(original_sizes[1]) else list(original_sizes[1])

                for i in range(b):
                    score = patch_scores[i].view(1, 1, grid_h, grid_w).to(device)
                    if gaussian_sigma > 0:
                        radius = max(1, int(3 * gaussian_sigma))
                        coords = torch.arange(-radius, radius + 1, dtype=score.dtype, device=score.device)
                        kernel = torch.exp(-(coords**2) / (2 * gaussian_sigma**2))
                        kernel = kernel / kernel.sum()
                        score = F.conv2d(F.pad(score, (radius, radius, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
                        score = F.conv2d(F.pad(score, (0, 0, radius, radius), mode="reflect"), kernel.view(1, 1, -1, 1))
                    score = F.interpolate(score, size=(int(heights[i]), int(widths[i])), mode="bilinear", align_corners=False)
                    anomaly_map = score.squeeze().cpu().numpy().astype(np.float32)
                    path = Path(batch_paths[i])
                    output_path = output_dir / f"{path.stem}.tiff"
                    if not np.isfinite(anomaly_map).all():
                        raise ValueError(f"Anomaly map contains non-finite values: {output_path}")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(output_path, anomaly_map.astype(np.float16))
            LOGGER.info("%s/%s: wrote %d maps", split, category, len(paths))
