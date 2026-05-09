from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchcore_baseline.config import load_config
from patchcore_baseline.data import public_samples
from patchcore_baseline.metrics import compute_pixel_metrics
from patchcore_baseline.patchcore import (
    debug_limit_items,
    load_memory_bank,
    predict_maps,
    read_mask,
    save_tiff_map,
    zero_mask_for_image,
)
from patchcore_baseline.utils import resolve_device, seed_everything, setup_logging


LOGGER = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    setup_logging(config.runtime.log_level)
    seed_everything(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    LOGGER.info("Using device: %s", device)

    all_results = {}
    for category in config.data.categories:
        samples = public_samples(config.paths.data_root, category)
        if config.debug.enabled:
            good = [sample for sample in samples if sample.label == 0]
            bad = [sample for sample in samples if sample.label == 1]
            per_class_limit = max(1, config.debug.limit_eval_images)
            samples = good[:per_class_limit] + bad[:per_class_limit]
        memory = load_memory_bank(config.paths.model_dir, category)
        paths = [sample.image_path for sample in samples]
        path_to_sample = {sample.image_path: sample for sample in samples}

        scores = []
        masks = []
        output_dir = config.paths.public_maps_dir / category
        for path, anomaly_map in predict_maps(paths, memory, config, device, category):
            output_path = output_dir / f"{path.stem}.tiff"
            save_tiff_map(anomaly_map, output_path)
            sample = path_to_sample[path]
            mask = read_mask(sample.mask_path) if sample.mask_path else zero_mask_for_image(path)
            scores.append(anomaly_map)
            masks.append(mask)

        metrics = compute_pixel_metrics(scores, masks)
        all_results[category] = {
            "pixel_auroc": metrics.auroc,
            "best_f1": metrics.best_f1,
            "best_iou": metrics.best_iou,
            "threshold": metrics.threshold,
        }
        LOGGER.info(
            "%s: AUROC=%.5f F1=%.5f IoU=%.5f threshold=%.6f",
            category,
            metrics.auroc,
            metrics.best_f1,
            metrics.best_iou,
            metrics.threshold,
        )

    config.paths.public_maps_dir.mkdir(parents=True, exist_ok=True)
    results_path = config.paths.public_maps_dir / "metrics.json"
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", results_path)


if __name__ == "__main__":
    main()
