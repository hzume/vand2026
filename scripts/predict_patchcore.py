from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchcore_baseline.config import load_config
from patchcore_baseline.data import private_paths
from patchcore_baseline.patchcore import debug_limit_items, load_memory_bank, predict_maps, save_tiff_map
from patchcore_baseline.utils import resolve_device, seed_everything, setup_logging


LOGGER = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    setup_logging(config.runtime.log_level)
    seed_everything(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    LOGGER.info("Using device: %s", device)

    for split in config.data.private_splits:
        for category in config.data.categories:
            paths = private_paths(config.paths.data_root, category, split)
            paths = debug_limit_items(paths, config.debug.limit_predict_images, config.debug.enabled)
            memory = load_memory_bank(config.paths.model_dir, category)
            output_dir = config.paths.private_maps_dir / split / category
            for path, anomaly_map in predict_maps(paths, memory, config, device, category):
                save_tiff_map(anomaly_map, output_dir / f"{path.stem}.tiff")
            LOGGER.info("%s/%s: wrote %d maps", split, category, len(paths))


if __name__ == "__main__":
    main()
