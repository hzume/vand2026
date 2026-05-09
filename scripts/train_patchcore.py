from __future__ import annotations

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchcore_baseline.config import load_config
from patchcore_baseline.data import training_paths
from patchcore_baseline.patchcore import debug_limit_items, extract_memory_bank, save_model_artifacts
from patchcore_baseline.utils import resolve_device, seed_everything, setup_logging


LOGGER = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    setup_logging(config.runtime.log_level)
    seed_everything(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    LOGGER.info("Using device: %s", device)

    for category in config.data.categories:
        paths = training_paths(config.paths.data_root, category, config.data.include_validation_good)
        paths = debug_limit_items(paths, config.debug.limit_train_images, config.debug.enabled)
        LOGGER.info("%s: training with %d normal images", category, len(paths))
        memory = extract_memory_bank(paths, config, device, category)
        save_model_artifacts(memory, config.paths.model_dir, category, config)
        LOGGER.info("%s: saved memory bank with shape=%s", category, tuple(memory.shape))


if __name__ == "__main__":
    main()
