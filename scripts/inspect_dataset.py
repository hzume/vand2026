from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patchcore_baseline.config import load_config
from patchcore_baseline.data import summarize_category
from patchcore_baseline.utils import setup_logging


def main() -> None:
    config = load_config()
    setup_logging(config.runtime.log_level)

    print(f"data_root: {config.paths.data_root}")
    for category in config.data.categories:
        summary = summarize_category(config.paths.data_root, category)
        print(
            f"{summary.name:12s} size={summary.image_size} mode={summary.image_mode:3s} "
            f"train={summary.train_good:3d} val={summary.validation_good:3d} "
            f"public_good={summary.public_good:3d} public_bad={summary.public_bad:3d} "
            f"masks={summary.public_masks:3d} private={summary.private:3d} mixed={summary.private_mixed:3d}"
        )


if __name__ == "__main__":
    main()
