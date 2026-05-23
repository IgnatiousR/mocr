from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manga_translator.model_manager import (
    ModelDownloadError,
    download_realesrgan_model,
    download_translation_model,
    ensure_model_dirs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download project-local model files.")
    parser.add_argument("--translation", action="store_true", help="Download the configured GGUF translation model.")
    parser.add_argument("--upscale", action="store_true", help="Download the configured Real-ESRGAN model.")
    parser.add_argument("--all", action="store_true", help="Download all configured models.")
    args = parser.parse_args()

    if not (args.translation or args.upscale or args.all):
        parser.print_help()
        return 0

    ensure_model_dirs()
    try:
        if args.translation or args.all:
            path = download_translation_model()
            print(f"Downloaded translation model: {path}")
        if args.upscale or args.all:
            path = download_realesrgan_model()
            print(f"Downloaded Real-ESRGAN model: {path}")
    except ModelDownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
