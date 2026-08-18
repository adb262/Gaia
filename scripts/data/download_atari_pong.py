"""Download the public HuggingFace Atari Pong dataset.

Fetches ArrayRecord shards from ``p-doom/atari-pong-dataset`` (10M frames of
84x84x3 uint8 Pong gameplay collected during Rainbow agent training) into a
local directory laid out as ``<local_dir>/{train,test}/data_XXXX.array_record``
plus ``metadata.json``. That layout is what ``AtariPongDatasetCreator``
expects via ``--atari-pong-data-dir``.

Usage:

    uv run python -m scripts.data.download_atari_pong --local_dir data/atari_pong

The dataset is public; ``HF_TOKEN`` (from the environment or ``.env``) is
only needed if you hit anonymous rate limits. Use ``--max-shards-per-split``
to grab a small subset for smoke tests.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import tyro
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

REPO_ID = "p-doom/atari-pong-dataset"


@dataclass
class DownloadAtariPongArgs:
    local_dir: str = "data/atari_pong"
    repo_id: str = REPO_ID
    splits: list[str] = field(default_factory=lambda: ["train", "test"])
    # Limit how many shards to download per split (each train shard holds 100
    # episodes of ~150 frames). None downloads the full split.
    max_shards_per_split: Optional[int] = None


def _allow_patterns(args: DownloadAtariPongArgs) -> list[str]:
    patterns = ["metadata.json"]
    for split in args.splits:
        if args.max_shards_per_split is None:
            patterns.append(f"{split}/*.array_record")
        else:
            patterns.extend(
                f"{split}/data_{shard_index:04d}.array_record"
                for shard_index in range(args.max_shards_per_split)
            )
    return patterns


def download_atari_pong(args: DownloadAtariPongArgs) -> str:
    load_dotenv()
    token = os.environ.get("HF_TOKEN")

    logger.info(
        f"Downloading {args.repo_id} splits={args.splits} "
        f"max_shards_per_split={args.max_shards_per_split} "
        f"into {args.local_dir}"
    )
    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=_allow_patterns(args),
        token=token,
    )
    logger.info(f"Dataset ready at {local_path}")
    return local_path


def main() -> None:
    args = tyro.cli(DownloadAtariPongArgs)
    download_atari_pong(args)


if __name__ == "__main__":
    main()
