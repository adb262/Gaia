"""Dataset factory shared by every training/eval entry point.

``build_datasets(config, local_cache)`` hides the differences between the
two dataset backends (``--dataset-type``):

- ``atari_pong``: ArrayRecord shards from HuggingFace
  ``p-doom/atari-pong-dataset`` (download with
  ``scripts.data.download_atari_pong``), windowed by
  ``AtariPongDatasetCreator``.
- ``pokemon``: scraped open-world PNG frames + JSON window logs, windowed by
  ``OpenWorldRunningDatasetCreator`` (optionally pulled from S3).

Any config dataclass with the shared data fields works
(``VideoTokenizerTrainingConfig``, ``DynamicsModelTrainingConfig``,
``LatentActionModelTrainingConfig``, ...).
"""

import logging
from typing import Optional, Protocol

from torch.utils.data import Dataset

from data.datasets.atari_pong.atari_pong_dataset import AtariPongDataset
from data.datasets.atari_pong.atari_pong_dataset_creator import AtariPongDatasetCreator
from data.datasets.cache import Cache
from data.datasets.open_world.open_world_dataset import OpenWorldRunningDataset
from data.datasets.open_world.open_world_running_dataset_creator import (
    OpenWorldRunningDatasetCreator,
)

logger = logging.getLogger(__name__)


class DatasetConfig(Protocol):
    """The data-related fields shared by all training config dataclasses."""

    dataset_type: str
    image_size: int
    num_images_in_video: int
    frame_spacing: int
    num_unique_frames: Optional[int]
    dataset_limit: int
    # atari_pong backend
    atari_pong_data_dir: Optional[str]
    atari_pong_crop_scoreboard: bool
    atari_pong_require_full_gameplay: bool
    # pokemon backend
    frames_dir: str
    use_s3: bool
    dataset_train_key: Optional[str]
    sync_from_s3: bool


def build_datasets(
    config: DatasetConfig,
    local_cache: Cache,
    num_frames_in_video: Optional[int] = None,
    train_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
) -> tuple[Dataset, Dataset]:
    """Build the (train, test) window datasets for ``config.dataset_type``.

    ``num_frames_in_video`` overrides ``config.num_images_in_video`` for
    callers that need longer windows than the model context (e.g. rollout
    evals that use 2T-frame windows). ``train_limit``/``test_limit`` cap the
    number of windows kept in each split.
    """
    frames_per_window = (
        num_frames_in_video
        if num_frames_in_video is not None
        else config.num_images_in_video
    )

    if config.dataset_type == "atari_pong":
        return _build_atari_pong_datasets(
            config, frames_per_window, train_limit, test_limit
        )
    if config.dataset_type == "pokemon":
        return _build_pokemon_datasets(
            config, local_cache, frames_per_window, train_limit, test_limit
        )
    raise ValueError(f"Unknown dataset_type: {config.dataset_type!r}")


def _build_atari_pong_datasets(
    config: DatasetConfig,
    frames_per_window: int,
    train_limit: Optional[int],
    test_limit: Optional[int],
) -> tuple[AtariPongDataset, AtariPongDataset]:
    if config.atari_pong_data_dir is None:
        raise ValueError(
            "atari_pong_data_dir is required when dataset_type='atari_pong'"
        )

    creator = AtariPongDatasetCreator(
        data_dir=config.atari_pong_data_dir,
        num_frames_in_video=frames_per_window,
        limit=config.dataset_limit,
        image_size=config.image_size,
        frame_spacing=config.frame_spacing,
        require_full_gameplay=config.atari_pong_require_full_gameplay,
    )
    train_log, test_log = creator.setup()

    train_dataset = AtariPongDataset(
        dataset=train_log,
        image_size=config.image_size,
        num_images_in_video=frames_per_window,
        crop_scoreboard=config.atari_pong_crop_scoreboard,
        limit=train_limit,
    )
    test_dataset = AtariPongDataset(
        dataset=test_log,
        image_size=config.image_size,
        num_images_in_video=frames_per_window,
        crop_scoreboard=config.atari_pong_crop_scoreboard,
        limit=test_limit,
    )
    logger.info(
        f"Built atari_pong datasets: {len(train_dataset)} train / "
        f"{len(test_dataset)} test windows of {frames_per_window} frames"
    )
    return train_dataset, test_dataset


def _build_pokemon_datasets(
    config: DatasetConfig,
    local_cache: Cache,
    frames_per_window: int,
    train_limit: Optional[int],
    test_limit: Optional[int],
) -> tuple[OpenWorldRunningDataset, OpenWorldRunningDataset]:
    creator = OpenWorldRunningDatasetCreator(
        dataset_dir=config.frames_dir,
        num_frames_in_video=frames_per_window,
        output_log_json_file_name=config.frames_dir,
        local_cache=local_cache,
        limit=config.dataset_limit,
        image_size=config.image_size,
        use_s3=config.use_s3,
        frame_spacing=config.frame_spacing,
    )

    if config.dataset_train_key is not None:
        # Pre-built window logs (typically produced on another machine and
        # synced alongside the frames): load them directly instead of
        # re-scanning the frames directory.
        train_key = config.dataset_train_key
        test_key = train_key.replace("train", "test")
        train_log = creator.load_existing_dataset(train_key)
        test_log = creator.load_existing_dataset(test_key)
        if config.sync_from_s3:
            creator.ensure_files_exist(train_log)
            creator.ensure_files_exist(test_log)
    else:
        train_log, test_log = creator.setup()

    train_dataset = OpenWorldRunningDataset(
        dataset=train_log,
        local_cache=local_cache,
        image_size=config.image_size,
        num_images_in_video=frames_per_window,
        num_unique_frames=config.num_unique_frames,
        limit=train_limit,
    )
    test_dataset = OpenWorldRunningDataset(
        dataset=test_log,
        local_cache=local_cache,
        image_size=config.image_size,
        num_images_in_video=frames_per_window,
        num_unique_frames=config.num_unique_frames,
        limit=test_limit,
    )
    logger.info(
        f"Built pokemon datasets: {len(train_dataset)} train / "
        f"{len(test_dataset)} test windows of {frames_per_window} frames"
    )
    return train_dataset, test_dataset
