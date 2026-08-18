import logging
import traceback

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.datasets.atari_pong.atari_pong_dataset_creator import (
    SCOREBOARD_ROWS,
    decode_record_frames,
)
from data.datasets.atari_pong.atari_pong_types import AtariPongVideoLog

logger = logging.getLogger(__name__)


class AtariPongDataset(Dataset):
    """Dataset of pre-indexed Atari Pong windows stored in ArrayRecord shards.

    Each sample is a ``(num_images_in_video, 3, image_size, image_size)``
    float tensor in ``[0, 1]``, matching what ``OpenWorldRunningDataset``
    yields so both backends can share ``VideoWindowLoader``. Windows are
    produced ahead of time by ``AtariPongDatasetCreator``; this class only
    reads frames back out of the shards.
    """

    def __init__(
        self,
        dataset: AtariPongVideoLog,
        image_size: int,
        num_images_in_video: int,
        crop_scoreboard: bool = False,
        limit: int | None = None,
    ):
        windows = dataset.windows
        if limit is not None:
            windows = windows[:limit]

        self.samples = [
            w for w in windows if w.num_frames >= num_images_in_video
        ]
        self.image_size = image_size
        self.num_images_in_video = num_images_in_video
        self.crop_scoreboard = crop_scoreboard
        # Shard readers are opened lazily (and per dataloader worker, since
        # they are dropped from the pickled state) because ArrayRecord
        # readers hold OS file handles that can't cross process boundaries.
        self._readers: dict[str, object] = {}

        logger.info(
            f"Using {len(self.samples)} atari_pong window samples "
            f"(num_images_in_video={num_images_in_video}, "
            f"crop_scoreboard={crop_scoreboard})"
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_readers"] = {}
        return state

    def _get_reader(self, shard_path: str):
        reader = self._readers.get(shard_path)
        if reader is None:
            from array_record.python.array_record_data_source import (
                ArrayRecordDataSource,
            )

            reader = ArrayRecordDataSource([shard_path])
            self._readers[shard_path] = reader
        return reader

    def _preprocess_frames(self, frames_uint8) -> torch.Tensor:
        # (T, H, W, C) uint8 -> (T, C, H, W) float in [0, 1]
        video = torch.from_numpy(frames_uint8.copy()).permute(0, 3, 1, 2).float()
        video = video / 255.0

        if self.crop_scoreboard:
            video = video[:, :, SCOREBOARD_ROWS:, :]

        if video.shape[-2:] != (self.image_size, self.image_size):
            video = F.interpolate(
                video,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return video

    #### BASIC DATASET METHODS ####

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        window = self.samples[idx]
        try:
            reader = self._get_reader(window.shard_path)
            frames = decode_record_frames(reader[window.record_index])

            frame_indices = [
                window.start_frame + i * window.frame_spacing
                for i in range(self.num_images_in_video)
            ]
            return self._preprocess_frames(frames[frame_indices])
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error loading atari_pong window: {e}")
            return None
