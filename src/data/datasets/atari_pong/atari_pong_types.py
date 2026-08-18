from pydantic import BaseModel


class AtariPongWindow(BaseModel):
    """A single training window inside one ArrayRecord episode record.

    ``shard_path`` points at the ``.array_record`` file, ``record_index`` is
    the episode record inside that shard, and the window covers frames
    ``start_frame + i * frame_spacing`` for ``i in range(num_frames)``.
    """

    shard_path: str
    record_index: int
    start_frame: int
    num_frames: int
    frame_spacing: int

    def __hash__(self):
        return hash(
            (
                self.shard_path,
                self.record_index,
                self.start_frame,
                self.num_frames,
                self.frame_spacing,
            )
        )


class AtariPongVideoLog(BaseModel):
    windows: list[AtariPongWindow]

    def __len__(self):
        return len(self.windows)
