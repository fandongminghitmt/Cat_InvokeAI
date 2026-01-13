from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
    VideoRecord,
    VideoRecordChanges,
)
from invokeai.app.services.images.images_common import VideoDTO
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class VideoServiceABC(ABC):
    """High-level service for video management."""

    @abstractmethod
    def create(
        self,
        video_file: bytes, # Or path, or stream
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
    ) -> VideoDTO:
        """Creates a video, storing the file and its metadata."""
        pass

    @abstractmethod
    def get_dto(self, video_name: str) -> VideoDTO:
        """Gets a video DTO."""
        pass
