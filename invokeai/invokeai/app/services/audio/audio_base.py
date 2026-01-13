from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
    AudioRecord,
    AudioRecordChanges,
)
from invokeai.app.services.images.images_common import AudioDTO
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class AudioServiceABC(ABC):
    """High-level service for audio management."""

    @abstractmethod
    def create(
        self,
        audio_file: bytes,
        audio_origin: ResourceOrigin,
        audio_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
    ) -> AudioDTO:
        """Creates an audio, storing the file and its metadata."""
        pass

    @abstractmethod
    def get_dto(self, audio_name: str) -> AudioDTO:
        """Gets an audio DTO."""
        pass
