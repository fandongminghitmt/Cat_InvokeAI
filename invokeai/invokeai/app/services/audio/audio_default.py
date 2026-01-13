from typing import Optional
import io

from invokeai.app.services.audio.audio_base import AudioServiceABC
from invokeai.app.services.images.images_common import AudioDTO
from invokeai.app.services.image_records.image_records_common import ResourceOrigin, ImageCategory
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.image_records.audio_records_sqlite import SqliteAudioRecordStorage
from invokeai.app.services.image_files.audio_files_disk import DiskAudioFileStorage

class AudioService(AudioServiceABC):
    __invoker: Invoker
    __records: SqliteAudioRecordStorage
    __files: DiskAudioFileStorage

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self.__records = SqliteAudioRecordStorage(invoker.services.image_records._db)
        output_folder = invoker.services.configuration.outputs_path
        self.__files = DiskAudioFileStorage(f"{output_folder}/audio")

    def create(
        self,
        audio_file: bytes,
        audio_origin: ResourceOrigin,
        audio_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
    ) -> AudioDTO:
        
        audio_name = self.__invoker.services.names.create_image_name().replace(".png", ".mp3") # Hacky extension swap
        
        # Placeholder for metadata extraction
        duration = 60.0
        
        self.__files.save(audio_name, audio_file)
        
        self.__records.save(
            audio_name=audio_name,
            audio_origin=audio_origin,
            audio_category=audio_category,
            duration=duration,
            has_workflow=False,
            is_intermediate=is_intermediate,
            node_id=node_id,
            session_id=session_id
        )

        return self.get_dto(audio_name)

    def get_dto(self, audio_name: str) -> AudioDTO:
        record = self.__records.get(audio_name)
        base_url = "api/v1/media/audio"
        
        return AudioDTO(
            audio_name=record.audio_name,
            audio_origin=record.audio_origin,
            audio_category=record.audio_category,
            audio_url=f"{base_url}/{audio_name}",
            duration=record.duration,
            created_at=str(record.created_at),
            updated_at=str(record.updated_at),
            deleted_at=str(record.deleted_at) if record.deleted_at else None,
            is_intermediate=record.is_intermediate,
            session_id=record.session_id,
            node_id=record.node_id,
            starred=record.starred,
            has_workflow=record.has_workflow,
            board_id=None
        )
