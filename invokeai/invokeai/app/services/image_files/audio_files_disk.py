from pathlib import Path
from typing import Union, Optional
import shutil

from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)

class DiskAudioFileStorage:
    """Stores audio files on disk"""

    def __init__(self, output_folder: Union[str, Path]):
        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__validate_storage_folders()

    def get_path(self, audio_name: str) -> Path:
        return self.__output_folder / audio_name

    def save(self, audio_name: str, audio_data: bytes) -> None:
        try:
            self.__validate_storage_folders()
            audio_path = self.get_path(audio_name)

            with open(audio_path, "wb") as f:
                f.write(audio_data)
                
        except Exception as e:
            raise ImageFileSaveException from e

    def validate_path(self, path: Union[str, Path]) -> bool:
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()

    def __validate_storage_folders(self) -> None:
        self.__output_folder.mkdir(parents=True, exist_ok=True)
