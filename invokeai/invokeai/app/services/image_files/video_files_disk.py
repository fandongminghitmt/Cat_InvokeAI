from pathlib import Path
from typing import Union, Optional
from PIL import Image
import shutil

from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from invokeai.app.util.thumbnails import get_thumbnail_name

class DiskVideoFileStorage:
    """Stores video files on disk"""

    def __init__(self, output_folder: Union[str, Path]):
        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__thumbnails_folder = self.__output_folder / "thumbnails"
        self.__validate_storage_folders()

    def get_path(self, video_name: str, thumbnail: bool = False) -> Path:
        base_folder = self.__thumbnails_folder if thumbnail else self.__output_folder
        filename = get_thumbnail_name(video_name) if thumbnail else video_name
        return base_folder / filename

    def save(self, video_name: str, video_data: bytes, thumbnail: Optional[Image.Image] = None) -> None:
        try:
            self.__validate_storage_folders()
            video_path = self.get_path(video_name)

            with open(video_path, "wb") as f:
                f.write(video_data)

            if thumbnail:
                thumbnail_name = get_thumbnail_name(video_name)
                thumbnail_path = self.get_path(thumbnail_name, thumbnail=True)
                thumbnail.save(thumbnail_path)
                
        except Exception as e:
            raise ImageFileSaveException from e

    def validate_path(self, path: Union[str, Path]) -> bool:
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()

    def __validate_storage_folders(self) -> None:
        folders: list[Path] = [self.__output_folder, self.__thumbnails_folder]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
