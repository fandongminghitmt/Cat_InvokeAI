from typing import Optional

from PIL.Image import Image as PILImageType

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageNamesResult,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    InvalidImageCategoryException,
    InvalidOriginException,
    ResourceOrigin,
)
from invokeai.app.services.images.images_base import ImageServiceABC
from invokeai.app.services.images.images_common import ImageDTO, image_record_to_dto
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class ImageService(ImageServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def create(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
    ) -> ImageDTO:
        if image_origin not in ResourceOrigin:
            raise InvalidOriginException

        if image_category not in ImageCategory:
            raise InvalidImageCategoryException

        image_name = self.__invoker.services.names.create_image_name()

        (width, height) = image.size

        try:
            # TODO: Consider using a transaction here to ensure consistency between storage and database
            self.__invoker.services.image_records.save(
                # Non-nullable fields
                image_name=image_name,
                image_origin=image_origin,
                image_category=image_category,
                width=width,
                height=height,
                has_workflow=workflow is not None or graph is not None,
                # Meta fields
                is_intermediate=is_intermediate,
                # Nullable fields
                node_id=node_id,
                metadata=metadata,
                session_id=session_id,
            )
            if board_id is not None:
                try:
                    self.__invoker.services.board_image_records.add_image_to_board(
                        board_id=board_id, image_name=image_name
                    )
                except Exception as e:
                    self.__invoker.services.logger.warning(f"Failed to add image to board {board_id}: {str(e)}")
            self.__invoker.services.image_files.save(
                image_name=image_name, image=image, metadata=metadata, workflow=workflow, graph=graph
            )
            image_dto = self.get_dto(image_name)

            self._on_changed(image_dto)
            return image_dto
        except ImageRecordSaveException:
            self.__invoker.services.logger.error("Failed to save image record")
            raise
        except ImageFileSaveException:
            self.__invoker.services.logger.error("Failed to save image file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem saving image record and file: {str(e)}")
            raise e

    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> ImageDTO:
        try:
            self.__invoker.services.image_records.update(image_name, changes)
            image_dto = self.get_dto(image_name)
            self._on_changed(image_dto)
            return image_dto
        except ImageRecordSaveException:
            self.__invoker.services.logger.error("Failed to update image record")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem updating image record")
            raise e

    def get_pil_image(self, image_name: str) -> PILImageType:
        try:
            return self.__invoker.services.image_files.get(image_name)
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Failed to get image file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image file")
            raise e

    def get_record(self, image_name: str) -> ImageRecord:
        try:
            return self.__invoker.services.image_records.get(image_name)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image record")
            raise e

    def get_dto(self, image_name: str) -> ImageDTO:
        try:
            # Try images first
            try:
                image_record = self.__invoker.services.image_records.get(image_name)
                
                image_dto = image_record_to_dto(
                    image_record=image_record,
                    image_url=self.__invoker.services.urls.get_image_url(image_name),
                    thumbnail_url=self.__invoker.services.urls.get_image_url(image_name, True),
                    board_id=self.__invoker.services.board_image_records.get_board_for_image(image_name),
                )
                return image_dto
            except ImageRecordNotFoundException:
                # Try videos
                try:
                    video_service = getattr(self.__invoker.services, 'videos', None)
                    if video_service:
                        video_record = video_service._VideoService__records.get(image_name) # Accessing private member for quick hack, better to expose method
                        # Or use video_service.get_dto but that returns VideoDTO
                        
                        # Let's manually construct ImageDTO from VideoRecord
                        # We treat video_url as image_url, but frontend might get confused if it tries to load it as img
                        # Ideally image_url should point to thumbnail for gallery view?
                        # But ImageViewer needs the real URL.
                        
                        base_url = "api/v1/media/video"
                        return ImageDTO(
                            image_name=video_record.video_name,
                            image_origin=video_record.video_origin,
                            image_category="general", # Hack: Force category to 'general' for frontend visibility
                            image_url=f"{base_url}/{video_record.video_name}/full",
                            thumbnail_url=f"{base_url}/{video_record.video_name}/thumbnail",
                            width=video_record.width,
                            height=video_record.height,
                            created_at=str(video_record.created_at),
                            updated_at=str(video_record.updated_at),
                            deleted_at=str(video_record.deleted_at) if video_record.deleted_at else None,
                            is_intermediate=video_record.is_intermediate,
                            session_id=video_record.session_id,
                            node_id=video_record.node_id,
                            starred=video_record.starred,
                            has_workflow=video_record.has_workflow,
                            board_id=None
                        )
                except Exception:
                    pass
                
                # Try audios
                try:
                    audio_service = getattr(self.__invoker.services, 'audios', None)
                    if audio_service:
                        audio_record = audio_service._AudioService__records.get(image_name)
                        base_url = "api/v1/media/audio"
                        return ImageDTO(
                            image_name=audio_record.audio_name,
                            image_origin=audio_record.audio_origin,
                            image_category="general", # Hack: Force category to 'general' for frontend visibility
                            image_url=f"{base_url}/{audio_record.audio_name}",
                            thumbnail_url="", # Hack: Frontend requires string, not None.
                            width=1, # Hack: Frontend requires width > 0
                            height=1, # Hack: Frontend requires height > 0
                            created_at=str(audio_record.created_at),
                            updated_at=str(audio_record.updated_at),
                            deleted_at=str(audio_record.deleted_at) if audio_record.deleted_at else None,
                            is_intermediate=audio_record.is_intermediate,
                            session_id=audio_record.session_id,
                            node_id=audio_record.node_id,
                            starred=audio_record.starred,
                            has_workflow=audio_record.has_workflow,
                            board_id=None
                        )
                except Exception:
                    pass
                
                raise ImageRecordNotFoundException

        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image/Media record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image/media DTO")
            raise e

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        try:
            return self.__invoker.services.image_records.get_metadata(image_name)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image metadata")
            raise e

    def get_workflow(self, image_name: str) -> Optional[str]:
        try:
            return self.__invoker.services.image_files.get_workflow(image_name)
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Image file not found")
            raise
        except Exception:
            self.__invoker.services.logger.error("Problem getting image workflow")
            raise

    def get_graph(self, image_name: str) -> Optional[str]:
        try:
            return self.__invoker.services.image_files.get_graph(image_name)
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Image file not found")
            raise
        except Exception:
            self.__invoker.services.logger.error("Problem getting image graph")
            raise

    def get_path(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return str(self.__invoker.services.image_files.get_path(image_name, thumbnail))
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image path")
            raise e

    def validate_path(self, path: str) -> bool:
        try:
            return self.__invoker.services.image_files.validate_path(path)
        except Exception as e:
            self.__invoker.services.logger.error("Problem validating image path")
            raise e

    def get_url(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return self.__invoker.services.urls.get_image_url(image_name, thumbnail)
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image path")
            raise e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> OffsetPaginatedResults[ImageDTO]:
        try:
            results = self.__invoker.services.image_records.get_many(
                offset,
                limit,
                starred_first,
                order_dir,
                image_origin,
                categories,
                is_intermediate,
                board_id,
                search_term,
            )

            image_dtos = []
            for r in results.items:
                # Hack: Fix dimensions for Audio/Video if they are 0, to pass frontend validation
                if r.width == 0: r.width = 1
                if r.height == 0: r.height = 1
                
                image_dtos.append(image_record_to_dto(
                    image_record=r,
                    image_url=self.__invoker.services.urls.get_image_url(r.image_name),
                    thumbnail_url=self.__invoker.services.urls.get_image_url(r.image_name, True),
                    board_id=self.__invoker.services.board_image_records.get_board_for_image(r.image_name),
                ))

            return OffsetPaginatedResults[ImageDTO](
                items=image_dtos,
                offset=results.offset,
                limit=results.limit,
                total=results.total,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting paginated image DTOs")
            raise e

    def delete(self, image_name: str):
        try:
            try:
                self.__invoker.services.image_files.delete(image_name)
            except Exception:
                self.__invoker.services.logger.warning(f"Failed to delete image file for {image_name}, but proceeding to delete record.")
            
            self.__invoker.services.image_records.delete(image_name)
            self._on_deleted(image_name)
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image record")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting image record and file")
            raise e

    def delete_images_on_board(self, board_id: str):
        try:
            image_names = self.__invoker.services.board_image_records.get_all_board_image_names_for_board(
                board_id,
                categories=None,
                is_intermediate=None,
            )
            for image_name in image_names:
                self.__invoker.services.image_files.delete(image_name)
            self.__invoker.services.image_records.delete_many(image_names)
            for image_name in image_names:
                self._on_deleted(image_name)
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem deleting image records and files: {str(e)}")
            raise e

    def delete_intermediates(self) -> int:
        try:
            image_names = self.__invoker.services.image_records.delete_intermediates()
            count = len(image_names)
            for image_name in image_names:
                self.__invoker.services.image_files.delete(image_name)
                self._on_deleted(image_name)
            return count
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting image records and files")
            raise e

    def get_intermediates_count(self) -> int:
        try:
            return self.__invoker.services.image_records.get_intermediates_count()
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting intermediates count")
            raise e

    def get_image_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
    ) -> ImageNamesResult:
        try:
            return self.__invoker.services.image_records.get_image_names(
                starred_first=starred_first,
                order_dir=order_dir,
                image_origin=image_origin,
                categories=categories,
                is_intermediate=is_intermediate,
                board_id=board_id,
                search_term=search_term,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image names")
            raise e
