from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

from PIL.Image import Image
from pydantic.networks import AnyHttpUrl
from torch import Tensor

from invokeai.app.invocations.constants import IMAGE_MODES
from invokeai.app.invocations.fields import MetadataField, WithBoard, WithMetadata
from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy
from invokeai.app.services.boards.boards_common import BoardDTO
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import ImageDTO, VideoDTO, AudioDTO
from invokeai.app.services.video.video_base import VideoServiceABC
from invokeai.app.services.audio.audio_base import AudioServiceABC
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.model_records.model_records_base import UnknownModelException
from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.util.step_callback import diffusion_step_callback
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_base import LoadedModel, LoadedModelWithoutConfig
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData

if TYPE_CHECKING:
    from invokeai.app.invocations.baseinvocation import BaseInvocation
    from invokeai.app.invocations.model import ModelIdentifierField
    from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem

"""
The InvocationContext provides access to various services and data about the current invocation.

We do not provide the invocation services directly, as their methods are both dangerous and
inconvenient to use.

For example:
- The `images` service allows nodes to delete or unsafely modify existing images.
- The `configuration` service allows nodes to change the app's config at runtime.
- The `events` service allows nodes to emit arbitrary events.

Wrapping these services provides a simpler and safer interface for nodes to use.

When a node executes, a fresh `InvocationContext` is built for it, ensuring nodes cannot interfere
with each other.

Many of the wrappers have the same signature as the methods they wrap. This allows us to write
user-facing docstrings and not need to go and update the internal services to match.

Note: The docstrings are in weird places, but that's where they must be to get IDEs to see them.
"""


@dataclass
class InvocationContextData:
    queue_item: "SessionQueueItem"
    """The queue item that is being executed."""
    invocation: "BaseInvocation"
    """The invocation that is being executed."""
    source_invocation_id: str
    """The ID of the invocation from which the currently executing invocation was prepared."""


class InvocationContextInterface:
    def __init__(self, services: InvocationServices, data: InvocationContextData) -> None:
        self._services = services
        self._data = data


class BoardsInterface(InvocationContextInterface):
    def create(self, board_name: str) -> BoardDTO:
        """Creates a board.

        Args:
            board_name: The name of the board to create.

        Returns:
            The created board DTO.
        """
        return self._services.boards.create(board_name)

    def get_dto(self, board_id: str) -> BoardDTO:
        """Gets a board DTO.

        Args:
            board_id: The ID of the board to get.

        Returns:
            The board DTO.
        """
        return self._services.boards.get_dto(board_id)

    def get_all(self) -> list[BoardDTO]:
        """Gets all boards.

        Returns:
            A list of all boards.
        """
        return self._services.boards.get_all(
            order_by=BoardRecordOrderBy.CreatedAt, direction=SQLiteDirection.Descending
        )

    def add_image_to_board(self, board_id: str, image_name: str) -> None:
        """Adds an image to a board.

        Args:
            board_id: The ID of the board to add the image to.
            image_name: The name of the image to add to the board.
        """
        return self._services.board_images.add_image_to_board(board_id, image_name)

    def get_all_image_names_for_board(self, board_id: str) -> list[str]:
        """Gets all image names for a board.

        Args:
            board_id: The ID of the board to get the image names for.

        Returns:
            A list of all image names for the board.
        """
        return self._services.board_images.get_all_board_image_names_for_board(
            board_id,
            categories=None,
            is_intermediate=None,
        )


class LoggerInterface(InvocationContextInterface):
    def debug(self, message: str) -> None:
        """Logs a debug message.

        Args:
            message: The message to log.
        """
        self._services.logger.debug(message)

    def info(self, message: str) -> None:
        """Logs an info message.

        Args:
            message: The message to log.
        """
        self._services.logger.info(message)

    def warning(self, message: str) -> None:
        """Logs a warning message.

        Args:
            message: The message to log.
        """
        self._services.logger.warning(message)

    def error(self, message: str) -> None:
        """Logs an error message.

        Args:
            message: The message to log.
        """
        self._services.logger.error(message)


class ImagesInterface(InvocationContextInterface):
    def __init__(self, services: InvocationServices, data: InvocationContextData, util: "UtilInterface") -> None:
        super().__init__(services, data)
        self._util = util

    def save(
        self,
        image: Image,
        board_id: Optional[str] = None,
        image_category: ImageCategory = ImageCategory.GENERAL,
        metadata: Optional[MetadataField] = None,
    ) -> ImageDTO:
        """Saves an image, returning its DTO.

        If the current queue item has a workflow or metadata, it is automatically saved with the image.

        Args:
            image: The image to save, as a PIL image.
            board_id: The board ID to add the image to, if it should be added. It the invocation \
            inherits from `WithBoard`, that board will be used automatically. **Use this only if \
            you want to override or provide a board manually!**
            image_category: The category of the image. Only the GENERAL category is added \
            to the gallery.
            metadata: The metadata to save with the image, if it should have any. If the \
            invocation inherits from `WithMetadata`, that metadata will be used automatically. \
            **Use this only if you want to override or provide metadata manually!**

        Returns:
            The saved image DTO.
        """

        self._util.signal_progress("Saving image")

        # If `metadata` is provided directly, use that. Else, use the metadata provided by `WithMetadata`, falling back to None.
        metadata_ = None
        if metadata:
            metadata_ = metadata.model_dump_json()
        elif isinstance(self._data.invocation, WithMetadata) and self._data.invocation.metadata:
            metadata_ = self._data.invocation.metadata.model_dump_json()

        # If `board_id` is provided directly, use that. Else, use the board provided by `WithBoard`, falling back to None.
        board_id_ = None
        if board_id:
            board_id_ = board_id
        elif isinstance(self._data.invocation, WithBoard) and self._data.invocation.board:
            board_id_ = self._data.invocation.board.board_id

        workflow_ = None
        if self._data.queue_item.workflow:
            workflow_ = self._data.queue_item.workflow.model_dump_json()

        graph_ = None
        if self._data.queue_item.session.graph:
            graph_ = self._data.queue_item.session.graph.model_dump_json()

        return self._services.images.create(
            image=image,
            is_intermediate=self._data.invocation.is_intermediate,
            image_category=image_category,
            board_id=board_id_,
            metadata=metadata_,
            image_origin=ResourceOrigin.INTERNAL,
            workflow=workflow_,
            graph=graph_,
            session_id=self._data.queue_item.session_id,
            node_id=self._data.invocation.id,
        )

    def get_pil(self, image_name: str, mode: IMAGE_MODES | None = None) -> Image:
        """Gets an image as a PIL Image object. This method returns a copy of the image.

        Args:
            image_name: The name of the image to get.
            mode: The color mode to convert the image to. If None, the original mode is used.

        Returns:
            The image as a PIL Image object.
        """
        image = self._services.images.get_pil_image(image_name)
        if mode and mode != image.mode:
            try:
                # convert makes a copy!
                image = image.convert(mode)
            except ValueError:
                self._services.logger.warning(
                    f"Could not convert image from {image.mode} to {mode}. Using original mode instead."
                )
        else:
            # copy the image to prevent the user from modifying the original
            image = image.copy()
        return image

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        """Gets an image's metadata, if it has any.

        Args:
            image_name: The name of the image to get the metadata for.

        Returns:
            The image's metadata, if it has any.
        """
        return self._services.images.get_metadata(image_name)

    def get_dto(self, image_name: str) -> ImageDTO:
        """Gets an image as an ImageDTO object.

        Args:
            image_name: The name of the image to get.

        Returns:
            The image as an ImageDTO object.
        """
        return self._services.images.get_dto(image_name)

    def get_path(self, image_name: str, thumbnail: bool = False) -> Path:
        """Gets the internal path to an image or thumbnail.

        Args:
            image_name: The name of the image to get the path of.
            thumbnail: Get the path of the thumbnail instead of the full image

        Returns:
            The local path of the image or thumbnail.
        """
        return Path(self._services.images.get_path(image_name, thumbnail))

class VideosInterface(InvocationContextInterface):
    def __init__(self, services: InvocationServices, data: InvocationContextData) -> None:
        super().__init__(services, data)

    def get_dto(self, video_name: str) -> VideoDTO:
        """Gets a video as a VideoDTO object.

        Args:
            video_name: The name of the video to get.

        Returns:
            The video as a VideoDTO object.
        """
        return self._services.videos.get_dto(video_name)


class AudiosInterface(InvocationContextInterface):
    def __init__(self, services: InvocationServices, data: InvocationContextData) -> None:
        super().__init__(services, data)

    def get_dto(self, audio_name: str) -> AudioDTO:
        """Gets an audio as a AudioDTO object.

        Args:
            audio_name: The name of the audio to get.

        Returns:
            The audio as a AudioDTO object.
        """
        return self._services.audios.get_dto(audio_name)


    """

    def __init__(
        self,
        images: ImagesInterface,
        tensors: TensorsInterface,
        conditioning: ConditioningInterface,
        models: ModelsInterface,
        logger: LoggerInterface,
        config: ConfigInterface,
        util: UtilInterface,
        boards: BoardsInterface,
        videos: VideosInterface,
        audios: AudiosInterface,
        """Methods to interact with boards."""
        self._data = data
        """An internal API providing access to data about the current queue item and invocation. You probably shouldn't use this. It may change without warning."""
        self._services = services
        """An internal API providing access to all application services. You probably shouldn't use this. It may change without warning."""


def build_invocation_context(
    services: InvocationServices,
    data: InvocationContextData,
    is_canceled: Callable[[], bool],
) -> InvocationContext:
    """Builds the invocation context for a specific invocation execution.

    Args:
        services: The invocation services to wrap.
        data: The invocation context data.

    Returns:
        The invocation context.
    """

    logger = LoggerInterface(services=services, data=data)
    tensors = TensorsInterface(services=services, data=data)
    config = ConfigInterface(services=services, data=data)
    util = UtilInterface(services=services, data=data, is_canceled=is_canceled)
    conditioning = ConditioningInterface(services=services, data=data)
    models = ModelsInterface(services=services, data=data, util=util)
    images = ImagesInterface(services=services, data=data, util=util)
    boards = BoardsInterface(services=services, data=data)
    videos = VideosInterface(services=services, data=data)
    audios = AudiosInterface(services=services, data=data)
    )

    return ctx
