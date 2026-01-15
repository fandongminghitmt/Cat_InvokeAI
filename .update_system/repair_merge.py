import os
import sys

def write_file(path, content):
    print(f"Writing {path}...")
    try:
        # Ensure dir exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Success.")
    except Exception as e:
        print(f"Error writing {path}: {e}")
        sys.exit(1)

def repair_invocation_context():
    # Based on official SHA 97291230e045f69a64ce414128ea1d9ebf578ea3
    # Added: Video/Audio imports, Interfaces, and Context fields
    content = r'''from copy import deepcopy
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


class TensorsInterface(InvocationContextInterface):
    def save(self, tensor: Tensor) -> str:
        """Saves a tensor, returning its name.

        Args:
            tensor: The tensor to save.

        Returns:
            The name of the saved tensor.
        """

        name = self._services.tensors.save(obj=tensor)
        return name

    def load(self, name: str) -> Tensor:
        """Loads a tensor by name. This method returns a copy of the tensor.

        Args:
            name: The name of the tensor to load.

        Returns:
            The tensor.
        """
        return self._services.tensors.load(name).clone()


class ConditioningInterface(InvocationContextInterface):
    def save(self, conditioning_data: ConditioningFieldData) -> str:
        """Saves a conditioning data object, returning its name.

        Args:
            conditioning_data: The conditioning data to save.

        Returns:
            The name of the saved conditioning data.
        """

        name = self._services.conditioning.save(obj=conditioning_data)
        return name

    def load(self, name: str) -> ConditioningFieldData:
        """Loads conditioning data by name. This method returns a copy of the conditioning data.

        Args:
            name: The name of the conditioning data to load.

        Returns:
            The conditioning data.
        """

        return deepcopy(self._services.conditioning.load(name))


class ModelsInterface(InvocationContextInterface):
    """Common API for loading, downloading and managing models."""

    def __init__(self, services: InvocationServices, data: InvocationContextData, util: "UtilInterface") -> None:
        super().__init__(services, data)
        self._util = util

    def exists(self, identifier: Union[str, "ModelIdentifierField"]) -> bool:
        """Check if a model exists.

        Args:
            identifier: The key or ModelField representing the model.

        Returns:
            True if the model exists, False if not.
        """
        if isinstance(identifier, str):
            return self._services.model_manager.store.exists(identifier)
        else:
            return self._services.model_manager.store.exists(identifier.key)

    def load(
        self, identifier: Union[str, "ModelIdentifierField"], submodel_type: Optional[SubModelType] = None
    ) -> LoadedModel:
        """Load a model.

        Args:
            identifier: The key or ModelField representing the model.
            submodel_type: The submodel of the model to get.

        Returns:
            An object representing the loaded model.
        """

        # The model manager emits events as it loads the model. It needs the context data to build
        # the event payloads.

        if isinstance(identifier, str):
            model = self._services.model_manager.store.get_model(identifier)
        else:
            submodel_type = submodel_type or identifier.submodel_type
            model = self._services.model_manager.store.get_model(identifier.key)

        message = f"Loading model {model.name}"
        if submodel_type:
            message += f" ({submodel_type.value})"
        self._util.signal_progress(message)
        return self._services.model_manager.load.load_model(model, submodel_type)

    def load_by_attrs(
        self, name: str, base: BaseModelType, type: ModelType, submodel_type: Optional[SubModelType] = None
    ) -> LoadedModel:
        """Load a model by its attributes.

        Args:
            name: Name of the model.
            base: The models' base type, e.g. `BaseModelType.StableDiffusion1`, `BaseModelType.StableDiffusionXL`, etc.
            type: Type of the model, e.g. `ModelType.Main`, `ModelType.Vae`, etc.
            submodel_type: The type of submodel to load, e.g. `SubModelType.UNet`, `SubModelType.TextEncoder`, etc. Only main
            models have submodels.

        Returns:
            An object representing the loaded model.
        """

        configs = self._services.model_manager.store.search_by_attr(model_name=name, base_model=base, model_type=type)
        if len(configs) == 0:
            raise UnknownModelException(f"No model found with name {name}, base {base}, and type {type}")

        if len(configs) > 1:
            raise ValueError(f"More than one model found with name {name}, base {base}, and type {type}")

        message = f"Loading model {name}"
        if submodel_type:
            message += f" ({submodel_type.value})"
        self._util.signal_progress(message)
        return self._services.model_manager.load.load_model(configs[0], submodel_type)

    def get_config(self, identifier: Union[str, "ModelIdentifierField"]) -> AnyModelConfig:
        """Get a model's config.

        Args:
            identifier: The key or ModelField representing the model.

        Returns:
            The model's config.
        """
        if isinstance(identifier, str):
            return self._services.model_manager.store.get_model(identifier)
        else:
            return self._services.model_manager.store.get_model(identifier.key)

    def search_by_path(self, path: Path) -> list[AnyModelConfig]:
        """Search for models by path.

        Args:
            path: The path to search for.

        Returns:
            A list of models that match the path.
        """
        return self._services.model_manager.store.search_by_path(path)

    def search_by_attrs(
        self,
        name: Optional[str] = None,
        base: Optional[BaseModelType] = None,
        type: Optional[ModelType] = None,
        format: Optional[ModelFormat] = None,
    ) -> list[AnyModelConfig]:
        """Search for models by attributes.

        Args:
            name: The name to search for (exact match).
            base: The base to search for, e.g. `BaseModelType.StableDiffusion1`, `BaseModelType.StableDiffusionXL`, etc.
            type: Type type of model to search for, e.g. `ModelType.Main`, `ModelType.Vae`, etc.
            format: The format of model to search for, e.g. `ModelFormat.Checkpoint`, `ModelFormat.Diffusers`, etc.

        Returns:
            A list of models that match the attributes.
        """

        return self._services.model_manager.store.search_by_attr(
            model_name=name,
            base_model=base,
            model_type=type,
            model_format=format,
        )

    def download_and_cache_model(
        self,
        source: str | AnyHttpUrl,
    ) -> Path:
        """
        Download the model file located at source to the models cache and return its Path.

        This can be used to single-file install models and other resources of arbitrary types
        which should not get registered with the database. If the model is already
        installed, the cached path will be returned. Otherwise it will be downloaded.

        Args:
            source: A URL that points to the model, or a huggingface repo_id.

        Returns:
            Path to the downloaded model
        """
        self._util.signal_progress(f"Downloading model {source}")
        return self._services.model_manager.load.download_and_cache_model(source)


class ConfigInterface(InvocationContextInterface):
    def get(self) -> InvokeAIAppConfig:
        """Gets the app config.

        Returns:
            The app config.
        """
        return self._services.configuration.get_config()


class UtilInterface(InvocationContextInterface):
    def __init__(
        self, services: InvocationServices, data: InvocationContextData, is_canceled: Callable[[], bool]
    ) -> None:
        super().__init__(services, data)
        self._is_canceled = is_canceled

    def is_canceled(self) -> bool:
        """Checks if the current invocation has been canceled.

        Returns:
            True if the invocation has been canceled, False otherwise.
        """
        return self._is_canceled()

    def signal_progress(
        self,
        message: Optional[str] = None,
        percentage: Optional[float] = None,
        image: Optional[ProgressImage] = None,
    ) -> None:
        """Signals progress to the session processor.

        Args:
            message: The message to display.
            percentage: The percentage of progress, between 0 and 1.
            image: The image to display.
        """
        self._services.session_processor.signal_progress(
            self._data.queue_item,
            self._data.invocation,
            self._data.source_invocation_id,
            message,
            percentage,
            image,
        )

    def sd_step_callback(self, intermediate_state: PipelineIntermediateState, base_model: BaseModelType) -> None:
        """Callback to be called at each step of the diffusion process.

        Args:
            intermediate_state: The intermediate state of the diffusion process.
            base_model: The base model of the diffusion process.
        """
        diffusion_step_callback(
            self._services.session_processor,
            self._data.queue_item,
            self._data.invocation,
            self._data.source_invocation_id,
            intermediate_state,
            base_model,
        )


class InvocationContext:
    """
    The InvocationContext provides access to various services and data about the current invocation.
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
        data: InvocationContextData,
        services: InvocationServices,
    ) -> None:
        self.images = images
        """Methods to interact with images."""
        self.tensors = tensors
        """Methods to interact with tensors."""
        self.conditioning = conditioning
        """Methods to interact with conditioning data."""
        self.models = models
        """Methods to interact with models."""
        self.logger = logger
        """Methods to log messages."""
        self.config = config
        """Methods to interact with the app config."""
        self.util = util
        """Methods to interact with utility services."""
        self.boards = boards
        """Methods to interact with boards."""
        self.videos = videos
        """Methods to interact with videos."""
        self.audios = audios
        """Methods to interact with audios."""
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

    ctx = InvocationContext(
        images=images,
        tensors=tensors,
        conditioning=conditioning,
        models=models,
        logger=logger,
        config=config,
        util=util,
        boards=boards,
        videos=videos,
        audios=audios,
        data=data,
        services=services,
    )

    return ctx
'''
    write_file("d:\\Cat_InvokeAI\\invokeai\\invokeai\\app\\services\\shared\\invocation_context.py", content)

def repair_invocation_services():
    # Based on official SHA 52fb064596da015cf048ec02a63d01be3acd9c37
    # Added: Video/Audio services fields
    content = r'''# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
from __future__ import annotations

from typing import TYPE_CHECKING

from invokeai.app.services.object_serializer.object_serializer_base import ObjectSerializerBase
from invokeai.app.services.style_preset_images.style_preset_images_base import StylePresetImageFileStorageBase
from invokeai.app.services.style_preset_records.style_preset_records_base import StylePresetRecordsStorageBase

if TYPE_CHECKING:
    from logging import Logger

    import torch

    from invokeai.app.services.board_image_records.board_image_records_base import BoardImageRecordStorageBase
    from invokeai.app.services.board_images.board_images_base import BoardImagesServiceABC
    from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
    from invokeai.app.services.boards.boards_base import BoardServiceABC
    from invokeai.app.services.bulk_download.bulk_download_base import BulkDownloadBase
    from invokeai.app.services.client_state_persistence.client_state_persistence_base import ClientStatePersistenceABC
    from invokeai.app.services.config import InvokeAIAppConfig
    from invokeai.app.services.download import DownloadQueueServiceBase
    from invokeai.app.services.events.events_base import EventServiceBase
    from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
    from invokeai.app.services.image_records.image_records_base import ImageRecordStorageBase
    from invokeai.app.services.images.images_base import ImageServiceABC
    from invokeai.app.services.video.video_base import VideoServiceABC
    from invokeai.app.services.audio.audio_base import AudioServiceABC
    from invokeai.app.services.invocation_cache.invocation_cache_base import InvocationCacheBase
    from invokeai.app.services.invocation_stats.invocation_stats_base import InvocationStatsServiceBase
    from invokeai.app.services.model_images.model_images_base import ModelImageFileStorageBase
    from invokeai.app.services.model_manager.model_manager_base import ModelManagerServiceBase
    from invokeai.app.services.model_relationship_records.model_relationship_records_base import (
        ModelRelationshipRecordStorageBase,
    )
    from invokeai.app.services.model_relationships.model_relationships_base import ModelRelationshipsServiceABC
    from invokeai.app.services.names.names_base import NameServiceBase
    from invokeai.app.services.session_processor.session_processor_base import SessionProcessorBase
    from invokeai.app.services.session_queue.session_queue_base import SessionQueueBase
    from invokeai.app.services.urls.urls_base import UrlServiceBase
    from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
    from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_base import WorkflowThumbnailServiceBase
    from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData


class InvocationServices:
    """Services that can be used by invocations"""

    def __init__(
        self,
        board_images: "BoardImagesServiceABC",
        board_image_records: "BoardImageRecordStorageBase",
        boards: "BoardServiceABC",
        board_records: "BoardRecordStorageBase",
        bulk_download: "BulkDownloadBase",
        configuration: "InvokeAIAppConfig",
        events: "EventServiceBase",
        images: "ImageServiceABC",
        videos: "VideoServiceABC",
        audios: "AudioServiceABC",
        image_files: "ImageFileStorageBase",
        image_records: "ImageRecordStorageBase",
        logger: "Logger",
        model_images: "ModelImageFileStorageBase",
        model_manager: "ModelManagerServiceBase",
        model_relationships: "ModelRelationshipsServiceABC",
        model_relationship_records: "ModelRelationshipRecordStorageBase",
        download_queue: "DownloadQueueServiceBase",
        performance_statistics: "InvocationStatsServiceBase",
        session_queue: "SessionQueueBase",
        session_processor: "SessionProcessorBase",
        invocation_cache: "InvocationCacheBase",
        names: "NameServiceBase",
        urls: "UrlServiceBase",
        workflow_records: "WorkflowRecordsStorageBase",
        tensors: "ObjectSerializerBase[torch.Tensor]",
        conditioning: "ObjectSerializerBase[ConditioningFieldData]",
        style_preset_records: "StylePresetRecordsStorageBase",
        style_preset_image_files: "StylePresetImageFileStorageBase",
        workflow_thumbnails: "WorkflowThumbnailServiceBase",
        client_state_persistence: "ClientStatePersistenceABC",
    ):
        self.board_images = board_images
        self.board_image_records = board_image_records
        self.boards = boards
        self.board_records = board_records
        self.bulk_download = bulk_download
        self.configuration = configuration
        self.events = events
        self.images = images
        self.videos = videos
        self.audios = audios
        self.image_files = image_files
        self.image_records = image_records
        self.logger = logger
        self.model_images = model_images
        self.model_manager = model_manager
        self.model_relationships = model_relationships
        self.model_relationship_records = model_relationship_records
        self.download_queue = download_queue
        self.performance_statistics = performance_statistics
        self.session_queue = session_queue
        self.session_processor = session_processor
        self.invocation_cache = invocation_cache
        self.names = names
        self.urls = urls
        self.workflow_records = workflow_records
        self.tensors = tensors
        self.conditioning = conditioning
        self.style_preset_records = style_preset_records
        self.style_preset_image_files = style_preset_image_files
        self.workflow_thumbnails = workflow_thumbnails
        self.client_state_persistence = client_state_persistence
'''
    write_file("d:\\Cat_InvokeAI\\invokeai\\invokeai\\app\\services\\invocation_services.py", content)

def repair_dependencies():
    # Based on official SHA 466a57f804c14bb1c51610da98ec29391c67d777
    # Added: Video/Audio service initialization and passing to InvocationServices
    content = r'''# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import asyncio
from logging import Logger

import torch

from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_images.board_images_default import BoardImagesService
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.boards.boards_default import BoardService
from invokeai.app.services.bulk_download.bulk_download_default import BulkDownloadService
from invokeai.app.services.client_state_persistence.client_state_persistence_sqlite import ClientStatePersistenceSqlite
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.download.download_default import DownloadQueueService
from invokeai.app.services.events.events_fastapievents import FastAPIEventService
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.video.video_default import VideoService
from invokeai.app.services.audio.audio_default import AudioService
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.invocation_stats.invocation_stats_default import InvocationStatsService
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_images.model_images_default import ModelImageFileStorageDisk
from invokeai.app.services.model_manager.model_manager_default import ModelManagerService
from invokeai.app.services.model_records.model_records_sql import ModelRecordServiceSQL
from invokeai.app.services.model_relationship_records.model_relationship_records_sqlite import (
    SqliteModelRelationshipRecordStorage,
)
from invokeai.app.services.model_relationships.model_relationships_default import ModelRelationshipsService
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.object_serializer.object_serializer_disk import ObjectSerializerDisk
from invokeai.app.services.object_serializer.object_serializer_forward_cache import ObjectSerializerForwardCache
from invokeai.app.services.session_processor.session_processor_default import (
    DefaultSessionProcessor,
    DefaultSessionRunner,
)
from invokeai.app.services.session_queue.session_queue_sqlite import SqliteSessionQueue
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.app.services.style_preset_images.style_preset_images_disk import StylePresetImageFileStorageDisk
from invokeai.app.services.style_preset_records.style_preset_records_sqlite import SqliteStylePresetRecordsStorage
from invokeai.app.services.urls.urls_default import LocalUrlService
from invokeai.app.services.workflow_records.workflow_records_sqlite import SqliteWorkflowRecordsStorage
from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_disk import WorkflowThumbnailFileStorageDisk
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    BasicConditioningInfo,
    CogView4ConditioningInfo,
    ConditioningFieldData,
    FLUXConditioningInfo,
    SD3ConditioningInfo,
    SDXLConditioningInfo,
    ZImageConditioningInfo,
)
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.version.invokeai_version import __version__


# TODO: is there a better way to achieve this?
def check_internet() -> bool:
    """
    Return true if the internet is reachable.
    It does this by pinging huggingface.co.
    """
    import urllib.request

    host = "http://huggingface.co"
    try:
        urllib.request.urlopen(host, timeout=1)
        return True
    except Exception:
        return False


logger = InvokeAILogger.get_logger()


class ApiDependencies:
    """Contains and initializes all dependencies for the API"""

    invoker: Invoker

    @staticmethod
    def initialize(
        config: InvokeAIAppConfig,
        event_handler_id: int,
        loop: asyncio.AbstractEventLoop,
        logger: Logger = logger,
    ) -> None:
        logger.info(f"InvokeAI version {__version__}")
        logger.info(f"Root directory = {str(config.root_path)}")

        output_folder = config.outputs_path
        if output_folder is None:
            raise ValueError("Output folder is not set")

        image_files = DiskImageFileStorage(f"{output_folder}/images")

        model_images_folder = config.models_path
        style_presets_folder = config.style_presets_path
        workflow_thumbnails_folder = config.workflow_thumbnails_path

        db = init_db(config=config, logger=logger, image_files=image_files)

        configuration = config
        logger = logger

        board_image_records = SqliteBoardImageRecordStorage(db=db)
        board_images = BoardImagesService()
        board_records = SqliteBoardRecordStorage(db=db)
        boards = BoardService()
        events = FastAPIEventService(event_handler_id, loop=loop)
        bulk_download = BulkDownloadService()
        image_records = SqliteImageRecordStorage(db=db)
        images = ImageService()
        videos = VideoService()
        audios = AudioService()
        invocation_cache = MemoryInvocationCache(max_cache_size=config.node_cache_size)
        tensors = ObjectSerializerForwardCache(
            ObjectSerializerDisk[torch.Tensor](
                output_folder / "tensors",
                safe_globals=[torch.Tensor],
                ephemeral=True,
            ),
        )
        conditioning = ObjectSerializerForwardCache(
            ObjectSerializerDisk[ConditioningFieldData](
                output_folder / "conditioning",
                safe_globals=[
                    ConditioningFieldData,
                    BasicConditioningInfo,
                    SDXLConditioningInfo,
                    FLUXConditioningInfo,
                    SD3ConditioningInfo,
                    CogView4ConditioningInfo,
                    ZImageConditioningInfo,
                ],
                ephemeral=True,
            ),
        )
        download_queue_service = DownloadQueueService(app_config=configuration, event_bus=events)
        model_images_service = ModelImageFileStorageDisk(model_images_folder / "model_images")
        model_manager = ModelManagerService.build_model_manager(
            app_config=configuration,
            model_record_service=ModelRecordServiceSQL(db=db, logger=logger),
            download_queue=download_queue_service,
            events=events,
        )
        model_relationships = ModelRelationshipsService()
        model_relationship_records = SqliteModelRelationshipRecordStorage(db=db)
        names = SimpleNameService()
        performance_statistics = InvocationStatsService()
        session_processor = DefaultSessionProcessor(session_runner=DefaultSessionRunner())
        session_queue = SqliteSessionQueue(db=db)
        urls = LocalUrlService()
        workflow_records = SqliteWorkflowRecordsStorage(db=db)
        style_preset_records = SqliteStylePresetRecordsStorage(db=db)
        style_preset_image_files = StylePresetImageFileStorageDisk(style_presets_folder / "images")
        workflow_thumbnails = WorkflowThumbnailFileStorageDisk(workflow_thumbnails_folder)
        client_state_persistence = ClientStatePersistenceSqlite(db=db)

        services = InvocationServices(
            board_image_records=board_image_records,
            board_images=board_images,
            board_records=board_records,
            boards=boards,
            bulk_download=bulk_download,
            configuration=configuration,
            events=events,
            image_files=image_files,
            image_records=image_records,
            images=images,
            videos=videos,
            audios=audios,
            invocation_cache=invocation_cache,
            logger=logger,
            model_images=model_images_service,
            model_manager=model_manager,
            model_relationships=model_relationships,
            model_relationship_records=model_relationship_records,
            download_queue=download_queue_service,
            names=names,
            performance_statistics=performance_statistics,
            session_processor=session_processor,
            session_queue=session_queue,
            urls=urls,
            workflow_records=workflow_records,
            tensors=tensors,
            conditioning=conditioning,
            style_preset_records=style_preset_records,
            style_preset_image_files=style_preset_image_files,
            workflow_thumbnails=workflow_thumbnails,
            client_state_persistence=client_state_persistence,
        )

        ApiDependencies.invoker = Invoker(services)
        db.clean()

    @staticmethod
    def shutdown() -> None:
        if ApiDependencies.invoker:
            ApiDependencies.invoker.stop()
'''
    write_file("d:\\Cat_InvokeAI\\invokeai\\invokeai\\app\\api\\dependencies.py", content)

if __name__ == "__main__":
    print("Starting repair...")
    repair_invocation_context()
    repair_invocation_services()
    repair_dependencies()
    print("Repair complete. Please verify with check_update.py or start the app.")
