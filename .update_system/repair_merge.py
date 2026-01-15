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


def repair_primitives():
    import base64
    print("Repairing primitives.py...")
    content_b64 = "IyBDb3B5cmlnaHQgKGMpIDIwMjMgS3lsZSBTY2hvdXZpbGxlciAoaHR0cHM6Ly9naXRodWIuY29tL2t5bGUwNjU0KQoKZnJvbSB0eXBpbmcgaW1wb3J0IE9wdGlvbmFsCgppbXBvcnQgdG9yY2gKCmZyb20gaW52b2tlYWkuYXBwLmludm9jYXRpb25zLmJhc2VpbnZvY2F0aW9uIGltcG9ydCAoCiAgICBCYXNlSW52b2NhdGlvbiwKICAgIEJhc2VJbnZvY2F0aW9uT3V0cHV0LAogICAgaW52b2NhdGlvbiwKICAgIGludm9jYXRpb25fb3V0cHV0LAopCmZyb20gaW52b2tlYWkuYXBwLmludm9jYXRpb25zLmNvbnN0YW50cyBpbXBvcnQgTEFURU5UX1NDQUxFX0ZBQ1RPUgpmcm9tIGludm9rZWFpLmFwcC5pbnZvY2F0aW9ucy5maWVsZHMgaW1wb3J0ICgKICAgIEJvdW5kaW5nQm94RmllbGQsCiAgICBDb2dWaWV3NENvbmRpdGlvbmluZ0ZpZWxkLAogICAgQ29sb3JGaWVsZCwKICAgIENvbmRpdGlvbmluZ0ZpZWxkLAogICAgRGVub2lzZU1hc2tGaWVsZCwKICAgIEZpZWxkRGVzY3JpcHRpb25zLAogICAgRmx1eENvbmRpdGlvbmluZ0ZpZWxkLAogICAgSW1hZ2VGaWVsZCwKICAgIElucHV0LAogICAgSW5wdXRGaWVsZCwKICAgIExhdGVudHNGaWVsZCwKICAgIE91dHB1dEZpZWxkLAogICAgU0QzQ29uZGl0aW9uaW5nRmllbGQsCiAgICBUZW5zb3JGaWVsZCwKICAgIFVJQ29tcG9uZW50LAogICAgWkltYWdlQ29uZGl0aW9uaW5nRmllbGQsCiAgICBWaWRlb0ZpZWxkLAogICAgQXVkaW9GaWVsZCwKKQpmcm9tIGludm9rZWFpLmFwcC5zZXJ2aWNlcy5pbWFnZXMuaW1hZ2VzX2NvbW1vbiBpbXBvcnQgSW1hZ2VEVE8sIFZpZGVvRFRPLCBBdWRpb0RUTwpmcm9tIGludm9rZWFpLmFwcC5zZXJ2aWNlcy5zaGFyZWQuaW52b2NhdGlvbl9jb250ZXh0IGltcG9ydCBJbnZvY2F0aW9uQ29udGV4dAoKIiIiClByaW1pdGl2ZXM6IEJvb2xlYW4sIEludGVnZXIsIEZsb2F0LCBTdHJpbmcsIEltYWdlLCBMYXRlbnRzLCBDb25kaXRpb25pbmcsIENvbG9yCi0gcHJpbWl0aXZlIG5vZGVzCi0gcHJpbWl0aXZlIG91dHB1dHMKLSBwcmltaXRpdmUgY29sbGVjdGlvbiBvdXRwdXRzCiIiIgoKIyByZWdpb24gQm9vbGVhbgoKCkBpbnZvY2F0aW9uX291dHB1dCgiYm9vbGVhbl9vdXRwdXQiKQpjbGFzcyBCb29sZWFuT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIGJvb2xlYW4iIiIKCiAgICB2YWx1ZTogYm9vbCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgb3V0cHV0IGJvb2xlYW4iKQoKCkBpbnZvY2F0aW9uX291dHB1dCgiYm9vbGVhbl9jb2xsZWN0aW9uX291dHB1dCIpCmNsYXNzIEJvb2xlYW5Db2xsZWN0aW9uT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgY29sbGVjdGlvbiBvZiBib29sZWFucyIiIgoKICAgIGNvbGxlY3Rpb246IGxpc3RbYm9vbF0gPSBPdXRwdXRGaWVsZCgKICAgICAgICBkZXNjcmlwdGlvbj0iVGhlIG91dHB1dCBib29sZWFuIGNvbGxlY3Rpb24iLAogICAgKQoKCkBpbnZvY2F0aW9uKAogICAgImJvb2xlYW4iLCB0aXRsZT0iQm9vbGVhbiBQcmltaXRpdmUiLCB0YWdzPVsicHJpbWl0aXZlcyIsICJib29sZWFuIl0sIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwgdmVyc2lvbj0iMS4wLjEiCikKY2xhc3MgQm9vbGVhbkludm9jYXRpb24oQmFzZUludm9jYXRpb24pOgogICAgIiIiQSBib29sZWFuIHByaW1pdGl2ZSB2YWx1ZSIiIgoKICAgIHZhbHVlOiBib29sID0gSW5wdXRGaWVsZChkZWZhdWx0PUZhbHNlLCBkZXNjcmlwdGlvbj0iVGhlIGJvb2xlYW4gdmFsdWUiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IEJvb2xlYW5PdXRwdXQ6CiAgICAgICAgcmV0dXJuIEJvb2xlYW5PdXRwdXQodmFsdWU9c2VsZi52YWx1ZSkKCgpAaW52b2NhdGlvbigKICAgICJib29sZWFuX2NvbGxlY3Rpb24iLAogICAgdGl0bGU9IkJvb2xlYW4gQ29sbGVjdGlvbiBQcmltaXRpdmUiLAogICAgdGFncz1bInByaW1pdGl2ZXMiLCAiYm9vbGVhbiIsICJjb2xsZWN0aW9uIl0sCiAgICBjYXRlZ29yeT0icHJpbWl0aXZlcyIsCiAgICB2ZXJzaW9uPSIxLjAuMiIsCikKY2xhc3MgQm9vbGVhbkNvbGxlY3Rpb25JbnZvY2F0aW9uKEJhc2VJbnZvY2F0aW9uKToKICAgICIiIkEgY29sbGVjdGlvbiBvZiBib29sZWFuIHByaW1pdGl2ZSB2YWx1ZXMiIiIKCiAgICBjb2xsZWN0aW9uOiBsaXN0W2Jvb2xdID0gSW5wdXRGaWVsZChkZWZhdWx0PVtdLCBkZXNjcmlwdGlvbj0iVGhlIGNvbGxlY3Rpb24gb2YgYm9vbGVhbiB2YWx1ZXMiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IEJvb2xlYW5Db2xsZWN0aW9uT3V0cHV0OgogICAgICAgIHJldHVybiBCb29sZWFuQ29sbGVjdGlvbk91dHB1dChjb2xsZWN0aW9uPXNlbGYuY29sbGVjdGlvbikKCgojIGVuZHJlZ2lvbgoKIyByZWdpb24gSW50ZWdlcgoKCkBpbnZvY2F0aW9uX291dHB1dCgiaW50ZWdlcl9vdXRwdXQiKQpjbGFzcyBJbnRlZ2VyT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIGludGVnZXIiIiIKCiAgICB2YWx1ZTogaW50ID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBvdXRwdXQgaW50ZWdlciIpCgoKQGludm9jYXRpb25fb3V0cHV0KCJpbnRlZ2VyX2NvbGxlY3Rpb25fb3V0cHV0IikKY2xhc3MgSW50ZWdlckNvbGxlY3Rpb25PdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBjb2xsZWN0aW9uIG9mIGludGVnZXJzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtpbnRdID0gT3V0cHV0RmllbGQoCiAgICAgICAgZGVzY3JpcHRpb249IlRoZSBpbnQgY29sbGVjdGlvbiIsCiAgICApCgoKQGludm9jYXRpb24oCiAgICAiaW50ZWdlciIsIHRpdGxlPSJJbnRlZ2VyIFByaW1pdGl2ZSIsIHRhZ3M9WyJwcmltaXRpdmVzIiwgImludGVnZXIiXSwgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLCB2ZXJzaW9uPSIxLjAuMSIKKQpjbGFzcyBJbnRlZ2VySW52b2NhdGlvbihCYXNlSW52b2NhdGlvbik6CiAgICAiIiJBbiBpbnRlZ2VyIHByaW1pdGl2ZSB2YWx1ZSIiIgoKICAgIHZhbHVlOiBpbnQgPSBJbnB1dEZpZWxkKGRlZmF1bHQ9MCwgZGVzY3JpcHRpb249IlRoZSBpbnRlZ2VyIHZhbHVlIikKCiAgICBkZWYgaW52b2tlKHNlbGYsIGNvbnRleHQ6IEludm9jYXRpb25Db250ZXh0KSAtPiBJbnRlZ2VyT3V0cHV0OgogICAgICAgIHJldHVybiBJbnRlZ2VyT3V0cHV0KHZhbHVlPXNlbGYudmFsdWUpCgoKQGludm9jYXRpb24oCiAgICAiaW50ZWdlcl9jb2xsZWN0aW9uIiwKICAgIHRpdGxlPSJJbnRlZ2VyIENvbGxlY3Rpb24gUHJpbWl0aXZlIiwKICAgIHRhZ3M9WyJwcmltaXRpdmVzIiwgImludGVnZXIiLCAiY29sbGVjdGlvbiJdLAogICAgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLAogICAgdmVyc2lvbj0iMS4wLjIiLAopCmNsYXNzIEludGVnZXJDb2xsZWN0aW9uSW52b2NhdGlvbihCYXNlSW52b2NhdGlvbik6CiAgICAiIiJBIGNvbGxlY3Rpb24gb2YgaW50ZWdlciBwcmltaXRpdmUgdmFsdWVzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtpbnRdID0gSW5wdXRGaWVsZChkZWZhdWx0PVtdLCBkZXNjcmlwdGlvbj0iVGhlIGNvbGxlY3Rpb24gb2YgaW50ZWdlciB2YWx1ZXMiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IEludGVnZXJDb2xsZWN0aW9uT3V0cHV0OgogICAgICAgIHJldHVybiBJbnRlZ2VyQ29sbGVjdGlvbk91dHB1dChjb2xsZWN0aW9uPXNlbGYuY29sbGVjdGlvbikKCgojIGVuZHJlZ2lvbgoKIyByZWdpb24gRmxvYXQKCgpAaW52b2NhdGlvbl9vdXRwdXQoImZsb2F0X291dHB1dCIpCmNsYXNzIEZsb2F0T3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIGZsb2F0IiIiCgogICAgdmFsdWU6IGZsb2F0ID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBvdXRwdXQgZmxvYXQiKQoKCkBpbnZvY2F0aW9uX291dHB1dCgiZmxvYXRfY29sbGVjdGlvbl9vdXRwdXQiKQpjbGFzcyBGbG9hdENvbGxlY3Rpb25PdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBjb2xsZWN0aW9uIG9mIGZsb2F0cyIiIgoKICAgIGNvbGxlY3Rpb246IGxpc3RbZmxvYXRdID0gT3V0cHV0RmllbGQoCiAgICAgICAgZGVzY3JpcHRpb249IlRoZSBmbG9hdCBjb2xsZWN0aW9uIiwKICAgICkKCgpAaW52b2NhdGlvbigiZmxvYXQiLCB0aXRsZT0iRmxvYXQgUHJpbWl0aXZlIiwgdGFncz1bInByaW1pdGl2ZXMiLCAiZmxvYXQiXSwgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLCB2ZXJzaW9uPSIxLjAuMSIpCmNsYXNzIEZsb2F0SW52b2NhdGlvbihCYXNlSW52b2NhdGlvbik6CiAgICAiIiJBIGZsb2F0IHByaW1pdGl2ZSB2YWx1ZSIiIgoKICAgIHZhbHVlOiBmbG9hdCA9IElucHV0RmllbGQoZGVmYXVsdD0wLjAsIGRlc2NyaXB0aW9uPSJUaGUgZmxvYXQgdmFsdWUiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IEZsb2F0T3V0cHV0OgogICAgICAgIHJldHVybiBGbG9hdE91dHB1dCh2YWx1ZT1zZWxmLnZhbHVlKQoKCkBpbnZvY2F0aW9uKAogICAgImZsb2F0X2NvbGxlY3Rpb24iLAogICAgdGl0bGU9IkZsb2F0IENvbGxlY3Rpb24gUHJpbWl0aXZlIiwKICAgIHRhZ3M9WyJwcmltaXRpdmVzIiwgImZsb2F0IiwgImNvbGxlY3Rpb24iXSwKICAgIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwKICAgIHZlcnNpb249IjEuMC4yIiwKKQpjbGFzcyBGbG9hdENvbGxlY3Rpb25JbnZvY2F0aW9uKEJhc2VJbnZvY2F0aW9uKToKICAgICIiIkEgY29sbGVjdGlvbiBvZiBmbG9hdCBwcmltaXRpdmUgdmFsdWVzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtmbG9hdF0gPSBJbnB1dEZpZWxkKGRlZmF1bHQ9W10sIGRlc2NyaXB0aW9uPSJUaGUgY29sbGVjdGlvbiBvZiBmbG9hdCB2YWx1ZXMiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IEZsb2F0Q29sbGVjdGlvbk91dHB1dDoKICAgICAgICByZXR1cm4gRmxvYXRDb2xsZWN0aW9uT3V0cHV0KGNvbGxlY3Rpb249c2VsZi5jb2xsZWN0aW9uKQoKCiMgZW5kcmVnaW9uCgojIHJlZ2lvbiBTdHJpbmcKCgpAaW52b2NhdGlvbl9vdXRwdXQoInN0cmluZ19vdXRwdXQiKQpjbGFzcyBTdHJpbmdPdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBzaW5nbGUgc3RyaW5nIiIiCgogICAgdmFsdWU6IHN0ciA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgb3V0cHV0IHN0cmluZyIpCgoKQGludm9jYXRpb25fb3V0cHV0KCJzdHJpbmdfY29sbGVjdGlvbl9vdXRwdXQiKQpjbGFzcyBTdHJpbmdDb2xsZWN0aW9uT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgY29sbGVjdGlvbiBvZiBzdHJpbmdzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtzdHJdID0gT3V0cHV0RmllbGQoCiAgICAgICAgZGVzY3JpcHRpb249IlRoZSBvdXRwdXQgc3RyaW5ncyIsCiAgICApCgoKQGludm9jYXRpb24oInN0cmluZyIsIHRpdGxlPSJTdHJpbmcgUHJpbWl0aXZlIiwgdGFncz1bInByaW1pdGl2ZXMiLCAic3RyaW5nIl0sIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwgdmVyc2lvbj0iMS4wLjEiKQpjbGFzcyBTdHJpbmdJbnZvY2F0aW9uKEJhc2VJbnZvY2F0aW9uKToKICAgICIiIkEgc3RyaW5nIHByaW1pdGl2ZSB2YWx1ZSIiIgoKICAgIHZhbHVlOiBzdHIgPSBJbnB1dEZpZWxkKGRlZmF1bHQ9IiIsIGRlc2NyaXB0aW9uPSJUaGUgc3RyaW5nIHZhbHVlIiwgdWlfY29tcG9uZW50PVVJQ29tcG9uZW50LlRleHRhcmVhKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IFN0cmluZ091dHB1dDoKICAgICAgICByZXR1cm4gU3RyaW5nT3V0cHV0KHZhbHVlPXNlbGYudmFsdWUpCgoKQGludm9jYXRpb24oCiAgICAic3RyaW5nX2NvbGxlY3Rpb24iLAogICAgdGl0bGU9IlN0cmluZyBDb2xsZWN0aW9uIFByaW1pdGl2ZSIsCiAgICB0YWdzPVsicHJpbWl0aXZlcyIsICJzdHJpbmciLCAiY29sbGVjdGlvbiJdLAogICAgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLAogICAgdmVyc2lvbj0iMS4wLjIiLAopCmNsYXNzIFN0cmluZ0NvbGxlY3Rpb25JbnZvY2F0aW9uKEJhc2VJbnZvY2F0aW9uKToKICAgICIiIkEgY29sbGVjdGlvbiBvZiBzdHJpbmcgcHJpbWl0aXZlIHZhbHVlcyIiIgoKICAgIGNvbGxlY3Rpb246IGxpc3Rbc3RyXSA9IElucHV0RmllbGQoZGVmYXVsdD1bXSwgZGVzY3JpcHRpb249IlRoZSBjb2xsZWN0aW9uIG9mIHN0cmluZyB2YWx1ZXMiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IFN0cmluZ0NvbGxlY3Rpb25PdXRwdXQ6CiAgICAgICAgcmV0dXJuIFN0cmluZ0NvbGxlY3Rpb25PdXRwdXQoY29sbGVjdGlvbj1zZWxmLmNvbGxlY3Rpb24pCgoKIyBlbmRyZWdpb24KCiMgcmVnaW9uIEltYWdlCgoKQGludm9jYXRpb25fb3V0cHV0KCJpbWFnZV9vdXRwdXQiKQpjbGFzcyBJbWFnZU91dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJCYXNlIGNsYXNzIGZvciBub2RlcyB0aGF0IG91dHB1dCBhIHNpbmdsZSBpbWFnZSIiIgoKICAgIGltYWdlOiBJbWFnZUZpZWxkID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBvdXRwdXQgaW1hZ2UiKQogICAgd2lkdGg6IGludCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgd2lkdGggb2YgdGhlIGltYWdlIGluIHBpeGVscyIpCiAgICBoZWlnaHQ6IGludCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgaGVpZ2h0IG9mIHRoZSBpbWFnZSBpbiBwaXhlbHMiKQoKICAgIEBjbGFzc21ldGhvZAogICAgZGVmIGJ1aWxkKGNscywgaW1hZ2VfZHRvOiBJbWFnZURUTykgLT4gIkltYWdlT3V0cHV0IjoKICAgICAgICByZXR1cm4gY2xzKAogICAgICAgICAgICBpbWFnZT1JbWFnZUZpZWxkKGltYWdlX25hbWU9aW1hZ2VfZHRvLmltYWdlX25hbWUpLAogICAgICAgICAgICB3aWR0aD1pbWFnZV9kdG8ud2lkdGgsCiAgICAgICAgICAgIGhlaWdodD1pbWFnZV9kdG8uaGVpZ2h0LAogICAgICAgICkKCgpAaW52b2NhdGlvbl9vdXRwdXQoImltYWdlX2NvbGxlY3Rpb25fb3V0cHV0IikKY2xhc3MgSW1hZ2VDb2xsZWN0aW9uT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgY29sbGVjdGlvbiBvZiBpbWFnZXMiIiIKCiAgICBjb2xsZWN0aW9uOiBsaXN0W0ltYWdlRmllbGRdID0gT3V0cHV0RmllbGQoCiAgICAgICAgZGVzY3JpcHRpb249IlRoZSBvdXRwdXQgaW1hZ2VzIiwKICAgICkKCgpAaW52b2NhdGlvbigiaW1hZ2UiLCB0aXRsZT0iSW1hZ2UgUHJpbWl0aXZlIiwgdGFncz1bInByaW1pdGl2ZXMiLCAiaW1hZ2UiXSwgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLCB2ZXJzaW9uPSIxLjAuMiIpCmNsYXNzIEltYWdlSW52b2NhdGlvbihCYXNlSW52b2NhdGlvbik6CiAgICAiIiJBbiBpbWFnZSBwcmltaXRpdmUgdmFsdWUiIiIKCiAgICBpbWFnZTogSW1hZ2VGaWVsZCA9IElucHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBpbWFnZSB0byBsb2FkIikKCiAgICBkZWYgaW52b2tlKHNlbGYsIGNvbnRleHQ6IEludm9jYXRpb25Db250ZXh0KSAtPiBJbWFnZU91dHB1dDoKICAgICAgICBpbWFnZV9kdG8gPSBjb250ZXh0LmltYWdlcy5nZXRfZHRvKHNlbGYuaW1hZ2UuaW1hZ2VfbmFtZSkKCiAgICAgICAgcmV0dXJuIEltYWdlT3V0cHV0LmJ1aWxkKGltYWdlX2R0bz1pbWFnZV9kdG8pCgoKQGludm9jYXRpb24oCiAgICAiaW1hZ2VfY29sbGVjdGlvbiIsCiAgICB0aXRsZT0iSW1hZ2UgQ29sbGVjdGlvbiBQcmltaXRpdmUiLAogICAgdGFncz1bInByaW1pdGl2ZXMiLCAiaW1hZ2UiLCAiY29sbGVjdGlvbiJdLAogICAgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLAogICAgdmVyc2lvbj0iMS4wLjEiLAopCmNsYXNzIEltYWdlQ29sbGVjdGlvbkludm9jYXRpb24oQmFzZUludm9jYXRpb24pOgogICAgIiIiQSBjb2xsZWN0aW9uIG9mIGltYWdlIHByaW1pdGl2ZSB2YWx1ZXMiIiIKCiAgICBjb2xsZWN0aW9uOiBsaXN0W0ltYWdlRmllbGRdID0gSW5wdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIGNvbGxlY3Rpb24gb2YgaW1hZ2UgdmFsdWVzIikKCiAgICBkZWYgaW52b2tlKHNlbGYsIGNvbnRleHQ6IEludm9jYXRpb25Db250ZXh0KSAtPiBJbWFnZUNvbGxlY3Rpb25PdXRwdXQ6CiAgICAgICAgcmV0dXJuIEltYWdlQ29sbGVjdGlvbk91dHB1dChjb2xsZWN0aW9uPXNlbGYuY29sbGVjdGlvbikKCgojIGVuZHJlZ2lvbgoKIyByZWdpb24gVmlkZW8KCgpAaW52b2NhdGlvbl9vdXRwdXQoInZpZGVvX291dHB1dCIpCmNsYXNzIFZpZGVvT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIHZpZGVvIiIiCgogICAgdmlkZW86IFZpZGVvRmllbGQgPSBPdXRwdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIG91dHB1dCB2aWRlbyIpCiAgICB3aWR0aDogaW50ID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSB3aWR0aCBvZiB0aGUgdmlkZW8gaW4gcGl4ZWxzIikKICAgIGhlaWdodDogaW50ID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBoZWlnaHQgb2YgdGhlIHZpZGVvIGluIHBpeGVscyIpCiAgICBkdXJhdGlvbjogZmxvYXQgPSBPdXRwdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIGR1cmF0aW9uIG9mIHRoZSB2aWRlbyBpbiBzZWNvbmRzIikKICAgIGZwczogZmxvYXQgPSBPdXRwdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIGZyYW1lIHJhdGUgb2YgdGhlIHZpZGVvIikKCiAgICBAY2xhc3NtZXRob2QKICAgIGRlZiBidWlsZChjbHMsIHZpZGVvX2R0bzogVmlkZW9EVE8pIC0+ICJWaWRlb091dHB1dCI6CiAgICAgICAgcmV0dXJuIGNscygKICAgICAgICAgICAgdmlkZW89VmlkZW9GaWVsZCh2aWRlb19uYW1lPXZpZGVvX2R0by52aWRlb19uYW1lKSwKICAgICAgICAgICAgd2lkdGg9dmlkZW9fZHRvLndpZHRoLAogICAgICAgICAgICBoZWlnaHQ9dmlkZW9fZHRvLmhlaWdodCwKICAgICAgICAgICAgZHVyYXRpb249dmlkZW9fZHRvLmR1cmF0aW9uLAogICAgICAgICAgICBmcHM9dmlkZW9fZHRvLmZwcywKICAgICAgICApCgoKQGludm9jYXRpb25fb3V0cHV0KCJ2aWRlb19jb2xsZWN0aW9uX291dHB1dCIpCmNsYXNzIFZpZGVvQ29sbGVjdGlvbk91dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJCYXNlIGNsYXNzIGZvciBub2RlcyB0aGF0IG91dHB1dCBhIGNvbGxlY3Rpb24gb2YgdmlkZW9zIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtWaWRlb0ZpZWxkXSA9IE91dHB1dEZpZWxkKAogICAgICAgIGRlc2NyaXB0aW9uPSJUaGUgb3V0cHV0IHZpZGVvcyIsCiAgICApCgoKQGludm9jYXRpb24oInZpZGVvIiwgdGl0bGU9IlZpZGVvIFByaW1pdGl2ZSIsIHRhZ3M9WyJwcmltaXRpdmVzIiwgInZpZGVvIl0sIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwgdmVyc2lvbj0iMS4wLjAiKQpjbGFzcyBWaWRlb0ludm9jYXRpb24oQmFzZUludm9jYXRpb24pOgogICAgIiIiQSB2aWRlbyBwcmltaXRpdmUgdmFsdWUiIiIKCiAgICB2aWRlbzogVmlkZW9GaWVsZCA9IElucHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSB2aWRlbyB0byBsb2FkIikKCiAgICBkZWYgaW52b2tlKHNlbGYsIGNvbnRleHQ6IEludm9jYXRpb25Db250ZXh0KSAtPiBWaWRlb091dHB1dDoKICAgICAgICAjIFBsYWNlaG9sZGVyIGltcGxlbWVudGF0aW9uCiAgICAgICAgIyB2aWRlb19kdG8gPSBjb250ZXh0LnZpZGVvcy5nZXRfZHRvKHNlbGYudmlkZW8udmlkZW9fbmFtZSkKICAgICAgICAjIHJldHVybiBWaWRlb091dHB1dC5idWlsZCh2aWRlb19kdG89dmlkZW9fZHRvKQogICAgICAgIHBhc3MKCgpAaW52b2NhdGlvbigKICAgICJ2aWRlb19jb2xsZWN0aW9uIiwKICAgIHRpdGxlPSJWaWRlbyBDb2xsZWN0aW9uIFByaW1pdGl2ZSIsCiAgICB0YWdzPVsicHJpbWl0aXZlcyIsICJ2aWRlbyIsICJjb2xsZWN0aW9uIl0sCiAgICBjYXRlZ29yeT0icHJpbWl0aXZlcyIsCiAgICB2ZXJzaW9uPSIxLjAuMCIsCikKY2xhc3MgVmlkZW9Db2xsZWN0aW9uSW52b2NhdGlvbihCYXNlSW52b2NhdGlvbik6CiAgICAiIiJBIGNvbGxlY3Rpb24gb2YgdmlkZW8gcHJpbWl0aXZlIHZhbHVlcyIiIgoKICAgIGNvbGxlY3Rpb246IGxpc3RbVmlkZW9GaWVsZF0gPSBJbnB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgY29sbGVjdGlvbiBvZiB2aWRlbyB2YWx1ZXMiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IFZpZGVvQ29sbGVjdGlvbk91dHB1dDoKICAgICAgICByZXR1cm4gVmlkZW9Db2xsZWN0aW9uT3V0cHV0KGNvbGxlY3Rpb249c2VsZi5jb2xsZWN0aW9uKQoKCiMgZW5kcmVnaW9uCgojIHJlZ2lvbiBBdWRpbwoKCkBpbnZvY2F0aW9uX291dHB1dCgiYXVkaW9fb3V0cHV0IikKY2xhc3MgQXVkaW9PdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBzaW5nbGUgYXVkaW8iIiIKCiAgICBhdWRpbzogQXVkaW9GaWVsZCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgb3V0cHV0IGF1ZGlvIikKICAgIGR1cmF0aW9uOiBmbG9hdCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgZHVyYXRpb24gb2YgdGhlIGF1ZGlvIGluIHNlY29uZHMiKQoKICAgIEBjbGFzc21ldGhvZAogICAgZGVmIGJ1aWxkKGNscywgYXVkaW9fZHRvOiBBdWRpb0RUTykgLT4gIkF1ZGlvT3V0cHV0IjoKICAgICAgICByZXR1cm4gY2xzKAogICAgICAgICAgICBhdWRpbz1BdWRpb0ZpZWxkKGF1ZGlvX25hbWU9YXVkaW9fZHRvLmF1ZGlvX25hbWUpLAogICAgICAgICAgICBkdXJhdGlvbj1hdWRpb19kdG8uZHVyYXRpb24sCiAgICAgICAgKQoKCkBpbnZvY2F0aW9uX291dHB1dCgiYXVkaW9fY29sbGVjdGlvbl9vdXRwdXQiKQpjbGFzcyBBdWRpb0NvbGxlY3Rpb25PdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBjb2xsZWN0aW9uIG9mIGF1ZGlvcyIiIgoKICAgIGNvbGxlY3Rpb246IGxpc3RbQXVkaW9GaWVsZF0gPSBPdXRwdXRGaWVsZCgKICAgICAgICBkZXNjcmlwdGlvbj0iVGhlIG91dHB1dCBhdWRpb3MiLAogICAgKQoKCkBpbnZvY2F0aW9uKCJhdWRpbyIsIHRpdGxlPSJBdWRpbyBQcmltaXRpdmUiLCB0YWdzPVsicHJpbWl0aXZlcyIsICJhdWRpbyJdLCBjYXRlZ29yeT0icHJpbWl0aXZlcyIsIHZlcnNpb249IjEuMC4wIikKY2xhc3MgQXVkaW9JbnZvY2F0aW9uKEJhc2VJbnZvY2F0aW9uKToKICAgICIiIkFuIGF1ZGlvIHByaW1pdGl2ZSB2YWx1ZSIiIgoKICAgIGF1ZGlvOiBBdWRpb0ZpZWxkID0gSW5wdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIGF1ZGlvIHRvIGxvYWQiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IEF1ZGlvT3V0cHV0OgogICAgICAgICMgUGxhY2Vob2xkZXIgaW1wbGVtZW50YXRpb24KICAgICAgICAjIGF1ZGlvX2R0byA9IGNvbnRleHQuYXVkaW9zLmdldF9kdG8oc2VsZi5hdWRpby5hdWRpb19uYW1lKQogICAgICAgICMgcmV0dXJuIEF1ZGlvT3V0cHV0LmJ1aWxkKGF1ZGlvX2R0bz1hdWRpb19kdG8pCiAgICAgICAgcGFzcwoKCkBpbnZvY2F0aW9uKAogICAgImF1ZGlvX2NvbGxlY3Rpb24iLAogICAgdGl0bGU9IkF1ZGlvIENvbGxlY3Rpb24gUHJpbWl0aXZlIiwKICAgIHRhZ3M9WyJwcmltaXRpdmVzIiwgImF1ZGlvIiwgImNvbGxlY3Rpb24iXSwKICAgIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwKICAgIHZlcnNpb249IjEuMC4wIiwKKQpjbGFzcyBBdWRpb0NvbGxlY3Rpb25JbnZvY2F0aW9uKEJhc2VJbnZvY2F0aW9uKToKICAgICIiIkEgY29sbGVjdGlvbiBvZiBhdWRpbyBwcmltaXRpdmUgdmFsdWVzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtBdWRpb0ZpZWxkXSA9IElucHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBjb2xsZWN0aW9uIG9mIGF1ZGlvIHZhbHVlcyIpCgogICAgZGVmIGludm9rZShzZWxmLCBjb250ZXh0OiBJbnZvY2F0aW9uQ29udGV4dCkgLT4gQXVkaW9Db2xsZWN0aW9uT3V0cHV0OgogICAgICAgIHJldHVybiBBdWRpb0NvbGxlY3Rpb25PdXRwdXQoY29sbGVjdGlvbj1zZWxmLmNvbGxlY3Rpb24pCgoKIyBlbmRyZWdpb24KIyByZWdpb24gRGVub2lzZU1hc2sKCgpAaW52b2NhdGlvbl9vdXRwdXQoImRlbm9pc2VfbWFza19vdXRwdXQiKQpjbGFzcyBEZW5vaXNlTWFza091dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJCYXNlIGNsYXNzIGZvciBub2RlcyB0aGF0IG91dHB1dCBhIHNpbmdsZSBpbWFnZSIiIgoKICAgIGRlbm9pc2VfbWFzazogRGVub2lzZU1hc2tGaWVsZCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJNYXNrIGZvciBkZW5vaXNlIG1vZGVsIHJ1biIpCgogICAgQGNsYXNzbWV0aG9kCiAgICBkZWYgYnVpbGQoCiAgICAgICAgY2xzLCBtYXNrX25hbWU6IHN0ciwgbWFza2VkX2xhdGVudHNfbmFtZTogT3B0aW9uYWxbc3RyXSA9IE5vbmUsIGdyYWRpZW50OiBib29sID0gRmFsc2UKICAgICkgLT4gIkRlbm9pc2VNYXNrT3V0cHV0IjoKICAgICAgICByZXR1cm4gY2xzKAogICAgICAgICAgICBkZW5vaXNlX21hc2s9RGVub2lzZU1hc2tGaWVsZCgKICAgICAgICAgICAgICAgIG1hc2tfbmFtZT1tYXNrX25hbWUsIG1hc2tlZF9sYXRlbnRzX25hbWU9bWFza2VkX2xhdGVudHNfbmFtZSwgZ3JhZGllbnQ9Z3JhZGllbnQKICAgICAgICAgICAgKSwKICAgICAgICApCgoKIyBlbmRyZWdpb24KCiMgcmVnaW9uIExhdGVudHMKCgpAaW52b2NhdGlvbl9vdXRwdXQoImxhdGVudHNfb3V0cHV0IikKY2xhc3MgTGF0ZW50c091dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJCYXNlIGNsYXNzIGZvciBub2RlcyB0aGF0IG91dHB1dCBhIHNpbmdsZSBsYXRlbnRzIHRlbnNvciIiIgoKICAgIGxhdGVudHM6IExhdGVudHNGaWVsZCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPUZpZWxkRGVzY3JpcHRpb25zLmxhdGVudHMpCiAgICB3aWR0aDogaW50ID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249RmllbGREZXNjcmlwdGlvbnMud2lkdGgpCiAgICBoZWlnaHQ6IGludCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPUZpZWxkRGVzY3JpcHRpb25zLmhlaWdodCkKCiAgICBAY2xhc3NtZXRob2QKICAgIGRlZiBidWlsZChjbHMsIGxhdGVudHNfbmFtZTogc3RyLCBsYXRlbnRzOiB0b3JjaC5UZW5zb3IsIHNlZWQ6IE9wdGlvbmFsW2ludF0gPSBOb25lKSAtPiAiTGF0ZW50c091dHB1dCI6CiAgICAgICAgcmV0dXJuIGNscygKICAgICAgICAgICAgbGF0ZW50cz1MYXRlbnRzRmllbGQobGF0ZW50c19uYW1lPWxhdGVudHNfbmFtZSwgc2VlZD1zZWVkKSwKICAgICAgICAgICAgd2lkdGg9bGF0ZW50cy5zaXplKClbM10gKiBMQVRFTlRfU0NBTEVfRkFDVE9SLAogICAgICAgICAgICBoZWlnaHQ9bGF0ZW50cy5zaXplKClbMl0gKiBMQVRFTlRfU0NBTEVfRkFDVE9SLAogICAgICAgICkKCgpAaW52b2NhdGlvbl9vdXRwdXQoImxhdGVudHNfY29sbGVjdGlvbl9vdXRwdXQiKQpjbGFzcyBMYXRlbnRzQ29sbGVjdGlvbk91dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJCYXNlIGNsYXNzIGZvciBub2RlcyB0aGF0IG91dHB1dCBhIGNvbGxlY3Rpb24gb2YgbGF0ZW50cyB0ZW5zb3JzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtMYXRlbnRzRmllbGRdID0gT3V0cHV0RmllbGQoCiAgICAgICAgZGVzY3JpcHRpb249RmllbGREZXNjcmlwdGlvbnMubGF0ZW50cywKICAgICkKCgpAaW52b2NhdGlvbigKICAgICJsYXRlbnRzIiwgdGl0bGU9IkxhdGVudHMgUHJpbWl0aXZlIiwgdGFncz1bInByaW1pdGl2ZXMiLCAibGF0ZW50cyJdLCBjYXRlZ29yeT0icHJpbWl0aXZlcyIsIHZlcnNpb249IjEuMC4yIgopCmNsYXNzIExhdGVudHNJbnZvY2F0aW9uKEJhc2VJbnZvY2F0aW9uKToKICAgICIiIkEgbGF0ZW50cyB0ZW5zb3IgcHJpbWl0aXZlIHZhbHVlIiIiCgogICAgbGF0ZW50czogTGF0ZW50c0ZpZWxkID0gSW5wdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIGxhdGVudHMgdGVuc29yIiwgaW5wdXQ9SW5wdXQuQ29ubmVjdGlvbikKCiAgICBkZWYgaW52b2tlKHNlbGYsIGNvbnRleHQ6IEludm9jYXRpb25Db250ZXh0KSAtPiBMYXRlbnRzT3V0cHV0OgogICAgICAgIGxhdGVudHMgPSBjb250ZXh0LnRlbnNvcnMubG9hZChzZWxmLmxhdGVudHMubGF0ZW50c19uYW1lKQoKICAgICAgICByZXR1cm4gTGF0ZW50c091dHB1dC5idWlsZChzZWxmLmxhdGVudHMubGF0ZW50c19uYW1lLCBsYXRlbnRzKQoKCkBpbnZvY2F0aW9uKAogICAgImxhdGVudHNfY29sbGVjdGlvbiIsCiAgICB0aXRsZT0iTGF0ZW50cyBDb2xsZWN0aW9uIFByaW1pdGl2ZSIsCiAgICB0YWdzPVsicHJpbWl0aXZlcyIsICJsYXRlbnRzIiwgImNvbGxlY3Rpb24iXSwKICAgIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwKICAgIHZlcnNpb249IjEuMC4xIiwKKQpjbGFzcyBMYXRlbnRzQ29sbGVjdGlvbkludm9jYXRpb24oQmFzZUludm9jYXRpb24pOgogICAgIiIiQSBjb2xsZWN0aW9uIG9mIGxhdGVudHMgdGVuc29yIHByaW1pdGl2ZSB2YWx1ZXMiIiIKCiAgICBjb2xsZWN0aW9uOiBsaXN0W0xhdGVudHNGaWVsZF0gPSBJbnB1dEZpZWxkKAogICAgICAgIGRlc2NyaXB0aW9uPSJUaGUgY29sbGVjdGlvbiBvZiBsYXRlbnRzIHRlbnNvcnMiLAogICAgKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IExhdGVudHNDb2xsZWN0aW9uT3V0cHV0OgogICAgICAgIHJldHVybiBMYXRlbnRzQ29sbGVjdGlvbk91dHB1dChjb2xsZWN0aW9uPXNlbGYuY29sbGVjdGlvbikKCgojIGVuZHJlZ2lvbgoKIyByZWdpb24gQ29sb3IKCgpAaW52b2NhdGlvbl9vdXRwdXQoImNvbG9yX291dHB1dCIpCmNsYXNzIENvbG9yT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIGNvbG9yIiIiCgogICAgY29sb3I6IENvbG9yRmllbGQgPSBPdXRwdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIG91dHB1dCBjb2xvciIpCgoKQGludm9jYXRpb25fb3V0cHV0KCJjb2xvcl9jb2xsZWN0aW9uX291dHB1dCIpCmNsYXNzIENvbG9yQ29sbGVjdGlvbk91dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJCYXNlIGNsYXNzIGZvciBub2RlcyB0aGF0IG91dHB1dCBhIGNvbGxlY3Rpb24gb2YgY29sb3JzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtDb2xvckZpZWxkXSA9IE91dHB1dEZpZWxkKAogICAgICAgIGRlc2NyaXB0aW9uPSJUaGUgb3V0cHV0IGNvbG9ycyIsCiAgICApCgoKQGludm9jYXRpb24oImNvbG9yIiwgdGl0bGU9IkNvbG9yIFByaW1pdGl2ZSIsIHRhZ3M9WyJwcmltaXRpdmVzIiwgImNvbG9yIl0sIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwgdmVyc2lvbj0iMS4wLjEiKQpjbGFzcyBDb2xvckludm9jYXRpb24oQmFzZUludm9jYXRpb24pOgogICAgIiIiQSBjb2xvciBwcmltaXRpdmUgdmFsdWUiIiIKCiAgICBjb2xvcjogQ29sb3JGaWVsZCA9IElucHV0RmllbGQoZGVmYXVsdD1Db2xvckZpZWxkKHI9MCwgZz0wLCBiPTAsIGE9MjU1KSwgZGVzY3JpcHRpb249IlRoZSBjb2xvciB2YWx1ZSIpCgogICAgZGVmIGludm9rZShzZWxmLCBjb250ZXh0OiBJbnZvY2F0aW9uQ29udGV4dCkgLT4gQ29sb3JPdXRwdXQ6CiAgICAgICAgcmV0dXJuIENvbG9yT3V0cHV0KGNvbG9yPXNlbGYuY29sb3IpCgoKIyBlbmRyZWdpb24KCgojIHJlZ2lvbiBDb25kaXRpb25pbmcKCgpAaW52b2NhdGlvbl9vdXRwdXQoIm1hc2tfb3V0cHV0IikKY2xhc3MgTWFza091dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJBIHRvcmNoIG1hc2sgdGVuc29yLiIiIgoKICAgICMgc2hhcGU6IFsxLCBILCBXXSwgZHR5cGU6IGJvb2wKICAgIG1hc2s6IFRlbnNvckZpZWxkID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBtYXNrLiIpCiAgICB3aWR0aDogaW50ID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSB3aWR0aCBvZiB0aGUgbWFzayBpbiBwaXhlbHMuIikKICAgIGhlaWdodDogaW50ID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249IlRoZSBoZWlnaHQgb2YgdGhlIG1hc2sgaW4gcGl4ZWxzLiIpCgoKQGludm9jYXRpb25fb3V0cHV0KCJmbHV4X2NvbmRpdGlvbmluZ19vdXRwdXQiKQpjbGFzcyBGbHV4Q29uZGl0aW9uaW5nT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIGNvbmRpdGlvbmluZyB0ZW5zb3IiIiIKCiAgICBjb25kaXRpb25pbmc6IEZsdXhDb25kaXRpb25pbmdGaWVsZCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPUZpZWxkRGVzY3JpcHRpb25zLmNvbmQpCgogICAgQGNsYXNzbWV0aG9kCiAgICBkZWYgYnVpbGQoY2xzLCBjb25kaXRpb25pbmdfbmFtZTogc3RyKSAtPiAiRmx1eENvbmRpdGlvbmluZ091dHB1dCI6CiAgICAgICAgcmV0dXJuIGNscyhjb25kaXRpb25pbmc9Rmx1eENvbmRpdGlvbmluZ0ZpZWxkKGNvbmRpdGlvbmluZ19uYW1lPWNvbmRpdGlvbmluZ19uYW1lKSkKCgpAaW52b2NhdGlvbl9vdXRwdXQoImZsdXhfY29uZGl0aW9uaW5nX2NvbGxlY3Rpb25fb3V0cHV0IikKY2xhc3MgRmx1eENvbmRpdGlvbmluZ0NvbGxlY3Rpb25PdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBjb2xsZWN0aW9uIG9mIGNvbmRpdGlvbmluZyB0ZW5zb3JzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtGbHV4Q29uZGl0aW9uaW5nRmllbGRdID0gT3V0cHV0RmllbGQoCiAgICAgICAgZGVzY3JpcHRpb249IlRoZSBvdXRwdXQgY29uZGl0aW9uaW5nIHRlbnNvcnMiLAogICAgKQoKCkBpbnZvY2F0aW9uX291dHB1dCgic2QzX2NvbmRpdGlvbmluZ19vdXRwdXQiKQpjbGFzcyBTRDNDb25kaXRpb25pbmdPdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBzaW5nbGUgU0QzIGNvbmRpdGlvbmluZyB0ZW5zb3IiIiIKCiAgICBjb25kaXRpb25pbmc6IFNEM0NvbmRpdGlvbmluZ0ZpZWxkID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249RmllbGREZXNjcmlwdGlvbnMuY29uZCkKCiAgICBAY2xhc3NtZXRob2QKICAgIGRlZiBidWlsZChjbHMsIGNvbmRpdGlvbmluZ19uYW1lOiBzdHIpIC0+ICJTRDNDb25kaXRpb25pbmdPdXRwdXQiOgogICAgICAgIHJldHVybiBjbHMoY29uZGl0aW9uaW5nPVNEM0NvbmRpdGlvbmluZ0ZpZWxkKGNvbmRpdGlvbmluZ19uYW1lPWNvbmRpdGlvbmluZ19uYW1lKSkKCgpAaW52b2NhdGlvbl9vdXRwdXQoImNvZ3ZpZXc0X2NvbmRpdGlvbmluZ19vdXRwdXQiKQpjbGFzcyBDb2dWaWV3NENvbmRpdGlvbmluZ091dHB1dChCYXNlSW52b2NhdGlvbk91dHB1dCk6CiAgICAiIiJCYXNlIGNsYXNzIGZvciBub2RlcyB0aGF0IG91dHB1dCBhIENvZ1ZpZXcgdGV4dCBjb25kaXRpb25pbmcgdGVuc29yLiIiIgoKICAgIGNvbmRpdGlvbmluZzogQ29nVmlldzRDb25kaXRpb25pbmdGaWVsZCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPUZpZWxkRGVzY3JpcHRpb25zLmNvbmQpCgogICAgQGNsYXNzbWV0aG9kCiAgICBkZWYgYnVpbGQoY2xzLCBjb25kaXRpb25pbmdfbmFtZTogc3RyKSAtPiAiQ29nVmlldzRDb25kaXRpb25pbmdPdXRwdXQiOgogICAgICAgIHJldHVybiBjbHMoY29uZGl0aW9uaW5nPUNvZ1ZpZXc0Q29uZGl0aW9uaW5nRmllbGQoY29uZGl0aW9uaW5nX25hbWU9Y29uZGl0aW9uaW5nX25hbWUpKQoKCkBpbnZvY2F0aW9uX291dHB1dCgiel9pbWFnZV9jb25kaXRpb25pbmdfb3V0cHV0IikKY2xhc3MgWkltYWdlQ29uZGl0aW9uaW5nT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgWi1JbWFnZSB0ZXh0IGNvbmRpdGlvbmluZyB0ZW5zb3IuIiIiCgogICAgY29uZGl0aW9uaW5nOiBaSW1hZ2VDb25kaXRpb25pbmdGaWVsZCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPUZpZWxkRGVzY3JpcHRpb25zLmNvbmQpCgogICAgQGNsYXNzbWV0aG9kCiAgICBkZWYgYnVpbGQoY2xzLCBjb25kaXRpb25pbmdfbmFtZTogc3RyKSAtPiAiWkltYWdlQ29uZGl0aW9uaW5nT3V0cHV0IjoKICAgICAgICByZXR1cm4gY2xzKGNvbmRpdGlvbmluZz1aSW1hZ2VDb25kaXRpb25pbmdGaWVsZChjb25kaXRpb25pbmdfbmFtZT1jb25kaXRpb25pbmdfbmFtZSkpCgoKQGludm9jYXRpb25fb3V0cHV0KCJjb25kaXRpb25pbmdfb3V0cHV0IikKY2xhc3MgQ29uZGl0aW9uaW5nT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIGNvbmRpdGlvbmluZyB0ZW5zb3IiIiIKCiAgICBjb25kaXRpb25pbmc6IENvbmRpdGlvbmluZ0ZpZWxkID0gT3V0cHV0RmllbGQoZGVzY3JpcHRpb249RmllbGREZXNjcmlwdGlvbnMuY29uZCkKCiAgICBAY2xhc3NtZXRob2QKICAgIGRlZiBidWlsZChjbHMsIGNvbmRpdGlvbmluZ19uYW1lOiBzdHIpIC0+ICJDb25kaXRpb25pbmdPdXRwdXQiOgogICAgICAgIHJldHVybiBjbHMoY29uZGl0aW9uaW5nPUNvbmRpdGlvbmluZ0ZpZWxkKGNvbmRpdGlvbmluZ19uYW1lPWNvbmRpdGlvbmluZ19uYW1lKSkKCgpAaW52b2NhdGlvbl9vdXRwdXQoImNvbmRpdGlvbmluZ19jb2xsZWN0aW9uX291dHB1dCIpCmNsYXNzIENvbmRpdGlvbmluZ0NvbGxlY3Rpb25PdXRwdXQoQmFzZUludm9jYXRpb25PdXRwdXQpOgogICAgIiIiQmFzZSBjbGFzcyBmb3Igbm9kZXMgdGhhdCBvdXRwdXQgYSBjb2xsZWN0aW9uIG9mIGNvbmRpdGlvbmluZyB0ZW5zb3JzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtDb25kaXRpb25pbmdGaWVsZF0gPSBPdXRwdXRGaWVsZCgKICAgICAgICBkZXNjcmlwdGlvbj0iVGhlIG91dHB1dCBjb25kaXRpb25pbmcgdGVuc29ycyIsCiAgICApCgoKQGludm9jYXRpb24oCiAgICAiY29uZGl0aW9uaW5nIiwKICAgIHRpdGxlPSJDb25kaXRpb25pbmcgUHJpbWl0aXZlIiwKICAgIHRhZ3M9WyJwcmltaXRpdmVzIiwgImNvbmRpdGlvbmluZyJdLAogICAgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLAogICAgdmVyc2lvbj0iMS4wLjEiLAopCmNsYXNzIENvbmRpdGlvbmluZ0ludm9jYXRpb24oQmFzZUludm9jYXRpb24pOgogICAgIiIiQSBjb25kaXRpb25pbmcgdGVuc29yIHByaW1pdGl2ZSB2YWx1ZSIiIgoKICAgIGNvbmRpdGlvbmluZzogQ29uZGl0aW9uaW5nRmllbGQgPSBJbnB1dEZpZWxkKGRlc2NyaXB0aW9uPUZpZWxkRGVzY3JpcHRpb25zLmNvbmQsIGlucHV0PUlucHV0LkNvbm5lY3Rpb24pCgogICAgZGVmIGludm9rZShzZWxmLCBjb250ZXh0OiBJbnZvY2F0aW9uQ29udGV4dCkgLT4gQ29uZGl0aW9uaW5nT3V0cHV0OgogICAgICAgIHJldHVybiBDb25kaXRpb25pbmdPdXRwdXQoY29uZGl0aW9uaW5nPXNlbGYuY29uZGl0aW9uaW5nKQoKCkBpbnZvY2F0aW9uKAogICAgImNvbmRpdGlvbmluZ19jb2xsZWN0aW9uIiwKICAgIHRpdGxlPSJDb25kaXRpb25pbmcgQ29sbGVjdGlvbiBQcmltaXRpdmUiLAogICAgdGFncz1bInByaW1pdGl2ZXMiLCAiY29uZGl0aW9uaW5nIiwgImNvbGxlY3Rpb24iXSwKICAgIGNhdGVnb3J5PSJwcmltaXRpdmVzIiwKICAgIHZlcnNpb249IjEuMC4yIiwKKQpjbGFzcyBDb25kaXRpb25pbmdDb2xsZWN0aW9uSW52b2NhdGlvbihCYXNlSW52b2NhdGlvbik6CiAgICAiIiJBIGNvbGxlY3Rpb24gb2YgY29uZGl0aW9uaW5nIHRlbnNvciBwcmltaXRpdmUgdmFsdWVzIiIiCgogICAgY29sbGVjdGlvbjogbGlzdFtDb25kaXRpb25pbmdGaWVsZF0gPSBJbnB1dEZpZWxkKAogICAgICAgIGRlZmF1bHQ9W10sCiAgICAgICAgZGVzY3JpcHRpb249IlRoZSBjb2xsZWN0aW9uIG9mIGNvbmRpdGlvbmluZyB0ZW5zb3JzIiwKICAgICkKCiAgICBkZWYgaW52b2tlKHNlbGYsIGNvbnRleHQ6IEludm9jYXRpb25Db250ZXh0KSAtPiBDb25kaXRpb25pbmdDb2xsZWN0aW9uT3V0cHV0OgogICAgICAgIHJldHVybiBDb25kaXRpb25pbmdDb2xsZWN0aW9uT3V0cHV0KGNvbGxlY3Rpb249c2VsZi5jb2xsZWN0aW9uKQoKCiMgZW5kcmVnaW9uCgojIHJlZ2lvbiBCb3VuZGluZ0JveAoKCkBpbnZvY2F0aW9uX291dHB1dCgiYm91bmRpbmdfYm94X291dHB1dCIpCmNsYXNzIEJvdW5kaW5nQm94T3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgc2luZ2xlIGJvdW5kaW5nIGJveCIiIgoKICAgIGJvdW5kaW5nX2JveDogQm91bmRpbmdCb3hGaWVsZCA9IE91dHB1dEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgb3V0cHV0IGJvdW5kaW5nIGJveC4iKQoKCkBpbnZvY2F0aW9uX291dHB1dCgiYm91bmRpbmdfYm94X2NvbGxlY3Rpb25fb3V0cHV0IikKY2xhc3MgQm91bmRpbmdCb3hDb2xsZWN0aW9uT3V0cHV0KEJhc2VJbnZvY2F0aW9uT3V0cHV0KToKICAgICIiIkJhc2UgY2xhc3MgZm9yIG5vZGVzIHRoYXQgb3V0cHV0IGEgY29sbGVjdGlvbiBvZiBib3VuZGluZyBib3hlcyIiIgoKICAgIGNvbGxlY3Rpb246IGxpc3RbQm91bmRpbmdCb3hGaWVsZF0gPSBPdXRwdXRGaWVsZChkZXNjcmlwdGlvbj0iVGhlIG91dHB1dCBib3VuZGluZyBib3hlcy4iLCB0aXRsZT0iQm91bmRpbmcgQm94ZXMiKQoKCkBpbnZvY2F0aW9uKAogICAgImJvdW5kaW5nX2JveCIsCiAgICB0aXRsZT0iQm91bmRpbmcgQm94IiwKICAgIHRhZ3M9WyJwcmltaXRpdmVzIiwgInNlZ21lbnRhdGlvbiIsICJjb2xsZWN0aW9uIiwgImJvdW5kaW5nIGJveCJdLAogICAgY2F0ZWdvcnk9InByaW1pdGl2ZXMiLAogICAgdmVyc2lvbj0iMS4wLjAiLAopCmNsYXNzIEJvdW5kaW5nQm94SW52b2NhdGlvbihCYXNlSW52b2NhdGlvbik6CiAgICAiIiJDcmVhdGUgYSBib3VuZGluZyBib3ggbWFudWFsbHkgYnkgc3VwcGx5aW5nIGJveCBjb29yZGluYXRlcyIiIgoKICAgIHhfbWluOiBpbnQgPSBJbnB1dEZpZWxkKGRlZmF1bHQ9MCwgZGVzY3JpcHRpb249IngtY29vcmRpbmF0ZSBvZiB0aGUgYm91bmRpbmcgYm94J3MgdG9wIGxlZnQgdmVydGV4IikKICAgIHlfbWluOiBpbnQgPSBJbnB1dEZpZWxkKGRlZmF1bHQ9MCwgZGVzY3JpcHRpb249InktY29vcmRpbmF0ZSBvZiB0aGUgYm91bmRpbmcgYm94J3MgdG9wIGxlZnQgdmVydGV4IikKICAgIHhfbWF4OiBpbnQgPSBJbnB1dEZpZWxkKGRlZmF1bHQ9MCwgZGVzY3JpcHRpb249IngtY29vcmRpbmF0ZSBvZiB0aGUgYm91bmRpbmcgYm94J3MgYm90dG9tIHJpZ2h0IHZlcnRleCIpCiAgICB5X21heDogaW50ID0gSW5wdXRGaWVsZChkZWZhdWx0PTAsIGRlc2NyaXB0aW9uPSJ5LWNvb3JkaW5hdGUgb2YgdGhlIGJvdW5kaW5nIGJveCdzIGJvdHRvbSByaWdodCB2ZXJ0ZXgiKQoKICAgIGRlZiBpbnZva2Uoc2VsZiwgY29udGV4dDogSW52b2NhdGlvbkNvbnRleHQpIC0+IEJvdW5kaW5nQm94T3V0cHV0OgogICAgICAgIGJvdW5kaW5nX2JveCA9IEJvdW5kaW5nQm94RmllbGQoeF9taW49c2VsZi54X21pbiwgeV9taW49c2VsZi55X21pbiwgeF9tYXg9c2VsZi54X21heCwgeV9tYXg9c2VsZi55X21heCkKICAgICAgICByZXR1cm4gQm91bmRpbmdCb3hPdXRwdXQoYm91bmRpbmdfYm94PWJvdW5kaW5nX2JveCkKCgojIGVuZHJlZ2lvbgo="
    content = base64.b64decode(content_b64).decode('utf-8')
    write_file(r"d:\Cat_InvokeAI\invokeai\invokeai\app\invocations\primitives.py", content)


def repair_fields():
    import base64
    print("Repairing fields.py...")
    content_b64 = "ZnJvbSBlbnVtIGltcG9ydCBFbnVtCmZyb20gdHlwaW5nIGltcG9ydCBBbnksIENhbGxhYmxlLCBPcHRpb25hbCwgVHVwbGUKCmZyb20gcHlkYW50aWMgaW1wb3J0IEJhc2VNb2RlbCwgQ29uZmlnRGljdCwgRmllbGQsIFJvb3RNb2RlbCwgVHlwZUFkYXB0ZXIKZnJvbSBweWRhbnRpYy5maWVsZHMgaW1wb3J0IF9VbnNldApmcm9tIHB5ZGFudGljX2NvcmUgaW1wb3J0IFB5ZGFudGljVW5kZWZpbmVkCgpmcm9tIGludm9rZWFpLmFwcC51dGlsLm1ldGFlbnVtIGltcG9ydCBNZXRhRW51bQpmcm9tIGludm9rZWFpLmJhY2tlbmQuaW1hZ2VfdXRpbC5zZWdtZW50X2FueXRoaW5nLnNoYXJlZCBpbXBvcnQgQm91bmRpbmdCb3gKZnJvbSBpbnZva2VhaS5iYWNrZW5kLm1vZGVsX21hbmFnZXIudGF4b25vbXkgaW1wb3J0ICgKICAgIEJhc2VNb2RlbFR5cGUsCiAgICBDbGlwVmFyaWFudFR5cGUsCiAgICBNb2RlbEZvcm1hdCwKICAgIE1vZGVsVHlwZSwKICAgIE1vZGVsVmFyaWFudFR5cGUsCikKZnJvbSBpbnZva2VhaS5iYWNrZW5kLnV0aWwubG9nZ2luZyBpbXBvcnQgSW52b2tlQUlMb2dnZXIKCmxvZ2dlciA9IEludm9rZUFJTG9nZ2VyLmdldF9sb2dnZXIoKQoKCmNsYXNzIFVJVHlwZShzdHIsIEVudW0sIG1ldGFjbGFzcz1NZXRhRW51bSk6CiAgICAiIiIKICAgIFR5cGUgaGludHMgZm9yIHRoZSBVSSBmb3Igc2l0dWF0aW9ucyBpbiB3aGljaCB0aGUgZmllbGQgdHlwZSBpcyBub3QgZW5vdWdoIHRvIGluZmVyIHRoZSBjb3JyZWN0IFVJIHR5cGUuCgogICAgLSBNb2RlbCBGaWVsZHMKICAgIFRoZSBtb3N0IGNvbW1vbiBub2RlLWF1dGhvci1mYWNpbmcgdXNlIHdpbGwgYmUgZm9yIG1vZGVsIGZpZWxkcy4gSW50ZXJuYWxseSwgdGhlcmUgaXMgbm8gZGlmZmVyZW5jZQogICAgYmV0d2VlbiBTRC0xLCBTRC0yIGFuZCBTRFhMIG1vZGVsIGZpZWxkcyAtIHRoZXkgYWxsIHVzZSB0aGUgY2xhc3MgYE1haW5Nb2RlbEZpZWxkYC4gVG8gZW5zdXJlIHRoZQogICAgYmFzZS1tb2RlbC1zcGVjaWZpYyBVSSBpcyByZW5kZXJlZCwgdXNlIGUuZy4gYHVpX3R5cGU9VUlUeXBlLlNEWExNYWluTW9kZWxGaWVsZGAgdG8gaW5kaWNhdGUgdGhhdAogICAgdGhlIGZpZWxkIGlzIGFuIFNEWEwgbWFpbiBtb2RlbCBmaWVsZC4KCiAgICAtIEFueSBGaWVsZAogICAgV2UgY2Fubm90IGluZmVyIHRoZSB1c2FnZSBvZiBgdHlwaW5nLkFueWAgdmlhIHNjaGVtYSBwYXJzaW5nLCBzbyB5b3UgKm11c3QqIHVzZSBgdWlfdHlwZT1VSVR5cGUuQW55YCB0bwogICAgaW5kaWNhdGUgdGhhdCB0aGUgZmllbGQgYWNjZXB0cyBhbnkgdHlwZS4gVXNlIHdpdGggY2F1dGlvbi4gVGhpcyBjYW5ub3QgYmUgdXNlZCBvbiBvdXRwdXRzLgoKICAgIC0gU2NoZWR1bGVyIEZpZWxkCiAgICBTcGVjaWFsIGhhbmRsaW5nIGluIHRoZSBVSSBpcyBuZWVkZWQgZm9yIHRoaXMgZmllbGQsIHdoaWNoIG90aGVyd2lzZSB3b3VsZCBiZSBwYXJzZWQgYXMgYSBwbGFpbiBlbnVtIGZpZWxkLgoKICAgIC0gSW50ZXJuYWwgRmllbGRzCiAgICBTaW1pbGFyIHRvIHRoZSBBbnkgRmllbGQsIHRoZSBgY29sbGVjdGAgYW5kIGBpdGVyYXRlYCBub2RlcyBtYWtlIHVzZSBvZiBgdHlwaW5nLkFueWAuIFRvIGZhY2lsaXRhdGUKICAgIGhhbmRsaW5nIHRoZXNlIHR5cGVzIGluIHRoZSBjbGllbnQsIHdlIHVzZSBgVUlUeXBlLl9Db2xsZWN0aW9uYCBhbmQgYFVJVHlwZS5fQ29sbGVjdGlvbkl0ZW1gLiBUaGVzZQogICAgc2hvdWxkIG5vdCBiZSB1c2VkIGJ5IG5vZGUgYXV0aG9ycy4KCiAgICAtIERFUFJFQ0FURUQgRmllbGRzCiAgICBUaGVzZSB0eXBlcyBhcmUgZGVwcmVjYXRlZCBhbmQgc2hvdWxkIG5vdCBiZSB1c2VkIGJ5IG5vZGUgYXV0aG9ycy4gQSB3YXJuaW5nIHdpbGwgYmUgbG9nZ2VkIGlmIG9uZSBpcwogICAgdXNlZCwgYW5kIHRoZSB0eXBlIHdpbGwgYmUgaWdub3JlZC4gVGhleSBhcmUgaW5jbHVkZWQgaGVyZSBmb3IgYmFja3dhcmRzIGNvbXBhdGliaWxpdHkuCiAgICAiIiIKCiAgICAjIHJlZ2lvbiBNaXNjIEZpZWxkIFR5cGVzCiAgICBTY2hlZHVsZXIgPSAiU2NoZWR1bGVyRmllbGQiCiAgICBBbnkgPSAiQW55RmllbGQiCiAgICAjIGVuZHJlZ2lvbgoKICAgICMgcmVnaW9uIEludGVybmFsIEZpZWxkIFR5cGVzCiAgICBfQ29sbGVjdGlvbiA9ICJDb2xsZWN0aW9uRmllbGQiCiAgICBfQ29sbGVjdGlvbkl0ZW0gPSAiQ29sbGVjdGlvbkl0ZW1GaWVsZCIKICAgIF9Jc0ludGVybWVkaWF0ZSA9ICJJc0ludGVybWVkaWF0ZSIKICAgICMgZW5kcmVnaW9uCgogICAgIyByZWdpb24gREVQUkVDQVRFRAogICAgQm9vbGVhbiA9ICJERVBSRUNBVEVEX0Jvb2xlYW4iCiAgICBDb2xvciA9ICJERVBSRUNBVEVEX0NvbG9yIgogICAgQ29uZGl0aW9uaW5nID0gIkRFUFJFQ0FURURfQ29uZGl0aW9uaW5nIgogICAgQ29udHJvbCA9ICJERVBSRUNBVEVEX0NvbnRyb2wiCiAgICBGbG9hdCA9ICJERVBSRUNBVEVEX0Zsb2F0IgogICAgSW1hZ2UgPSAiREVQUkVDQVRFRF9JbWFnZSIKICAgIEludGVnZXIgPSAiREVQUkVDQVRFRF9JbnRlZ2VyIgogICAgTGF0ZW50cyA9ICJERVBSRUNBVEVEX0xhdGVudHMiCiAgICBTdHJpbmcgPSAiREVQUkVDQVRFRF9TdHJpbmciCiAgICBCb29sZWFuQ29sbGVjdGlvbiA9ICJERVBSRUNBVEVEX0Jvb2xlYW5Db2xsZWN0aW9uIgogICAgQ29sb3JDb2xsZWN0aW9uID0gIkRFUFJFQ0FURURfQ29sb3JDb2xsZWN0aW9uIgogICAgQ29uZGl0aW9uaW5nQ29sbGVjdGlvbiA9ICJERVBSRUNBVEVEX0NvbmRpdGlvbmluZ0NvbGxlY3Rpb24iCiAgICBDb250cm9sQ29sbGVjdGlvbiA9ICJERVBSRUNBVEVEX0NvbnRyb2xDb2xsZWN0aW9uIgogICAgRmxvYXRDb2xsZWN0aW9uID0gIkRFUFJFQ0FURURfRmxvYXRDb2xsZWN0aW9uIgogICAgSW1hZ2VDb2xsZWN0aW9uID0gIkRFUFJFQ0FURURfSW1hZ2VDb2xsZWN0aW9uIgogICAgSW50ZWdlckNvbGxlY3Rpb24gPSAiREVQUkVDQVRFRF9JbnRlZ2VyQ29sbGVjdGlvbiIKICAgIExhdGVudHNDb2xsZWN0aW9uID0gIkRFUFJFQ0FURURfTGF0ZW50c0NvbGxlY3Rpb24iCiAgICBTdHJpbmdDb2xsZWN0aW9uID0gIkRFUFJFQ0FURURfU3RyaW5nQ29sbGVjdGlvbiIKICAgIEJvb2xlYW5Qb2x5bW9ycGhpYyA9ICJERVBSRUNBVEVEX0Jvb2xlYW5Qb2x5bW9ycGhpYyIKICAgIENvbG9yUG9seW1vcnBoaWMgPSAiREVQUkVDQVRFRF9Db2xvclBvbHltb3JwaGljIgogICAgQ29uZGl0aW9uaW5nUG9seW1vcnBoaWMgPSAiREVQUkVDQVRFRF9Db25kaXRpb25pbmdQb2x5bW9ycGhpYyIKICAgIENvbnRyb2xQb2x5bW9ycGhpYyA9ICJERVBSRUNBVEVEX0NvbnRyb2xQb2x5bW9ycGhpYyIKICAgIEZsb2F0UG9seW1vcnBoaWMgPSAiREVQUkVDQVRFRF9GbG9hdFBvbHltb3JwaGljIgogICAgSW1hZ2VQb2x5bW9ycGhpYyA9ICJERVBSRUNBVEVEX0ltYWdlUG9seW1vcnBoaWMiCiAgICBJbnRlZ2VyUG9seW1vcnBoaWMgPSAiREVQUkVDQVRFRF9JbnRlZ2VyUG9seW1vcnBoaWMiCiAgICBMYXRlbnRzUG9seW1vcnBoaWMgPSAiREVQUkVDQVRFRF9MYXRlbnRzUG9seW1vcnBoaWMiCiAgICBTdHJpbmdQb2x5bW9ycGhpYyA9ICJERVBSRUNBVEVEX1N0cmluZ1BvbHltb3JwaGljIgogICAgVU5ldCA9ICJERVBSRUNBVEVEX1VOZXQiCiAgICBWYWUgPSAiREVQUkVDQVRFRF9WYWUiCiAgICBDTElQID0gIkRFUFJFQ0FURURfQ0xJUCIKICAgIENvbGxlY3Rpb24gPSAiREVQUkVDQVRFRF9Db2xsZWN0aW9uIgogICAgQ29sbGVjdGlvbkl0ZW0gPSAiREVQUkVDQVRFRF9Db2xsZWN0aW9uSXRlbSIKICAgIEVudW0gPSAiREVQUkVDQVRFRF9FbnVtIgogICAgV29ya2Zsb3dGaWVsZCA9ICJERVBSRUNBVEVEX1dvcmtmbG93RmllbGQiCiAgICBCb2FyZEZpZWxkID0gIkRFUFJFQ0FURURfQm9hcmRGaWVsZCIKICAgIE1ldGFkYXRhSXRlbSA9ICJERVBSRUNBVEVEX01ldGFkYXRhSXRlbSIKICAgIE1ldGFkYXRhSXRlbUNvbGxlY3Rpb24gPSAiREVQUkVDQVRFRF9NZXRhZGF0YUl0ZW1Db2xsZWN0aW9uIgogICAgTWV0YWRhdGFJdGVtUG9seW1vcnBoaWMgPSAiREVQUkVDQVRFRF9NZXRhZGF0YUl0ZW1Qb2x5bW9ycGhpYyIKICAgIE1ldGFkYXRhRGljdCA9ICJERVBSRUNBVEVEX01ldGFkYXRhRGljdCIKCiAgICAjIERlcHJlY2F0ZWQgTW9kZWwgRmllbGQgVHlwZXMgLSB1c2UgdWlfbW9kZWxfW2Jhc2V8dHlwZXx2YXJpYW50fGZvcm1hdF0gaW5zdGVhZAogICAgTWFpbk1vZGVsID0gIkRFUFJFQ0FURURfTWFpbk1vZGVsRmllbGQiCiAgICBDb2dWaWV3NE1haW5Nb2RlbCA9ICJERVBSRUNBVEVEX0NvZ1ZpZXc0TWFpbk1vZGVsRmllbGQiCiAgICBGbHV4TWFpbk1vZGVsID0gIkRFUFJFQ0FURURfRmx1eE1haW5Nb2RlbEZpZWxkIgogICAgU0QzTWFpbk1vZGVsID0gIkRFUFJFQ0FURURfU0QzTWFpbk1vZGVsRmllbGQiCiAgICBTRFhMTWFpbk1vZGVsID0gIkRFUFJFQ0FURURfU0RYTE1haW5Nb2RlbEZpZWxkIgogICAgU0RYTFJlZmluZXJNb2RlbCA9ICJERVBSRUNBVEVEX1NEWExSZWZpbmVyTW9kZWxGaWVsZCIKICAgIE9OTlhNb2RlbCA9ICJERVBSRUNBVEVEX09OTlhNb2RlbEZpZWxkIgogICAgVkFFTW9kZWwgPSAiREVQUkVDQVRFRF9WQUVNb2RlbEZpZWxkIgogICAgRmx1eFZBRU1vZGVsID0gIkRFUFJFQ0FURURfRmx1eFZBRU1vZGVsRmllbGQiCiAgICBMb1JBTW9kZWwgPSAiREVQUkVDQVRFRF9Mb1JBTW9kZWxGaWVsZCIKICAgIENvbnRyb2xOZXRNb2RlbCA9ICJERVBSRUNBVEVEX0NvbnRyb2xOZXRNb2RlbEZpZWxkIgogICAgSVBBZGFwdGVyTW9kZWwgPSAiREVQUkVDQVRFRF9JUEFkYXB0ZXJNb2RlbEZpZWxkIgogICAgVDJJQWRhcHRlck1vZGVsID0gIkRFUFJFQ0FURURfVDJJQWRhcHRlck1vZGVsRmllbGQiCiAgICBUNUVuY29kZXJNb2RlbCA9ICJERVBSRUNBVEVEX1Q1RW5jb2Rlck1vZGVsRmllbGQiCiAgICBDTElQRW1iZWRNb2RlbCA9ICJERVBSRUNBVEVEX0NMSVBFbWJlZE1vZGVsRmllbGQiCiAgICBDTElQTEVtYmVkTW9kZWwgPSAiREVQUkVDQVRFRF9DTElQTEVtYmVkTW9kZWxGaWVsZCIKICAgIENMSVBHRW1iZWRNb2RlbCA9ICJERVBSRUNBVEVEX0NMSVBHRW1iZWRNb2RlbEZpZWxkIgogICAgU3BhbmRyZWxJbWFnZVRvSW1hZ2VNb2RlbCA9ICJERVBSRUNBVEVEX1NwYW5kcmVsSW1hZ2VUb0ltYWdlTW9kZWxGaWVsZCIKICAgIENvbnRyb2xMb1JBTW9kZWwgPSAiREVQUkVDQVRFRF9Db250cm9sTG9SQU1vZGVsRmllbGQiCiAgICBTaWdMaXBNb2RlbCA9ICJERVBSRUNBVEVEX1NpZ0xpcE1vZGVsRmllbGQiCiAgICBGbHV4UmVkdXhNb2RlbCA9ICJERVBSRUNBVEVEX0ZsdXhSZWR1eE1vZGVsRmllbGQiCiAgICBMbGF2YU9uZXZpc2lvbk1vZGVsID0gIkRFUFJFQ0FURURfTExhVkFNb2RlbEZpZWxkIgogICAgSW1hZ2VuM01vZGVsID0gIkRFUFJFQ0FURURfSW1hZ2VuM01vZGVsRmllbGQiCiAgICBJbWFnZW40TW9kZWwgPSAiREVQUkVDQVRFRF9JbWFnZW40TW9kZWxGaWVsZCIKICAgIENoYXRHUFQ0b01vZGVsID0gIkRFUFJFQ0FURURfQ2hhdEdQVDRvTW9kZWxGaWVsZCIKICAgIEdlbWluaTJfNU1vZGVsID0gIkRFUFJFQ0FURURfR2VtaW5pMl81TW9kZWxGaWVsZCIKICAgIEZsdXhLb250ZXh0TW9kZWwgPSAiREVQUkVDQVRFRF9GbHV4S29udGV4dE1vZGVsRmllbGQiCiAgICBWZW8zTW9kZWwgPSAiREVQUkVDQVRFRF9WZW8zTW9kZWxGaWVsZCIKICAgIFJ1bndheU1vZGVsID0gIkRFUFJFQ0FURURfUnVud2F5TW9kZWxGaWVsZCIKICAgICMgZW5kcmVnaW9uCgoKY2xhc3MgVUlDb21wb25lbnQoc3RyLCBFbnVtLCBtZXRhY2xhc3M9TWV0YUVudW0pOgogICAgIiIiCiAgICBUaGUgdHlwZSBvZiBVSSBjb21wb25lbnQgdG8gdXNlIGZvciBhIGZpZWxkLCB1c2VkIHRvIG92ZXJyaWRlIHRoZSBkZWZhdWx0IGNvbXBvbmVudHMsIHdoaWNoIGFyZQogICAgaW5mZXJyZWQgZnJvbSB0aGUgZmllbGQgdHlwZS4KICAgICIiIgoKICAgIE5vbmVfID0gIm5vbmUiCiAgICBUZXh0YXJlYSA9ICJ0ZXh0YXJlYSIKICAgIFNsaWRlciA9ICJzbGlkZXIiCgoKY2xhc3MgRmllbGREZXNjcmlwdGlvbnM6CiAgICBkZW5vaXNpbmdfc3RhcnQgPSAiV2hlbiB0byBzdGFydCBkZW5vaXNpbmcsIGV4cHJlc3NlZCBhIHBlcmNlbnRhZ2Ugb2YgdG90YWwgc3RlcHMiCiAgICBkZW5vaXNpbmdfZW5kID0gIldoZW4gdG8gc3RvcCBkZW5vaXNpbmcsIGV4cHJlc3NlZCBhIHBlcmNlbnRhZ2Ugb2YgdG90YWwgc3RlcHMiCiAgICBjZmdfc2NhbGUgPSAiQ2xhc3NpZmllci1GcmVlIEd1aWRhbmNlIHNjYWxlIgogICAgY2ZnX3Jlc2NhbGVfbXVsdGlwbGllciA9ICJSZXNjYWxlIG11bHRpcGxpZXIgZm9yIENGRyBndWlkYW5jZSwgdXNlZCBmb3IgbW9kZWxzIHRyYWluZWQgd2l0aCB6ZXJvLXRlcm1pbmFsIFNOUiIKICAgIHNjaGVkdWxlciA9ICJTY2hlZHVsZXIgdG8gdXNlIGR1cmluZyBpbmZlcmVuY2UiCiAgICBwb3NpdGl2ZV9jb25kID0gIlBvc2l0aXZlIGNvbmRpdGlvbmluZyB0ZW5zb3IiCiAgICBuZWdhdGl2ZV9jb25kID0gIk5lZ2F0aXZlIGNvbmRpdGlvbmluZyB0ZW5zb3IiCiAgICBub2lzZSA9ICJOb2lzZSB0ZW5zb3IiCiAgICBjbGlwID0gIkNMSVAgKHRva2VuaXplciwgdGV4dCBlbmNvZGVyLCBMb1JBcykgYW5kIHNraXBwZWQgbGF5ZXIgY291bnQiCiAgICB0NV9lbmNvZGVyID0gIlQ1IHRva2VuaXplciBhbmQgdGV4dCBlbmNvZGVyIgogICAgZ2xtX2VuY29kZXIgPSAiR0xNIChUSFVETSkgdG9rZW5pemVyIGFuZCB0ZXh0IGVuY29kZXIiCiAgICBxd2VuM19lbmNvZGVyID0gIlF3ZW4zIHRva2VuaXplciBhbmQgdGV4dCBlbmNvZGVyIgogICAgY2xpcF9lbWJlZF9tb2RlbCA9ICJDTElQIEVtYmVkIGxvYWRlciIKICAgIGNsaXBfZ19tb2RlbCA9ICJDTElQLUcgRW1iZWQgbG9hZGVyIgogICAgdW5ldCA9ICJVTmV0IChzY2hlZHVsZXIsIExvUkFzKSIKICAgIHRyYW5zZm9ybWVyID0gIlRyYW5zZm9ybWVyIgogICAgbW1kaXR4ID0gIk1NRGlUWCIKICAgIHZhZSA9ICJWQUUiCiAgICBjb25kID0gIkNvbmRpdGlvbmluZyB0ZW5zb3IiCiAgICBjb250cm9sbmV0X21vZGVsID0gIkNvbnRyb2xOZXQgbW9kZWwgdG8gbG9hZCIKICAgIHZhZV9tb2RlbCA9ICJWQUUgbW9kZWwgdG8gbG9hZCIKICAgIGxvcmFfbW9kZWwgPSAiTG9SQSBtb2RlbCB0byBsb2FkIgogICAgY29udHJvbF9sb3JhX21vZGVsID0gIkNvbnRyb2wgTG9SQSBtb2RlbCB0byBsb2FkIgogICAgbWFpbl9tb2RlbCA9ICJNYWluIG1vZGVsIChVTmV0LCBWQUUsIENMSVApIHRvIGxvYWQiCiAgICBmbHV4X21vZGVsID0gIkZsdXggbW9kZWwgKFRyYW5zZm9ybWVyKSB0byBsb2FkIgogICAgc2QzX21vZGVsID0gIlNEMyBtb2RlbCAoTU1EaVRYKSB0byBsb2FkIgogICAgY29ndmlldzRfbW9kZWwgPSAiQ29nVmlldzQgbW9kZWwgKFRyYW5zZm9ybWVyKSB0byBsb2FkIgogICAgel9pbWFnZV9tb2RlbCA9ICJaLUltYWdlIG1vZGVsIChUcmFuc2Zvcm1lcikgdG8gbG9hZCIKICAgIHNkeGxfbWFpbl9tb2RlbCA9ICJTRFhMIE1haW4gbW9kZWwgKFVOZXQsIFZBRSwgQ0xJUDEsIENMSVAyKSB0byBsb2FkIgogICAgc2R4bF9yZWZpbmVyX21vZGVsID0gIlNEWEwgUmVmaW5lciBNYWluIE1vZGRlIChVTmV0LCBWQUUsIENMSVAyKSB0byBsb2FkIgogICAgb25ueF9tYWluX21vZGVsID0gIk9OTlggTWFpbiBtb2RlbCAoVU5ldCwgVkFFLCBDTElQKSB0byBsb2FkIgogICAgc3BhbmRyZWxfaW1hZ2VfdG9faW1hZ2VfbW9kZWwgPSAiSW1hZ2UtdG8tSW1hZ2UgbW9kZWwiCiAgICB2bGxtX21vZGVsID0gIlZMTE0gbW9kZWwiCiAgICBsb3JhX3dlaWdodCA9ICJUaGUgd2VpZ2h0IGF0IHdoaWNoIHRoZSBMb1JBIGlzIGFwcGxpZWQgdG8gZWFjaCBtb2RlbCIKICAgIGNvbXBlbF9wcm9tcHQgPSAiUHJvbXB0IHRvIGJlIHBhcnNlZCBieSBDb21wZWwgdG8gY3JlYXRlIGEgY29uZGl0aW9uaW5nIHRlbnNvciIKICAgIHJhd19wcm9tcHQgPSAiUmF3IHByb21wdCB0ZXh0IChubyBwYXJzaW5nKSIKICAgIHNkeGxfYWVzdGhldGljID0gIlRoZSBhZXN0aGV0aWMgc2NvcmUgdG8gYXBwbHkgdG8gdGhlIGNvbmRpdGlvbmluZyB0ZW5zb3IiCiAgICBza2lwcGVkX2xheWVycyA9ICJOdW1iZXIgb2YgbGF5ZXJzIHRvIHNraXAgaW4gdGV4dCBlbmNvZGVyIgogICAgc2VlZCA9ICJTZWVkIGZvciByYW5kb20gbnVtYmVyIGdlbmVyYXRpb24iCiAgICBzdGVwcyA9ICJOdW1iZXIgb2Ygc3RlcHMgdG8gcnVuIgogICAgd2lkdGggPSAiV2lkdGggb2Ygb3V0cHV0IChweCkiCiAgICBoZWlnaHQgPSAiSGVpZ2h0IG9mIG91dHB1dCAocHgpIgogICAgY29udHJvbCA9ICJDb250cm9sTmV0KHMpIHRvIGFwcGx5IgogICAgaXBfYWRhcHRlciA9ICJJUC1BZGFwdGVyIHRvIGFwcGx5IgogICAgdDJpX2FkYXB0ZXIgPSAiVDJJLUFkYXB0ZXIocykgdG8gYXBwbHkiCiAgICBkZW5vaXNlZF9sYXRlbnRzID0gIkRlbm9pc2VkIGxhdGVudHMgdGVuc29yIgogICAgbGF0ZW50cyA9ICJMYXRlbnRzIHRlbnNvciIKICAgIHN0cmVuZ3RoID0gIlN0cmVuZ3RoIG9mIGRlbm9pc2luZyAocHJvcG9ydGlvbmFsIHRvIHN0ZXBzKSIKICAgIG1ldGFkYXRhID0gIk9wdGlvbmFsIG1ldGFkYXRhIHRvIGJlIHNhdmVkIHdpdGggdGhlIGltYWdlIgogICAgbWV0YWRhdGFfY29sbGVjdGlvbiA9ICJDb2xsZWN0aW9uIG9mIE1ldGFkYXRhIgogICAgbWV0YWRhdGFfaXRlbV9wb2x5bW9ycGhpYyA9ICJBIHNpbmdsZSBtZXRhZGF0YSBpdGVtIG9yIGNvbGxlY3Rpb24gb2YgbWV0YWRhdGEgaXRlbXMiCiAgICBtZXRhZGF0YV9pdGVtX2xhYmVsID0gIkxhYmVsIGZvciB0aGlzIG1ldGFkYXRhIGl0ZW0iCiAgICBtZXRhZGF0YV9pdGVtX3ZhbHVlID0gIlRoZSB2YWx1ZSBmb3IgdGhpcyBtZXRhZGF0YSBpdGVtIChtYXkgYmUgYW55IHR5cGUpIgogICAgd29ya2Zsb3cgPSAiT3B0aW9uYWwgd29ya2Zsb3cgdG8gYmUgc2F2ZWQgd2l0aCB0aGUgaW1hZ2UiCiAgICBpbnRlcnBfbW9kZSA9ICJJbnRlcnBvbGF0aW9uIG1vZGUiCiAgICB0b3JjaF9hbnRpYWxpYXMgPSAiV2hldGhlciBvciBub3QgdG8gYXBwbHkgYW50aWFsaWFzaW5nIChiaWxpbmVhciBvciBiaWN1YmljIG9ubHkpIgogICAgZnAzMiA9ICJXaGV0aGVyIG9yIG5vdCB0byB1c2UgZnVsbCBmbG9hdDMyIHByZWNpc2lvbiIKICAgIHByZWNpc2lvbiA9ICJQcmVjaXNpb24gdG8gdXNlIgogICAgdGlsZWQgPSAiUHJvY2Vzc2luZyB1c2luZyBvdmVybGFwcGluZyB0aWxlcyAocmVkdWNlIG1lbW9yeSBjb25zdW1wdGlvbikiCiAgICB2YWVfdGlsZV9zaXplID0gIlRoZSB0aWxlIHNpemUgZm9yIFZBRSB0aWxpbmcgaW4gcGl4ZWxzIChpbWFnZSBzcGFjZSkuIElmIHNldCB0byAwLCB0aGUgZGVmYXVsdCB0aWxlIHNpemUgZm9yIHRoZSBtb2RlbCB3aWxsIGJlIHVzZWQuIExhcmdlciB0aWxlIHNpemVzIGdlbmVyYWxseSBwcm9kdWNlIGJldHRlciByZXN1bHRzIGF0IHRoZSBjb3N0IG9mIGhpZ2hlciBtZW1vcnkgdXNhZ2UuIgogICAgZGV0ZWN0X3JlcyA9ICJQaXhlbCByZXNvbHV0aW9uIGZvciBkZXRlY3Rpb24iCiAgICBpbWFnZV9yZXMgPSAiUGl4ZWwgcmVzb2x1dGlvbiBmb3Igb3V0cHV0IGltYWdlIgogICAgc2FmZV9tb2RlID0gIldoZXRoZXIgb3Igbm90IHRvIHVzZSBzYWZlIG1vZGUiCiAgICBzY3JpYmJsZV9tb2RlID0gIldoZXRoZXIgb3Igbm90IHRvIHVzZSBzY3JpYmJsZSBtb2RlIgogICAgc2NhbGVfZmFjdG9yID0gIlRoZSBmYWN0b3IgYnkgd2hpY2ggdG8gc2NhbGUiCiAgICBibGVuZF9hbHBoYSA9ICgKICAgICAgICAiQmxlbmRpbmcgZmFjdG9yLiAwLjAgPSB1c2UgaW5wdXQgQSBvbmx5LCAxLjAgPSB1c2UgaW5wdXQgQiBvbmx5LCAwLjUgPSA1MCUgbWl4IG9mIGlucHV0IEEgYW5kIGlucHV0IEIuIgogICAgKQogICAgbnVtXzEgPSAiVGhlIGZpcnN0IG51bWJlciIKICAgIG51bV8yID0gIlRoZSBzZWNvbmQgbnVtYmVyIgogICAgZGVub2lzZV9tYXNrID0gIkEgbWFzayBvZiB0aGUgcmVnaW9uIHRvIGFwcGx5IHRoZSBkZW5vaXNpbmcgcHJvY2VzcyB0by4gVmFsdWVzIG9mIDAuMCByZXByZXNlbnQgdGhlIHJlZ2lvbnMgdG8gYmUgZnVsbHkgZGVub2lzZWQsIGFuZCAxLjAgcmVwcmVzZW50IHRoZSByZWdpb25zIHRvIGJlIHByZXNlcnZlZC4iCiAgICBib2FyZCA9ICJUaGUgYm9hcmQgdG8gc2F2ZSB0aGUgaW1hZ2UgdG8iCiAgICBpbWFnZSA9ICJUaGUgaW1hZ2UgdG8gcHJvY2VzcyIKICAgIHRpbGVfc2l6ZSA9ICJUaWxlIHNpemUiCiAgICBpbmNsdXNpdmVfbG93ID0gIlRoZSBpbmNsdXNpdmUgbG93IHZhbHVlIgogICAgZXhjbHVzaXZlX2hpZ2ggPSAiVGhlIGV4Y2x1c2l2ZSBoaWdoIHZhbHVlIgogICAgZGVjaW1hbF9wbGFjZXMgPSAiVGhlIG51bWJlciBvZiBkZWNpbWFsIHBsYWNlcyB0byByb3VuZCB0byIKICAgIGZyZWV1X3MxID0gJ1NjYWxpbmcgZmFjdG9yIGZvciBzdGFnZSAxIHRvIGF0dGVudWF0ZSB0aGUgY29udHJpYnV0aW9ucyBvZiB0aGUgc2tpcCBmZWF0dXJlcy4gVGhpcyBpcyBkb25lIHRvIG1pdGlnYXRlIHRoZSAib3ZlcnNtb290aGluZyBlZmZlY3QiIGluIHRoZSBlbmhhbmNlZCBkZW5vaXNpbmcgcHJvY2Vzcy4nCiAgICBmcmVldV9zMiA9ICdTY2FsaW5nIGZhY3RvciBmb3Igc3RhZ2UgMiB0byBhdHRlbnVhdGUgdGhlIGNvbnRyaWJ1dGlvbnMgb2YgdGhlIHNraXAgZmVhdHVyZXMuIFRoaXMgaXMgZG9uZSB0byBtaXRpZ2F0ZSB0aGUgIm92ZXJzbW9vdGhpbmcgZWZmZWN0IiBpbiB0aGUgZW5oYW5jZWQgZGVub2lzaW5nIHByb2Nlc3MuJwogICAgZnJlZXVfYjEgPSAiU2NhbGluZyBmYWN0b3IgZm9yIHN0YWdlIDEgdG8gYW1wbGlmeSB0aGUgY29udHJpYnV0aW9ucyBvZiBiYWNrYm9uZSBmZWF0dXJlcy4iCiAgICBmcmVldV9iMiA9ICJTY2FsaW5nIGZhY3RvciBmb3Igc3RhZ2UgMiB0byBhbXBsaWZ5IHRoZSBjb250cmlidXRpb25zIG9mIGJhY2tib25lIGZlYXR1cmVzLiIKICAgIGluc3RhbnR4X2NvbnRyb2xfbW9kZSA9ICJUaGUgY29udHJvbCBtb2RlIGZvciBJbnN0YW50WCBDb250cm9sTmV0IHVuaW9uIG1vZGVscy4gSWdub3JlZCBmb3Igb3RoZXIgQ29udHJvbE5ldCBtb2RlbHMuIFRoZSBzdGFuZGFyZCBtYXBwaW5nIGlzOiBjYW5ueSAoMCksIHRpbGUgKDEpLCBkZXB0aCAoMiksIGJsdXIgKDMpLCBwb3NlICg0KSwgZ3JheSAoNSksIGxvdyBxdWFsaXR5ICg2KS4gTmVnYXRpdmUgdmFsdWVzIHdpbGwgYmUgdHJlYXRlZCBhcyAnTm9uZScuIgogICAgZmx1eF9yZWR1eF9jb25kaXRpb25pbmcgPSAiRkxVWCBSZWR1eCBjb25kaXRpb25pbmcgdGVuc29yIgogICAgdmxsbV9tb2RlbCA9ICJUaGUgVkxMTSBtb2RlbCB0byB1c2UiCiAgICBmbHV4X2ZpbGxfY29uZGl0aW9uaW5nID0gIkZMVVggRmlsbCBjb25kaXRpb25pbmcgdGVuc29yIgogICAgZmx1eF9rb250ZXh0X2NvbmRpdGlvbmluZyA9ICJGTFVYIEtvbnRleHQgY29uZGl0aW9uaW5nIChyZWZlcmVuY2UgaW1hZ2UpIgoKCmNsYXNzIEltYWdlRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkFuIGltYWdlIHByaW1pdGl2ZSBmaWVsZCIiIgoKICAgIGltYWdlX25hbWU6IHN0ciA9IEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgbmFtZSBvZiB0aGUgaW1hZ2UiKQoKCmNsYXNzIFZpZGVvRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgdmlkZW8gcHJpbWl0aXZlIGZpZWxkIiIiCgogICAgdmlkZW9fbmFtZTogc3RyID0gRmllbGQoZGVzY3JpcHRpb249IlRoZSBuYW1lIG9mIHRoZSB2aWRlbyIpCgoKY2xhc3MgQXVkaW9GaWVsZChCYXNlTW9kZWwpOgogICAgIiIiQW4gYXVkaW8gcHJpbWl0aXZlIGZpZWxkIiIiCgogICAgYXVkaW9fbmFtZTogc3RyID0gRmllbGQoZGVzY3JpcHRpb249IlRoZSBuYW1lIG9mIHRoZSBhdWRpbyIpCmNsYXNzIEJvYXJkRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgYm9hcmQgcHJpbWl0aXZlIGZpZWxkIiIiCgogICAgYm9hcmRfaWQ6IHN0ciA9IEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgaWQgb2YgdGhlIGJvYXJkIikKCgpjbGFzcyBTdHlsZVByZXNldEZpZWxkKEJhc2VNb2RlbCk6CiAgICAiIiJBIHN0eWxlIHByZXNldCBwcmltaXRpdmUgZmllbGQiIiIKCiAgICBzdHlsZV9wcmVzZXRfaWQ6IHN0ciA9IEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgaWQgb2YgdGhlIHN0eWxlIHByZXNldCIpCgoKY2xhc3MgRGVub2lzZU1hc2tGaWVsZChCYXNlTW9kZWwpOgogICAgIiIiQW4gaW5wYWludCBtYXNrIGZpZWxkIiIiCgogICAgbWFza19uYW1lOiBzdHIgPSBGaWVsZChkZXNjcmlwdGlvbj0iVGhlIG5hbWUgb2YgdGhlIG1hc2sgaW1hZ2UiKQogICAgbWFza2VkX2xhdGVudHNfbmFtZTogT3B0aW9uYWxbc3RyXSA9IEZpZWxkKGRlZmF1bHQ9Tm9uZSwgZGVzY3JpcHRpb249IlRoZSBuYW1lIG9mIHRoZSBtYXNrZWQgaW1hZ2UgbGF0ZW50cyIpCiAgICBncmFkaWVudDogYm9vbCA9IEZpZWxkKGRlZmF1bHQ9RmFsc2UsIGRlc2NyaXB0aW9uPSJVc2VkIGZvciBncmFkaWVudCBpbnBhaW50aW5nIikKCgpjbGFzcyBUZW5zb3JGaWVsZChCYXNlTW9kZWwpOgogICAgIiIiQSB0ZW5zb3IgcHJpbWl0aXZlIGZpZWxkLiIiIgoKICAgIHRlbnNvcl9uYW1lOiBzdHIgPSBGaWVsZChkZXNjcmlwdGlvbj0iVGhlIG5hbWUgb2YgYSB0ZW5zb3IuIikKCgpjbGFzcyBMYXRlbnRzRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgbGF0ZW50cyB0ZW5zb3IgcHJpbWl0aXZlIGZpZWxkIiIiCgogICAgbGF0ZW50c19uYW1lOiBzdHIgPSBGaWVsZChkZXNjcmlwdGlvbj0iVGhlIG5hbWUgb2YgdGhlIGxhdGVudHMiKQogICAgc2VlZDogT3B0aW9uYWxbaW50XSA9IEZpZWxkKGRlZmF1bHQ9Tm9uZSwgZGVzY3JpcHRpb249IlNlZWQgdXNlZCB0byBnZW5lcmF0ZSB0aGlzIGxhdGVudHMiKQoKCmNsYXNzIENvbG9yRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgY29sb3IgcHJpbWl0aXZlIGZpZWxkIiIiCgogICAgcjogaW50ID0gRmllbGQoZ2U9MCwgbGU9MjU1LCBkZXNjcmlwdGlvbj0iVGhlIHJlZCBjb21wb25lbnQiKQogICAgZzogaW50ID0gRmllbGQoZ2U9MCwgbGU9MjU1LCBkZXNjcmlwdGlvbj0iVGhlIGdyZWVuIGNvbXBvbmVudCIpCiAgICBiOiBpbnQgPSBGaWVsZChnZT0wLCBsZT0yNTUsIGRlc2NyaXB0aW9uPSJUaGUgYmx1ZSBjb21wb25lbnQiKQogICAgYTogaW50ID0gRmllbGQoZ2U9MCwgbGU9MjU1LCBkZXNjcmlwdGlvbj0iVGhlIGFscGhhIGNvbXBvbmVudCIpCgogICAgZGVmIHR1cGxlKHNlbGYpIC0+IFR1cGxlW2ludCwgaW50LCBpbnQsIGludF06CiAgICAgICAgcmV0dXJuIChzZWxmLnIsIHNlbGYuZywgc2VsZi5iLCBzZWxmLmEpCgoKY2xhc3MgRmx1eENvbmRpdGlvbmluZ0ZpZWxkKEJhc2VNb2RlbCk6CiAgICAiIiJBIGNvbmRpdGlvbmluZyB0ZW5zb3IgcHJpbWl0aXZlIHZhbHVlIiIiCgogICAgY29uZGl0aW9uaW5nX25hbWU6IHN0ciA9IEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgbmFtZSBvZiBjb25kaXRpb25pbmcgdGVuc29yIikKICAgIG1hc2s6IE9wdGlvbmFsW1RlbnNvckZpZWxkXSA9IEZpZWxkKAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBkZXNjcmlwdGlvbj0iVGhlIG1hc2sgYXNzb2NpYXRlZCB3aXRoIHRoaXMgY29uZGl0aW9uaW5nIHRlbnNvci4gRXhjbHVkZWQgcmVnaW9ucyBzaG91bGQgYmUgc2V0IHRvIEZhbHNlLCAiCiAgICAgICAgImluY2x1ZGVkIHJlZ2lvbnMgc2hvdWxkIGJlIHNldCB0byBUcnVlLiIsCiAgICApCgoKY2xhc3MgRmx1eFJlZHV4Q29uZGl0aW9uaW5nRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgRkxVWCBSZWR1eCBjb25kaXRpb25pbmcgdGVuc29yIHByaW1pdGl2ZSB2YWx1ZSIiIgoKICAgIGNvbmRpdGlvbmluZzogVGVuc29yRmllbGQgPSBGaWVsZChkZXNjcmlwdGlvbj0iVGhlIFJlZHV4IGltYWdlIGNvbmRpdGlvbmluZyB0ZW5zb3IuIikKICAgIG1hc2s6IE9wdGlvbmFsW1RlbnNvckZpZWxkXSA9IEZpZWxkKAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBkZXNjcmlwdGlvbj0iVGhlIG1hc2sgYXNzb2NpYXRlZCB3aXRoIHRoaXMgY29uZGl0aW9uaW5nIHRlbnNvci4gRXhjbHVkZWQgcmVnaW9ucyBzaG91bGQgYmUgc2V0IHRvIEZhbHNlLCAiCiAgICAgICAgImluY2x1ZGVkIHJlZ2lvbnMgc2hvdWxkIGJlIHNldCB0byBUcnVlLiIsCiAgICApCgoKY2xhc3MgRmx1eEZpbGxDb25kaXRpb25pbmdGaWVsZChCYXNlTW9kZWwpOgogICAgIiIiQSBGTFVYIEZpbGwgY29uZGl0aW9uaW5nIGZpZWxkLiIiIgoKICAgIGltYWdlOiBJbWFnZUZpZWxkID0gRmllbGQoZGVzY3JpcHRpb249IlRoZSBGTFVYIEZpbGwgcmVmZXJlbmNlIGltYWdlLiIpCiAgICBtYXNrOiBUZW5zb3JGaWVsZCA9IEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgRkxVWCBGaWxsIGlucGFpbnQgbWFzay4iKQoKCmNsYXNzIEZsdXhLb250ZXh0Q29uZGl0aW9uaW5nRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgY29uZGl0aW9uaW5nIGZpZWxkIGZvciBGTFVYIEtvbnRleHQgKHJlZmVyZW5jZSBpbWFnZSkuIiIiCgogICAgaW1hZ2U6IEltYWdlRmllbGQgPSBGaWVsZChkZXNjcmlwdGlvbj0iVGhlIEtvbnRleHQgcmVmZXJlbmNlIGltYWdlLiIpCgoKY2xhc3MgU0QzQ29uZGl0aW9uaW5nRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgY29uZGl0aW9uaW5nIHRlbnNvciBwcmltaXRpdmUgdmFsdWUiIiIKCiAgICBjb25kaXRpb25pbmdfbmFtZTogc3RyID0gRmllbGQoZGVzY3JpcHRpb249IlRoZSBuYW1lIG9mIGNvbmRpdGlvbmluZyB0ZW5zb3IiKQoKCmNsYXNzIENvZ1ZpZXc0Q29uZGl0aW9uaW5nRmllbGQoQmFzZU1vZGVsKToKICAgICIiIkEgY29uZGl0aW9uaW5nIHRlbnNvciBwcmltaXRpdmUgdmFsdWUiIiIKCiAgICBjb25kaXRpb25pbmdfbmFtZTogc3RyID0gRmllbGQoZGVzY3JpcHRpb249IlRoZSBuYW1lIG9mIGNvbmRpdGlvbmluZyB0ZW5zb3IiKQoKCmNsYXNzIFpJbWFnZUNvbmRpdGlvbmluZ0ZpZWxkKEJhc2VNb2RlbCk6CiAgICAiIiJBIFotSW1hZ2UgY29uZGl0aW9uaW5nIHRlbnNvciBwcmltaXRpdmUgdmFsdWUiIiIKCiAgICBjb25kaXRpb25pbmdfbmFtZTogc3RyID0gRmllbGQoZGVzY3JpcHRpb249IlRoZSBuYW1lIG9mIGNvbmRpdGlvbmluZyB0ZW5zb3IiKQogICAgbWFzazogT3B0aW9uYWxbVGVuc29yRmllbGRdID0gRmllbGQoCiAgICAgICAgZGVmYXVsdD1Ob25lLAogICAgICAgIGRlc2NyaXB0aW9uPSJUaGUgbWFzayBhc3NvY2lhdGVkIHdpdGggdGhpcyBjb25kaXRpb25pbmcgdGVuc29yIGZvciByZWdpb25hbCBwcm9tcHRpbmcuICIKICAgICAgICAiRXhjbHVkZWQgcmVnaW9ucyBzaG91bGQgYmUgc2V0IHRvIEZhbHNlLCBpbmNsdWRlZCByZWdpb25zIHNob3VsZCBiZSBzZXQgdG8gVHJ1ZS4iLAogICAgKQoKCmNsYXNzIENvbmRpdGlvbmluZ0ZpZWxkKEJhc2VNb2RlbCk6CiAgICAiIiJBIGNvbmRpdGlvbmluZyB0ZW5zb3IgcHJpbWl0aXZlIHZhbHVlIiIiCgogICAgY29uZGl0aW9uaW5nX25hbWU6IHN0ciA9IEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgbmFtZSBvZiBjb25kaXRpb25pbmcgdGVuc29yIikKICAgIG1hc2s6IE9wdGlvbmFsW1RlbnNvckZpZWxkXSA9IEZpZWxkKAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBkZXNjcmlwdGlvbj0iVGhlIG1hc2sgYXNzb2NpYXRlZCB3aXRoIHRoaXMgY29uZGl0aW9uaW5nIHRlbnNvci4gRXhjbHVkZWQgcmVnaW9ucyBzaG91bGQgYmUgc2V0IHRvIEZhbHNlLCAiCiAgICAgICAgImluY2x1ZGVkIHJlZ2lvbnMgc2hvdWxkIGJlIHNldCB0byBUcnVlLiIsCiAgICApCgoKY2xhc3MgQm91bmRpbmdCb3hGaWVsZChCb3VuZGluZ0JveCk6CiAgICAiIiJBIGJvdW5kaW5nIGJveCBwcmltaXRpdmUgdmFsdWUuIiIiCgogICAgc2NvcmU6IE9wdGlvbmFsW2Zsb2F0XSA9IEZpZWxkKAogICAgICAgIGRlZmF1bHQ9Tm9uZSwKICAgICAgICBnZT0wLjAsCiAgICAgICAgbGU9MS4wLAogICAgICAgIGRlc2NyaXB0aW9uPSJUaGUgc2NvcmUgYXNzb2NpYXRlZCB3aXRoIHRoZSBib3VuZGluZyBib3guIEluIHRoZSByYW5nZSBbMCwgMV0uIFRoaXMgdmFsdWUgaXMgdHlwaWNhbGx5IHNldCAiCiAgICAgICAgIndoZW4gdGhlIGJvdW5kaW5nIGJveCB3YXMgcHJvZHVjZWQgYnkgYSBkZXRlY3RvciBhbmQgaGFzIGFuIGFzc29jaWF0ZWQgY29uZmlkZW5jZSBzY29yZS4iLAogICAgKQoKCmNsYXNzIE1ldGFkYXRhRmllbGQoUm9vdE1vZGVsW2RpY3Rbc3RyLCBBbnldXSk6CiAgICAiIiIKICAgIFB5ZGFudGljIG1vZGVsIGZvciBtZXRhZGF0YSB3aXRoIGN1c3RvbSByb290IG9mIHR5cGUgZGljdFtzdHIsIEFueV0uCiAgICBNZXRhZGF0YSBpcyBzdG9yZWQgd2l0aG91dCBhIHN0cmljdCBzY2hlbWEuCiAgICAiIiIKCiAgICByb290OiBkaWN0W3N0ciwgQW55XSA9IEZpZWxkKGRlc2NyaXB0aW9uPSJUaGUgbWV0YWRhdGEiKQoKCk1ldGFkYXRhRmllbGRWYWxpZGF0b3IgPSBUeXBlQWRhcHRlcihNZXRhZGF0YUZpZWxkKQoKCmNsYXNzIElucHV0KHN0ciwgRW51bSwgbWV0YWNsYXNzPU1ldGFFbnVtKToKICAgICIiIgogICAgVGhlIHR5cGUgb2YgaW5wdXQgYSBmaWVsZCBhY2NlcHRzLgogICAgLSBgSW5wdXQuRGlyZWN0YDogVGhlIGZpZWxkIG11c3QgaGF2ZSBpdHMgdmFsdWUgcHJvdmlkZWQgZGlyZWN0bHksIHdoZW4gdGhlIGludm9jYXRpb24gYW5kIGZpZWxkIFwKICAgICAgYXJlIGluc3RhbnRpYXRlZC4KICAgIC0gYElucHV0LkNvbm5lY3Rpb25gOiBUaGUgZmllbGQgbXVzdCBoYXZlIGl0cyB2YWx1ZSBwcm92aWRlZCBieSBhIGNvbm5lY3Rpb24uCiAgICAtIGBJbnB1dC5BbnlgOiBUaGUgZmllbGQgbWF5IGhhdmUgaXRzIHZhbHVlIHByb3ZpZGVkIGVpdGhlciBkaXJlY3RseSBvciBieSBhIGNvbm5lY3Rpb24uCiAgICAiIiIKCiAgICBDb25uZWN0aW9uID0gImNvbm5lY3Rpb24iCiAgICBEaXJlY3QgPSAiZGlyZWN0IgogICAgQW55ID0gImFueSIKCgpjbGFzcyBGaWVsZEtpbmQoc3RyLCBFbnVtLCBtZXRhY2xhc3M9TWV0YUVudW0pOgogICAgIiIiCiAgICBUaGUga2luZCBvZiBmaWVsZC4KICAgIC0gYElucHV0YDogQW4gaW5wdXQgZmllbGQgb24gYSBub2RlLgogICAgLSBgT3V0cHV0YDogQW4gb3V0cHV0IGZpZWxkIG9uIGEgbm9kZS4KICAgIC0gYEludGVybmFsYDogQSBmaWVsZCB3aGljaCBpcyB0cmVhdGVkIGFzIGFuIGlucHV0LCBidXQgY2Fubm90IGJlIHVzZWQgaW4gbm9kZSBkZWZpbml0aW9ucy4gTWV0YWRhdGEgaXMKICAgIG9uZSBleGFtcGxlLiBJdCBpcyBwcm92aWRlZCB0byBub2RlcyB2aWEgdGhlIFdpdGhNZXRhZGF0YSBjbGFzcywgYW5kIHdlIHdhbnQgdG8gcmVzZXJ2ZSB0aGUgZmllbGQgbmFtZQogICAgIm1ldGFkYXRhIiBmb3IgdGhpcyBvbiBhbGwgbm9kZXMuIGBGaWVsZEtpbmRgIGlzIHVzZWQgdG8gc2hvcnQtY2lyY3VpdCB0aGUgZmllbGQgbmFtZSB2YWxpZGF0aW9uIGxvZ2ljLAogICAgYWxsb3dpbmcgIm1ldGFkYXRhIiBmb3IgdGhhdCBmaWVsZC4KICAgIC0gYE5vZGVBdHRyaWJ1dGVgOiBUaGUgZmllbGQgaXMgYSBub2RlIGF0dHJpYnV0ZS4gVGhlc2UgYXJlIGZpZWxkcyB3aGljaCBhcmUgbm90IGlucHV0cyBvciBvdXRwdXRzLAogICAgYnV0IHdoaWNoIGFyZSB1c2VkIHRvIHN0b3JlIGluZm9ybWF0aW9uIGFib3V0IHRoZSBub2RlLiBGb3IgZXhhbXBsZSwgdGhlIGBpZGAgYW5kIGB0eXBlYCBmaWVsZHMgYXJlIG5vZGUKICAgIGF0dHJpYnV0ZXMuCgogICAgVGhlIHByZXNlbmNlIG9mIHRoaXMgaW4gYGpzb25fc2NoZW1hX2V4dHJhWyJmaWVsZF9raW5kIl1gIGlzIHVzZWQgd2hlbiBpbml0aWFsaXppbmcgbm9kZSBzY2hlbWFzIG9uIGFwcAogICAgc3RhcnR1cCwgYW5kIHdoZW4gZ2VuZXJhdGluZyB0aGUgT3BlbkFQSSBzY2hlbWEgZm9yIHRoZSB3b3JrZmxvdyBlZGl0b3IuCiAgICAiIiIKCiAgICBJbnB1dCA9ICJpbnB1dCIKICAgIE91dHB1dCA9ICJvdXRwdXQiCiAgICBJbnRlcm5hbCA9ICJpbnRlcm5hbCIKICAgIE5vZGVBdHRyaWJ1dGUgPSAibm9kZV9hdHRyaWJ1dGUiCgoKY2xhc3MgSW5wdXRGaWVsZEpTT05TY2hlbWFFeHRyYShCYXNlTW9kZWwpOgogICAgIiIiCiAgICBFeHRyYSBhdHRyaWJ1dGVzIHRvIGJlIGFkZGVkIHRvIGlucHV0IGZpZWxkcyBhbmQgdGhlaXIgT3BlbkFQSSBzY2hlbWEuIFVzZWQgZHVyaW5nIGdyYXBoIGV4ZWN1dGlvbiwKICAgIGFuZCBieSB0aGUgd29ya2Zsb3cgZWRpdG9yIGR1cmluZyBzY2hlbWEgcGFyc2luZyBhbmQgVUkgcmVuZGVyaW5nLgogICAgIiIiCgogICAgaW5wdXQ6IElucHV0CiAgICBmaWVsZF9raW5kOiBGaWVsZEtpbmQKICAgIG9yaWdfcmVxdWlyZWQ6IGJvb2wgPSBUcnVlCiAgICBkZWZhdWx0OiBPcHRpb25hbFtBbnldID0gTm9uZQogICAgb3JpZ19kZWZhdWx0OiBPcHRpb25hbFtBbnldID0gTm9uZQogICAgdWlfaGlkZGVuOiBib29sID0gRmFsc2UKICAgIHVpX3R5cGU6IE9wdGlvbmFsW1VJVHlwZV0gPSBOb25lCiAgICB1aV9jb21wb25lbnQ6IE9wdGlvbmFsW1VJQ29tcG9uZW50XSA9IE5vbmUKICAgIHVpX29yZGVyOiBPcHRpb25hbFtpbnRdID0gTm9uZQogICAgdWlfY2hvaWNlX2xhYmVsczogT3B0aW9uYWxbZGljdFtzdHIsIHN0cl1dID0gTm9uZQogICAgdWlfbW9kZWxfYmFzZTogT3B0aW9uYWxbbGlzdFtCYXNlTW9kZWxUeXBlXV0gPSBOb25lCiAgICB1aV9tb2RlbF90eXBlOiBPcHRpb25hbFtsaXN0W01vZGVsVHlwZV1dID0gTm9uZQogICAgdWlfbW9kZWxfdmFyaWFudDogT3B0aW9uYWxbbGlzdFtDbGlwVmFyaWFudFR5cGUgfCBNb2RlbFZhcmlhbnRUeXBlXV0gPSBOb25lCiAgICB1aV9tb2RlbF9mb3JtYXQ6IE9wdGlvbmFsW2xpc3RbTW9kZWxGb3JtYXRdXSA9IE5vbmUKCiAgICBtb2RlbF9jb25maWcgPSBDb25maWdEaWN0KAogICAgICAgIHZhbGlkYXRlX2Fzc2lnbm1lbnQ9VHJ1ZSwKICAgICAgICBqc29uX3NjaGVtYV9zZXJpYWxpemF0aW9uX2RlZmF1bHRzX3JlcXVpcmVkPVRydWUsCiAgICAgICAgdXNlX2VudW1fdmFsdWVzPVRydWUsCiAgICApCgoKY2xhc3MgV2l0aE1ldGFkYXRhKEJhc2VNb2RlbCk6CiAgICAiIiIKICAgIEluaGVyaXQgZnJvbSB0aGlzIGNsYXNzIGlmIHlvdXIgbm9kZSBuZWVkcyBhIG1ldGFkYXRhIGlucHV0IGZpZWxkLgogICAgIiIiCgogICAgbWV0YWRhdGE6IE9wdGlvbmFsW01ldGFkYXRhRmllbGRdID0gRmllbGQoCiAgICAgICAgZGVmYXVsdD1Ob25lLAogICAgICAgIGRlc2NyaXB0aW9uPUZpZWxkRGVzY3JpcHRpb25zLm1ldGFkYXRhLAogICAgICAgIGpzb25fc2NoZW1hX2V4dHJhPUlucHV0RmllbGRKU09OU2NoZW1hRXh0cmEoCiAgICAgICAgICAgIGZpZWxkX2tpbmQ9RmllbGRLaW5kLkludGVybmFsLAogICAgICAgICAgICBpbnB1dD1JbnB1dC5Db25uZWN0aW9uLAogICAgICAgICAgICBvcmlnX3JlcXVpcmVkPUZhbHNlLAogICAgICAgICkubW9kZWxfZHVtcChleGNsdWRlX25vbmU9VHJ1ZSksCiAgICApCgoKY2xhc3MgV2l0aFdvcmtmbG93OgogICAgd29ya2Zsb3cgPSBOb25lCgogICAgZGVmIF9faW5pdF9zdWJjbGFzc19fKGNscykgLT4gTm9uZToKICAgICAgICBsb2dnZXIud2FybmluZygKICAgICAgICAgICAgZiJ7Y2xzLl9fbW9kdWxlX18uc3BsaXQoJy4nKVswXX0ue2Nscy5fX25hbWVfX306IFdpdGhXb3JrZmxvdyBpcyBkZXByZWNhdGVkLiBVc2UgYGNvbnRleHQud29ya2Zsb3dgIHRvIGFjY2VzcyB0aGUgd29ya2Zsb3cuIgogICAgICAgICkKICAgICAgICBzdXBlcigpLl9faW5pdF9zdWJjbGFzc19fKCkKCgpjbGFzcyBXaXRoQm9hcmQoQmFzZU1vZGVsKToKICAgICIiIgogICAgSW5oZXJpdCBmcm9tIHRoaXMgY2xhc3MgaWYgeW91ciBub2RlIG5lZWRzIGEgYm9hcmQgaW5wdXQgZmllbGQuCiAgICAiIiIKCiAgICBib2FyZDogT3B0aW9uYWxbQm9hcmRGaWVsZF0gPSBGaWVsZCgKICAgICAgICBkZWZhdWx0PU5vbmUsCiAgICAgICAgZGVzY3JpcHRpb249RmllbGREZXNjcmlwdGlvbnMuYm9hcmQsCiAgICAgICAganNvbl9zY2hlbWFfZXh0cmE9SW5wdXRGaWVsZEpTT05TY2hlbWFFeHRyYSgKICAgICAgICAgICAgZmllbGRfa2luZD1GaWVsZEtpbmQuSW50ZXJuYWwsCiAgICAgICAgICAgIGlucHV0PUlucHV0LkRpcmVjdCwKICAgICAgICAgICAgb3JpZ19yZXF1aXJlZD1GYWxzZSwKICAgICAgICApLm1vZGVsX2R1bXAoZXhjbHVkZV9ub25lPVRydWUpLAogICAgKQoKCmNsYXNzIE91dHB1dEZpZWxkSlNPTlNjaGVtYUV4dHJhKEJhc2VNb2RlbCk6CiAgICAiIiIKICAgIEV4dHJhIGF0dHJpYnV0ZXMgdG8gYmUgYWRkZWQgdG8gaW5wdXQgZmllbGRzIGFuZCB0aGVpciBPcGVuQVBJIHNjaGVtYS4gVXNlZCBieSB0aGUgd29ya2Zsb3cgZWRpdG9yCiAgICBkdXJpbmcgc2NoZW1hIHBhcnNpbmcgYW5kIFVJIHJlbmRlcmluZy4KICAgICIiIgoKICAgIGZpZWxkX2tpbmQ6IEZpZWxkS2luZAogICAgdWlfaGlkZGVuOiBib29sID0gRmFsc2UKICAgIHVpX29yZGVyOiBPcHRpb25hbFtpbnRdID0gTm9uZQogICAgdWlfdHlwZTogT3B0aW9uYWxbVUlUeXBlXSA9IE5vbmUKCiAgICBtb2RlbF9jb25maWcgPSBDb25maWdEaWN0KAogICAgICAgIHZhbGlkYXRlX2Fzc2lnbm1lbnQ9VHJ1ZSwKICAgICAgICBqc29uX3NjaGVtYV9zZXJpYWxpemF0aW9uX2RlZmF1bHRzX3JlcXVpcmVkPVRydWUsCiAgICAgICAgdXNlX2VudW1fdmFsdWVzPVRydWUsCiAgICApCgoKZGVmIG1pZ3JhdGVfbW9kZWxfdWlfdHlwZSh1aV90eXBlOiBVSVR5cGUgfCBzdHIsIGpzb25fc2NoZW1hX2V4dHJhOiBkaWN0W3N0ciwgQW55XSkgLT4gYm9vbDoKICAgICIiIk1pZ3JhdGUgZGVwcmVjYXRlZCBtb2RlbC1zcGVjaWZpZXIgdWlfdHlwZSB2YWx1ZXMgdG8gbmV3LXN0eWxlIHVpX21vZGVsX1tiYXNlfHR5cGV8dmFyaWFudHxmb3JtYXRdIGluIGpzb25fc2NoZW1hX2V4dHJhLiIiIgogICAgaWYgbm90IGlzaW5zdGFuY2UodWlfdHlwZSwgVUlUeXBlKToKICAgICAgICB1aV90eXBlID0gVUlUeXBlKHVpX3R5cGUpCgogICAgdWlfbW9kZWxfdHlwZTogbGlzdFtNb2RlbFR5cGVdIHwgTm9uZSA9IE5vbmUKICAgIHVpX21vZGVsX2Jhc2U6IGxpc3RbQmFzZU1vZGVsVHlwZV0gfCBOb25lID0gTm9uZQogICAgdWlfbW9kZWxfZm9ybWF0OiBsaXN0W01vZGVsRm9ybWF0XSB8IE5vbmUgPSBOb25lCiAgICB1aV9tb2RlbF92YXJpYW50OiBsaXN0W0NsaXBWYXJpYW50VHlwZSB8IE1vZGVsVmFyaWFudFR5cGVdIHwgTm9uZSA9IE5vbmUKCiAgICBtYXRjaCB1aV90eXBlOgogICAgICAgIGNhc2UgVUlUeXBlLk1haW5Nb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfYmFzZSA9IFtCYXNlTW9kZWxUeXBlLlN0YWJsZURpZmZ1c2lvbjEsIEJhc2VNb2RlbFR5cGUuU3RhYmxlRGlmZnVzaW9uMl0KICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuTWFpbl0KICAgICAgICBjYXNlIFVJVHlwZS5Db2dWaWV3NE1haW5Nb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfYmFzZSA9IFtCYXNlTW9kZWxUeXBlLkNvZ1ZpZXc0XQogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5NYWluXQogICAgICAgIGNhc2UgVUlUeXBlLkZsdXhNYWluTW9kZWw6CiAgICAgICAgICAgIHVpX21vZGVsX2Jhc2UgPSBbQmFzZU1vZGVsVHlwZS5GbHV4XQogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5NYWluXQogICAgICAgIGNhc2UgVUlUeXBlLlNEM01haW5Nb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfYmFzZSA9IFtCYXNlTW9kZWxUeXBlLlN0YWJsZURpZmZ1c2lvbjNdCiAgICAgICAgICAgIHVpX21vZGVsX3R5cGUgPSBbTW9kZWxUeXBlLk1haW5dCiAgICAgICAgY2FzZSBVSVR5cGUuU0RYTE1haW5Nb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfYmFzZSA9IFtCYXNlTW9kZWxUeXBlLlN0YWJsZURpZmZ1c2lvblhMXQogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5NYWluXQogICAgICAgIGNhc2UgVUlUeXBlLlNEWExSZWZpbmVyTW9kZWw6CiAgICAgICAgICAgIHVpX21vZGVsX2Jhc2UgPSBbQmFzZU1vZGVsVHlwZS5TdGFibGVEaWZmdXNpb25YTFJlZmluZXJdCiAgICAgICAgICAgIHVpX21vZGVsX3R5cGUgPSBbTW9kZWxUeXBlLk1haW5dCiAgICAgICAgY2FzZSBVSVR5cGUuVkFFTW9kZWw6CiAgICAgICAgICAgIHVpX21vZGVsX3R5cGUgPSBbTW9kZWxUeXBlLlZBRV0KICAgICAgICBjYXNlIFVJVHlwZS5GbHV4VkFFTW9kZWw6CiAgICAgICAgICAgIHVpX21vZGVsX2Jhc2UgPSBbQmFzZU1vZGVsVHlwZS5GbHV4XQogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5WQUVdCiAgICAgICAgY2FzZSBVSVR5cGUuTG9SQU1vZGVsOgogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5Mb1JBXQogICAgICAgIGNhc2UgVUlUeXBlLkNvbnRyb2xOZXRNb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuQ29udHJvbE5ldF0KICAgICAgICBjYXNlIFVJVHlwZS5JUEFkYXB0ZXJNb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuSVBBZGFwdGVyXQogICAgICAgIGNhc2UgVUlUeXBlLlQySUFkYXB0ZXJNb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuVDJJQWRhcHRlcl0KICAgICAgICBjYXNlIFVJVHlwZS5UNUVuY29kZXJNb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuVDVFbmNvZGVyXQogICAgICAgIGNhc2UgVUlUeXBlLkNMSVBFbWJlZE1vZGVsOgogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5DTElQRW1iZWRdCiAgICAgICAgY2FzZSBVSVR5cGUuQ0xJUExFbWJlZE1vZGVsOgogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5DTElQRW1iZWRdCiAgICAgICAgICAgIHVpX21vZGVsX3ZhcmlhbnQgPSBbQ2xpcFZhcmlhbnRUeXBlLkxdCiAgICAgICAgY2FzZSBVSVR5cGUuQ0xJUEdFbWJlZE1vZGVsOgogICAgICAgICAgICB1aV9tb2RlbF90eXBlID0gW01vZGVsVHlwZS5DTElQRW1iZWRdCiAgICAgICAgICAgIHVpX21vZGVsX3ZhcmlhbnQgPSBbQ2xpcFZhcmlhbnRUeXBlLkddCiAgICAgICAgY2FzZSBVSVR5cGUuU3BhbmRyZWxJbWFnZVRvSW1hZ2VNb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuU3BhbmRyZWxJbWFnZVRvSW1hZ2VdCiAgICAgICAgY2FzZSBVSVR5cGUuQ29udHJvbExvUkFNb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuQ29udHJvbExvUmFdCiAgICAgICAgY2FzZSBVSVR5cGUuU2lnTGlwTW9kZWw6CiAgICAgICAgICAgIHVpX21vZGVsX3R5cGUgPSBbTW9kZWxUeXBlLlNpZ0xJUF0KICAgICAgICBjYXNlIFVJVHlwZS5GbHV4UmVkdXhNb2RlbDoKICAgICAgICAgICAgdWlfbW9kZWxfdHlwZSA9IFtNb2RlbFR5cGUuRmx1eFJlZHV4XQogICAgICAgIGNhc2UgVUlUeXBlLkxsYXZhT25ldmlzaW9uTW9kZWw6CiAgICAgICAgICAgIHVpX21vZGVsX3R5cGUgPSBbTW9kZWxUeXBlLkxsYXZhT25ldmlzaW9uXQogICAgICAgIGNhc2UgXzoKICAgICAgICAgICAgcGFzcwoKICAgIGRpZF9taWdyYXRlID0gRmFsc2UKCiAgICBpZiB1aV9tb2RlbF90eXBlIGlzIG5vdCBOb25lOgogICAgICAgIGpzb25fc2NoZW1hX2V4dHJhWyJ1aV9tb2RlbF90eXBlIl0gPSBbbS52YWx1ZSBmb3IgbSBpbiB1aV9tb2RlbF90eXBlXQogICAgICAgIGRpZF9taWdyYXRlID0gVHJ1ZQogICAgaWYgdWlfbW9kZWxfYmFzZSBpcyBub3QgTm9uZToKICAgICAgICBqc29uX3NjaGVtYV9leHRyYVsidWlfbW9kZWxfYmFzZSJdID0gW20udmFsdWUgZm9yIG0gaW4gdWlfbW9kZWxfYmFzZV0KICAgICAgICBkaWRfbWlncmF0ZSA9IFRydWUKICAgIGlmIHVpX21vZGVsX2Zvcm1hdCBpcyBub3QgTm9uZToKICAgICAgICBqc29uX3NjaGVtYV9leHRyYVsidWlfbW9kZWxfZm9ybWF0Il0gPSBbbS52YWx1ZSBmb3IgbSBpbiB1aV9tb2RlbF9mb3JtYXRdCiAgICAgICAgZGlkX21pZ3JhdGUgPSBUcnVlCiAgICBpZiB1aV9tb2RlbF92YXJpYW50IGlzIG5vdCBOb25lOgogICAgICAgIGpzb25fc2NoZW1hX2V4dHJhWyJ1aV9tb2RlbF92YXJpYW50Il0gPSBbbS52YWx1ZSBmb3IgbSBpbiB1aV9tb2RlbF92YXJpYW50XQogICAgICAgIGRpZF9taWdyYXRlID0gVHJ1ZQoKICAgIHJldHVybiBkaWRfbWlncmF0ZQoKCmRlZiBJbnB1dEZpZWxkKAogICAgIyBjb3BpZWQgZnJvbSBweWRhbnRpYydzIEZpZWxkCiAgICAjIFRPRE86IENhbiB3ZSBzdXBwb3J0IGRlZmF1bHRfZmFjdG9yeT8KICAgIGRlZmF1bHQ6IEFueSA9IF9VbnNldCwKICAgIGRlZmF1bHRfZmFjdG9yeTogQ2FsbGFibGVbW10sIEFueV0gfCBOb25lID0gX1Vuc2V0LAogICAgdGl0bGU6IHN0ciB8IE5vbmUgPSBfVW5zZXQsCiAgICBkZXNjcmlwdGlvbjogc3RyIHwgTm9uZSA9IF9VbnNldCwKICAgIHBhdHRlcm46IHN0ciB8IE5vbmUgPSBfVW5zZXQsCiAgICBzdHJpY3Q6IGJvb2wgfCBOb25lID0gX1Vuc2V0LAogICAgZ3Q6IGZsb2F0IHwgTm9uZSA9IF9VbnNldCwKICAgIGdlOiBmbG9hdCB8IE5vbmUgPSBfVW5zZXQsCiAgICBsdDogZmxvYXQgfCBOb25lID0gX1Vuc2V0LAogICAgbGU6IGZsb2F0IHwgTm9uZSA9IF9VbnNldCwKICAgIG11bHRpcGxlX29mOiBmbG9hdCB8IE5vbmUgPSBfVW5zZXQsCiAgICBhbGxvd19pbmZfbmFuOiBib29sIHwgTm9uZSA9IF9VbnNldCwKICAgIG1heF9kaWdpdHM6IGludCB8IE5vbmUgPSBfVW5zZXQsCiAgICBkZWNpbWFsX3BsYWNlczogaW50IHwgTm9uZSA9IF9VbnNldCwKICAgIG1pbl9sZW5ndGg6IGludCB8IE5vbmUgPSBfVW5zZXQsCiAgICBtYXhfbGVuZ3RoOiBpbnQgfCBOb25lID0gX1Vuc2V0LAogICAgIyBjdXN0b20KICAgIGlucHV0OiBJbnB1dCA9IElucHV0LkFueSwKICAgIHVpX3R5cGU6IE9wdGlvbmFsW1VJVHlwZV0gPSBOb25lLAogICAgdWlfY29tcG9uZW50OiBPcHRpb25hbFtVSUNvbXBvbmVudF0gPSBOb25lLAogICAgdWlfaGlkZGVuOiBPcHRpb25hbFtib29sXSA9IE5vbmUsCiAgICB1aV9vcmRlcjogT3B0aW9uYWxbaW50XSA9IE5vbmUsCiAgICB1aV9jaG9pY2VfbGFiZWxzOiBPcHRpb25hbFtkaWN0W3N0ciwgc3RyXV0gPSBOb25lLAogICAgdWlfbW9kZWxfYmFzZTogT3B0aW9uYWxbQmFzZU1vZGVsVHlwZSB8IGxpc3RbQmFzZU1vZGVsVHlwZV1dID0gTm9uZSwKICAgIHVpX21vZGVsX3R5cGU6IE9wdGlvbmFsW01vZGVsVHlwZSB8IGxpc3RbTW9kZWxUeXBlXV0gPSBOb25lLAogICAgdWlfbW9kZWxfdmFyaWFudDogT3B0aW9uYWxbQ2xpcFZhcmlhbnRUeXBlIHwgTW9kZWxWYXJpYW50VHlwZSB8IGxpc3RbQ2xpcFZhcmlhbnRUeXBlIHwgTW9kZWxWYXJpYW50VHlwZV1dID0gTm9uZSwKICAgIHVpX21vZGVsX2Zvcm1hdDogT3B0aW9uYWxbTW9kZWxGb3JtYXQgfCBsaXN0W01vZGVsRm9ybWF0XV0gPSBOb25lLAopIC0+IEFueToKICAgICIiIgogICAgQ3JlYXRlcyBhbiBpbnB1dCBmaWVsZCBmb3IgYW4gaW52b2NhdGlvbi4KCiAgICBUaGlzIGlzIGEgd3JhcHBlciBmb3IgUHlkYW50aWMncyBbRmllbGRdKGh0dHBzOi8vZG9jcy5weWRhbnRpYy5kZXYvbGF0ZXN0L2FwaS9maWVsZHMvI3B5ZGFudGljLmZpZWxkcy5GaWVsZCkKICAgIHRoYXQgYWRkcyBhIGZldyBleHRyYSBwYXJhbWV0ZXJzIHRvIHN1cHBvcnQgZ3JhcGggZXhlY3V0aW9uIGFuZCB0aGUgbm9kZSBlZGl0b3IgVUkuCgogICAgSWYgdGhlIGZpZWxkIGlzIGEgYE1vZGVsSWRlbnRpZmllckZpZWxkYCwgdXNlIHRoZSBgdWlfbW9kZWxfW2Jhc2V8dHlwZXx2YXJpYW50fGZvcm1hdF1gIGFyZ3MgdG8gZmlsdGVyIHRoZSBtb2RlbCBsaXN0CiAgICBpbiB0aGUgV29ya2Zsb3cgRWRpdG9yLiBPdGhlcndpc2UsIHVzZSBgdWlfdHlwZWAgdG8gcHJvdmlkZSBleHRyYSB0eXBlIGhpbnRzIGZvciB0aGUgVUkuCgogICAgRG9uJ3QgdXNlIGJvdGggYHVpX3R5cGVgIGFuZCBgdWlfbW9kZWxfW2Jhc2V8dHlwZXx2YXJpYW50fGZvcm1hdF1gIC0gaWYgYm90aCBhcmUgcHJvdmlkZWQsIGEgd2FybmluZyB3aWxsIGJlCiAgICBsb2dnZWQgYW5kIGB1aV90eXBlYCB3aWxsIGJlIGlnbm9yZWQuCgogICAgQXJnczoKICAgICAgICBpbnB1dDogVGhlIGtpbmQgb2YgaW5wdXQgdGhpcyBmaWVsZCByZXF1aXJlcy4KICAgICAgICAtIGBJbnB1dC5EaXJlY3RgIG1lYW5zIGEgdmFsdWUgbXVzdCBiZSBwcm92aWRlZCBvbiBpbnN0YW50aWF0aW9uLgogICAgICAgIC0gYElucHV0LkNvbm5lY3Rpb25gIG1lYW5zIHRoZSB2YWx1ZSBtdXN0IGJlIHByb3ZpZGVkIGJ5IGEgY29ubmVjdGlvbi4KICAgICAgICAtIGBJbnB1dC5BbnlgIG1lYW5zIGVpdGhlciB3aWxsIGRvLgoKICAgICAgICB1aV90eXBlOiBPcHRpb25hbGx5IHByb3ZpZGVzIGFuIGV4dHJhIHR5cGUgaGludCBmb3IgdGhlIFVJLiBJbiBzb21lIHNpdHVhdGlvbnMsIHRoZSBmaWVsZCdzIHR5cGUgaXMgbm90IGVub3VnaAogICAgICAgIHRvIGluZmVyIHRoZSBjb3JyZWN0IFVJIHR5cGUuIEZvciBleGFtcGxlLCBTY2hlZHVsZXIgZmllbGRzIGFyZSBlbnVtcywgYnV0IHdlIHdhbnQgdG8gcmVuZGVyIGEgc3BlY2lhbCBzY2hlZHVsZXIKICAgICAgICBkcm9wZG93biBpbiB0aGUgVUkuIFVzZSBgVUlUeXBlLlNjaGVkdWxlcmAgdG8gaW5kaWNhdGUgdGhpcy4KCiAgICAgICAgdWlfY29tcG9uZW50OiBPcHRpb25hbGx5IHNwZWNpZmllcyBhIHNwZWNpZmljIGNvbXBvbmVudCB0byB1c2UgaW4gdGhlIFVJLiBUaGUgVUkgd2lsbCBhbHdheXMgcmVuZGVyIGEgc3VpdGFibGUKICAgICAgICBjb21wb25lbnQsIGJ1dCBzb21ldGltZXMgeW91IHdhbnQgc29tZXRoaW5nIGRpZmZlcmVudCB0aGFuIHRoZSBkZWZhdWx0LiBGb3IgZXhhbXBsZSwgYSBgc3RyaW5nYCBmaWVsZCB3aWxsCiAgICAgICAgZGVmYXVsdCB0byBhIHNpbmdsZS1saW5lIGlucHV0LCBidXQgeW91IG1heSB3YW50IGEgbXVsdGktbGluZSB0ZXh0YXJlYSBpbnN0ZWFkLiBJbiB0aGlzIGNhc2UsIHlvdSBjb3VsZCB1c2UKICAgICAgICBgVUlDb21wb25lbnQuVGV4dGFyZWFgLgoKICAgICAgICB1aV9oaWRkZW46IFNwZWNpZmllcyB3aGV0aGVyIG9yIG5vdCB0aGlzIGZpZWxkIHNob3VsZCBiZSBoaWRkZW4gaW4gdGhlIFVJLgoKICAgICAgICB1aV9vcmRlcjogU3BlY2lmaWVzIHRoZSBvcmRlciBpbiB3aGljaCB0aGlzIGZpZWxkIHNob3VsZCBiZSByZW5kZXJlZCBpbiB0aGUgVUkuIElmIG9taXR0ZWQsIHRoZSBmaWVsZCB3aWxsIGJlCiAgICAgICAgcmVuZGVyZWQgYWZ0ZXIgYWxsIGZpZWxkcyB3aXRoIGFuIGV4cGxpY2l0IG9yZGVyLCBpbiB0aGUgb3JkZXIgdGhleSBhcmUgZGVmaW5lZCBpbiB0aGUgSW52b2NhdGlvbiBjbGFzcy4KCiAgICAgICAgdWlfbW9kZWxfYmFzZTogU3BlY2lmaWVzIHRoZSBiYXNlIG1vZGVsIGFyY2hpdGVjdHVyZXMgdG8gZmlsdGVyIHRoZSBtb2RlbCBsaXN0IGJ5IGluIHRoZSBXb3JrZmxvdyBFZGl0b3IuIEZvcgogICAgICAgIGV4YW1wbGUsIGB1aV9tb2RlbF9iYXNlPUJhc2VNb2RlbFR5cGUuU3RhYmxlRGlmZnVzaW9uWExgIHdpbGwgc2hvdyBvbmx5IFNEWEwgYXJjaGl0ZWN0dXJlIG1vZGVscy4gVGhpcyBhcmcgaXMKICAgICAgICBvbmx5IHZhbGlkIGlmIHRoaXMgSW5wdXQgZmllbGQgaXMgYW5ub3RhdGVkIGFzIGEgYE1vZGVsSWRlbnRpZmllckZpZWxkYC4KCiAgICAgICAgdWlfbW9kZWxfdHlwZTogU3BlY2lmaWVzIHRoZSBtb2RlbCB0eXBlKHMpIHRvIGZpbHRlciB0aGUgbW9kZWwgbGlzdCBieSBpbiB0aGUgV29ya2Zsb3cgRWRpdG9yLiBGb3IgZXhhbXBsZSwKICAgICAgICBgdWlfbW9kZWxfdHlwZT1Nb2RlbFR5cGUuVkFFYCB3aWxsIHNob3cgb25seSBWQUUgbW9kZWxzLiBUaGlzIGFyZyBpcyBvbmx5IHZhbGlkIGlmIHRoaXMgSW5wdXQgZmllbGQgaXMKICAgICAgICBhbm5vdGF0ZWQgYXMgYSBgTW9kZWxJZGVudGlmaWVyRmllbGRgLgoKICAgICAgICB1aV9tb2RlbF92YXJpYW50OiBTcGVjaWZpZXMgdGhlIG1vZGVsIHZhcmlhbnQocykgdG8gZmlsdGVyIHRoZSBtb2RlbCBsaXN0IGJ5IGluIHRoZSBXb3JrZmxvdyBFZGl0b3IuIEZvciBleGFtcGxlLAogICAgICAgIGB1aV9tb2RlbF92YXJpYW50PU1vZGVsVmFyaWFudFR5cGUuSW5wYWludGluZ2Agd2lsbCBzaG93IG9ubHkgaW5wYWludGluZyBtb2RlbHMuIFRoaXMgYXJnIGlzIG9ubHkgdmFsaWQgaWYgdGhpcwogICAgICAgIElucHV0IGZpZWxkIGlzIGFubm90YXRlZCBhcyBhIGBNb2RlbElkZW50aWZpZXJGaWVsZGAuCgogICAgICAgIHVpX21vZGVsX2Zvcm1hdDogU3BlY2lmaWVzIHRoZSBtb2RlbCBmb3JtYXQocykgdG8gZmlsdGVyIHRoZSBtb2RlbCBsaXN0IGJ5IGluIHRoZSBXb3JrZmxvdyBFZGl0b3IuIEZvciBleGFtcGxlLAogICAgICAgIGB1aV9tb2RlbF9mb3JtYXQ9TW9kZWxGb3JtYXQuRGlmZnVzZXJzYCB3aWxsIHNob3cgb25seSBtb2RlbHMgaW4gdGhlIGRpZmZ1c2VycyBmb3JtYXQuIFRoaXMgYXJnIGlzIG9ubHkgdmFsaWQKICAgICAgICBpZiB0aGlzIElucHV0IGZpZWxkIGlzIGFubm90YXRlZCBhcyBhIGBNb2RlbElkZW50aWZpZXJGaWVsZGAuCgogICAgICAgIHVpX2Nob2ljZV9sYWJlbHM6IFNwZWNpZmllcyB0aGUgbGFiZWxzIHRvIHVzZSBmb3IgdGhlIGNob2ljZXMgaW4gYW4gZW51bSBmaWVsZC4gSWYgb21pdHRlZCwgdGhlIGVudW0gdmFsdWVzCiAgICAgICAgd2lsbCBiZSB1c2VkLiBUaGlzIGFyZyBpcyBvbmx5IHZhbGlkIGlmIHRoZSBmaWVsZCBpcyBhbm5vdGF0ZWQgd2l0aCBhcyBhIGBMaXRlcmFsYC4gRm9yIGV4YW1wbGUsCiAgICAgICAgYExpdGVyYWxbImNob2ljZTEiLCAiY2hvaWNlMiIsICJjaG9pY2UzIl1gIHdpdGggYHVpX2Nob2ljZV9sYWJlbHM9eyJjaG9pY2UxIjogIkNob2ljZSAxIiwgImNob2ljZTIiOiAiQ2hvaWNlIDIiLAogICAgICAgICJjaG9pY2UzIjogIkNob2ljZSAzIn1gIHdpbGwgcmVuZGVyIGEgZHJvcGRvd24gd2l0aCB0aGUgbGFiZWxzICJDaG9pY2UgMSIsICJDaG9pY2UgMiIgYW5kICJDaG9pY2UgMyIuCiAgICAiIiIKCiAgICBqc29uX3NjaGVtYV9leHRyYV8gPSBJbnB1dEZpZWxkSlNPTlNjaGVtYUV4dHJhKAogICAgICAgIGlucHV0PWlucHV0LAogICAgICAgIGZpZWxkX2tpbmQ9RmllbGRLaW5kLklucHV0LAogICAgKQoKICAgIGlmIHVpX2NvbXBvbmVudCBpcyBub3QgTm9uZToKICAgICAgICBqc29uX3NjaGVtYV9leHRyYV8udWlfY29tcG9uZW50ID0gdWlfY29tcG9uZW50CiAgICBpZiB1aV9oaWRkZW4gaXMgbm90IE5vbmU6CiAgICAgICAganNvbl9zY2hlbWFfZXh0cmFfLnVpX2hpZGRlbiA9IHVpX2hpZGRlbgogICAgaWYgdWlfb3JkZXIgaXMgbm90IE5vbmU6CiAgICAgICAganNvbl9zY2hlbWFfZXh0cmFfLnVpX29yZGVyID0gdWlfb3JkZXIKICAgIGlmIHVpX2Nob2ljZV9sYWJlbHMgaXMgbm90IE5vbmU6CiAgICAgICAganNvbl9zY2hlbWFfZXh0cmFfLnVpX2Nob2ljZV9sYWJlbHMgPSB1aV9jaG9pY2VfbGFiZWxzCiAgICBpZiB1aV9tb2RlbF9iYXNlIGlzIG5vdCBOb25lOgogICAgICAgIGlmIGlzaW5zdGFuY2UodWlfbW9kZWxfYmFzZSwgbGlzdCk6CiAgICAgICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy51aV9tb2RlbF9iYXNlID0gdWlfbW9kZWxfYmFzZQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy51aV9tb2RlbF9iYXNlID0gW3VpX21vZGVsX2Jhc2VdCiAgICBpZiB1aV9tb2RlbF90eXBlIGlzIG5vdCBOb25lOgogICAgICAgIGlmIGlzaW5zdGFuY2UodWlfbW9kZWxfdHlwZSwgbGlzdCk6CiAgICAgICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy51aV9tb2RlbF90eXBlID0gdWlfbW9kZWxfdHlwZQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy51aV9tb2RlbF90eXBlID0gW3VpX21vZGVsX3R5cGVdCiAgICBpZiB1aV9tb2RlbF92YXJpYW50IGlzIG5vdCBOb25lOgogICAgICAgIGlmIGlzaW5zdGFuY2UodWlfbW9kZWxfdmFyaWFudCwgbGlzdCk6CiAgICAgICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy51aV9tb2RlbF92YXJpYW50ID0gdWlfbW9kZWxfdmFyaWFudAogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy51aV9tb2RlbF92YXJpYW50ID0gW3VpX21vZGVsX3ZhcmlhbnRdCiAgICBpZiB1aV9tb2RlbF9mb3JtYXQgaXMgbm90IE5vbmU6CiAgICAgICAgaWYgaXNpbnN0YW5jZSh1aV9tb2RlbF9mb3JtYXQsIGxpc3QpOgogICAgICAgICAgICBqc29uX3NjaGVtYV9leHRyYV8udWlfbW9kZWxfZm9ybWF0ID0gdWlfbW9kZWxfZm9ybWF0CiAgICAgICAgZWxzZToKICAgICAgICAgICAganNvbl9zY2hlbWFfZXh0cmFfLnVpX21vZGVsX2Zvcm1hdCA9IFt1aV9tb2RlbF9mb3JtYXRdCiAgICBpZiB1aV90eXBlIGlzIG5vdCBOb25lOgogICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy51aV90eXBlID0gdWlfdHlwZQoKICAgICIiIgogICAgVGhlcmUgaXMgYSBjb25mbGljdCBiZXR3ZWVuIHRoZSB0eXBpbmcgb2YgaW52b2NhdGlvbiBkZWZpbml0aW9ucyBhbmQgdGhlIHR5cGluZyBvZiBhbiBpbnZvY2F0aW9uJ3MKICAgIGBpbnZva2UoKWAgZnVuY3Rpb24uCgogICAgT24gaW5zdGFudGlhdGlvbiBvZiBhIG5vZGUsIHRoZSBpbnZvY2F0aW9uIGRlZmluaXRpb24gaXMgdXNlZCB0byBjcmVhdGUgdGhlIHB5dGhvbiBjbGFzcy4gQXQgdGhpcyB0aW1lLAogICAgYW55IG51bWJlciBvZiBmaWVsZHMgbWF5IGJlIG9wdGlvbmFsLCBiZWNhdXNlIHRoZXkgbWF5IGJlIHByb3ZpZGVkIGJ5IGNvbm5lY3Rpb25zLgoKICAgIE9uIGNhbGxpbmcgb2YgYGludm9rZSgpYCwgaG93ZXZlciwgdGhvc2UgZmllbGRzIG1heSBiZSByZXF1aXJlZC4KCiAgICBGb3IgZXhhbXBsZSwgY29uc2lkZXIgYW4gUmVzaXplSW1hZ2VJbnZvY2F0aW9uIHdpdGggYW4gYGltYWdlOiBJbWFnZUZpZWxkYCBmaWVsZC4KCiAgICBgaW1hZ2VgIGlzIHJlcXVpcmVkIGR1cmluZyB0aGUgY2FsbCB0byBgaW52b2tlKClgLCBidXQgd2hlbiB0aGUgcHl0aG9uIGNsYXNzIGlzIGluc3RhbnRpYXRlZCwKICAgIHRoZSBmaWVsZCBtYXkgbm90IGJlIHByZXNlbnQuIFRoaXMgaXMgZmluZSwgYmVjYXVzZSB0aGF0IGltYWdlIGZpZWxkIHdpbGwgYmUgcHJvdmlkZWQgYnkgYQogICAgY29ubmVjdGlvbiBmcm9tIGFuIGFuY2VzdG9yIG5vZGUsIHdoaWNoIG91dHB1dHMgYW4gaW1hZ2UuCgogICAgVGhpcyBtZWFucyB3ZSB3YW50IHRvIHR5cGUgdGhlIGBpbWFnZWAgZmllbGQgYXMgb3B0aW9uYWwgZm9yIHRoZSBub2RlIGNsYXNzIGRlZmluaXRpb24sIGJ1dCByZXF1aXJlZAogICAgZm9yIHRoZSBgaW52b2tlKClgIGZ1bmN0aW9uLgoKICAgIElmIHdlIHVzZSBgdHlwaW5nLk9wdGlvbmFsYCBpbiB0aGUgbm9kZSBjbGFzcyBkZWZpbml0aW9uLCB0aGUgZmllbGQgd2lsbCBiZSB0eXBlZCBhcyBvcHRpb25hbCBpbiB0aGUKICAgIGBpbnZva2UoKWAgbWV0aG9kLCBhbmQgd2UnbGwgaGF2ZSB0byBkbyBhIGxvdCBvZiBydW50aW1lIGNoZWNrcyB0byBlbnN1cmUgdGhlIGZpZWxkIGlzIHByZXNlbnQgLSBvcgogICAgYW55IHN0YXRpYyB0eXBlIGFuYWx5c2lzIHRvb2xzIHdpbGwgY29tcGxhaW4uCgogICAgVG8gZ2V0IGFyb3VuZCB0aGlzLCBpbiBub2RlIGNsYXNzIGRlZmluaXRpb25zLCB3ZSB0eXBlIGFsbCBmaWVsZHMgY29ycmVjdGx5IGZvciB0aGUgYGludm9rZSgpYCBmdW5jdGlvbiwKICAgIGJ1dCBzZWNyZXRseSBtYWtlIHRoZW0gb3B0aW9uYWwgaW4gYElucHV0RmllbGQoKWAuIFdlIGFsc28gc3RvcmUgdGhlIG9yaWdpbmFsIHJlcXVpcmVkIGJvb2wgYW5kL29yIGRlZmF1bHQKICAgIHZhbHVlLiBXaGVuIHdlIGNhbGwgYGludm9rZSgpYCwgd2UgdXNlIHRoaXMgc3RvcmVkIGluZm9ybWF0aW9uIHRvIGRvIGFuIGFkZGl0aW9uYWwgY2hlY2sgb24gdGhlIGNsYXNzLgogICAgIiIiCgogICAgaWYgZGVmYXVsdF9mYWN0b3J5IGlzIG5vdCBfVW5zZXQgYW5kIGRlZmF1bHRfZmFjdG9yeSBpcyBub3QgTm9uZToKICAgICAgICBkZWZhdWx0ID0gZGVmYXVsdF9mYWN0b3J5KCkKICAgICAgICBsb2dnZXIud2FybmluZygnImRlZmF1bHRfZmFjdG9yeSIgaXMgbm90IHN1cHBvcnRlZCwgY2FsbGluZyBpdCBub3cgdG8gc2V0ICJkZWZhdWx0IicpCgogICAgIyBUaGVzZSBhcmUgdGhlIGFyZ3Mgd2UgbWF5IHdpc2ggcGFzcyB0byB0aGUgcHlkYW50aWMgYEZpZWxkKClgIGZ1bmN0aW9uCiAgICBmaWVsZF9hcmdzID0gewogICAgICAgICJkZWZhdWx0IjogZGVmYXVsdCwKICAgICAgICAidGl0bGUiOiB0aXRsZSwKICAgICAgICAiZGVzY3JpcHRpb24iOiBkZXNjcmlwdGlvbiwKICAgICAgICAicGF0dGVybiI6IHBhdHRlcm4sCiAgICAgICAgInN0cmljdCI6IHN0cmljdCwKICAgICAgICAiZ3QiOiBndCwKICAgICAgICAiZ2UiOiBnZSwKICAgICAgICAibHQiOiBsdCwKICAgICAgICAibGUiOiBsZSwKICAgICAgICAibXVsdGlwbGVfb2YiOiBtdWx0aXBsZV9vZiwKICAgICAgICAiYWxsb3dfaW5mX25hbiI6IGFsbG93X2luZl9uYW4sCiAgICAgICAgIm1heF9kaWdpdHMiOiBtYXhfZGlnaXRzLAogICAgICAgICJkZWNpbWFsX3BsYWNlcyI6IGRlY2ltYWxfcGxhY2VzLAogICAgICAgICJtaW5fbGVuZ3RoIjogbWluX2xlbmd0aCwKICAgICAgICAibWF4X2xlbmd0aCI6IG1heF9sZW5ndGgsCiAgICB9CgogICAgIyBXZSBvbmx5IHdhbnQgdG8gcGFzcyB0aGUgYXJncyB0aGF0IHdlcmUgcHJvdmlkZWQsIG90aGVyd2lzZSB0aGUgYEZpZWxkKClgYCBmdW5jdGlvbiB3b24ndCB3b3JrIGFzIGV4cGVjdGVkCiAgICBwcm92aWRlZF9hcmdzID0ge2s6IHYgZm9yIChrLCB2KSBpbiBmaWVsZF9hcmdzLml0ZW1zKCkgaWYgdiBpcyBub3QgUHlkYW50aWNVbmRlZmluZWR9CgogICAgIyBCZWNhdXNlIHdlIGFyZSBtYW51YWxseSBtYWtpbmcgZmllbGRzIG9wdGlvbmFsLCB3ZSBuZWVkIHRvIHN0b3JlIHRoZSBvcmlnaW5hbCByZXF1aXJlZCBib29sIGZvciByZWZlcmVuY2UgbGF0ZXIKICAgIGpzb25fc2NoZW1hX2V4dHJhXy5vcmlnX3JlcXVpcmVkID0gZGVmYXVsdCBpcyBQeWRhbnRpY1VuZGVmaW5lZAoKICAgICMgTWFrZSBJbnB1dC5BbnkgYW5kIElucHV0LkNvbm5lY3Rpb24gZmllbGRzIG9wdGlvbmFsLCBwcm92aWRpbmcgTm9uZSBhcyBhIGRlZmF1bHQgaWYgdGhlIGZpZWxkIGRvZXNuJ3QgYWxyZWFkeSBoYXZlIG9uZQogICAgaWYgaW5wdXQgaXMgSW5wdXQuQW55IG9yIGlucHV0IGlzIElucHV0LkNvbm5lY3Rpb246CiAgICAgICAgZGVmYXVsdF8gPSBOb25lIGlmIGRlZmF1bHQgaXMgUHlkYW50aWNVbmRlZmluZWQgZWxzZSBkZWZhdWx0CiAgICAgICAgcHJvdmlkZWRfYXJncy51cGRhdGUoeyJkZWZhdWx0IjogZGVmYXVsdF99KQogICAgICAgIGlmIGRlZmF1bHQgaXMgbm90IFB5ZGFudGljVW5kZWZpbmVkOgogICAgICAgICAgICAjIEJlZm9yZSBpbnZva2luZywgd2UnbGwgY2hlY2sgZm9yIHRoZSBvcmlnaW5hbCBkZWZhdWx0IHZhbHVlIGFuZCBzZXQgaXQgb24gdGhlIGZpZWxkIGlmIHRoZSBmaWVsZCBoYXMgbm8gdmFsdWUKICAgICAgICAgICAganNvbl9zY2hlbWFfZXh0cmFfLmRlZmF1bHQgPSBkZWZhdWx0CiAgICAgICAgICAgIGpzb25fc2NoZW1hX2V4dHJhXy5vcmlnX2RlZmF1bHQgPSBkZWZhdWx0CiAgICBlbGlmIGRlZmF1bHQgaXMgbm90IFB5ZGFudGljVW5kZWZpbmVkOgogICAgICAgIGRlZmF1bHRfID0gZGVmYXVsdAogICAgICAgIHByb3ZpZGVkX2FyZ3MudXBkYXRlKHsiZGVmYXVsdCI6IGRlZmF1bHRffSkKICAgICAgICBqc29uX3NjaGVtYV9leHRyYV8ub3JpZ19kZWZhdWx0ID0gZGVmYXVsdF8KCiAgICByZXR1cm4gRmllbGQoCiAgICAgICAgKipwcm92aWRlZF9hcmdzLAogICAgICAgIGpzb25fc2NoZW1hX2V4dHJhPWpzb25fc2NoZW1hX2V4dHJhXy5tb2RlbF9kdW1wKGV4Y2x1ZGVfdW5zZXQ9VHJ1ZSksCiAgICApCgoKZGVmIE91dHB1dEZpZWxkKAogICAgIyBjb3BpZWQgZnJvbSBweWRhbnRpYydzIEZpZWxkCiAgICBkZWZhdWx0OiBBbnkgPSBfVW5zZXQsCiAgICB0aXRsZTogc3RyIHwgTm9uZSA9IF9VbnNldCwKICAgIGRlc2NyaXB0aW9uOiBzdHIgfCBOb25lID0gX1Vuc2V0LAogICAgcGF0dGVybjogc3RyIHwgTm9uZSA9IF9VbnNldCwKICAgIHN0cmljdDogYm9vbCB8IE5vbmUgPSBfVW5zZXQsCiAgICBndDogZmxvYXQgfCBOb25lID0gX1Vuc2V0LAogICAgZ2U6IGZsb2F0IHwgTm9uZSA9IF9VbnNldCwKICAgIGx0OiBmbG9hdCB8IE5vbmUgPSBfVW5zZXQsCiAgICBsZTogZmxvYXQgfCBOb25lID0gX1Vuc2V0LAogICAgbXVsdGlwbGVfb2Y6IGZsb2F0IHwgTm9uZSA9IF9VbnNldCwKICAgIGFsbG93X2luZl9uYW46IGJvb2wgfCBOb25lID0gX1Vuc2V0LAogICAgbWF4X2RpZ2l0czogaW50IHwgTm9uZSA9IF9VbnNldCwKICAgIGRlY2ltYWxfcGxhY2VzOiBpbnQgfCBOb25lID0gX1Vuc2V0LAogICAgbWluX2xlbmd0aDogaW50IHwgTm9uZSA9IF9VbnNldCwKICAgIG1heF9sZW5ndGg6IGludCB8IE5vbmUgPSBfVW5zZXQsCiAgICAjIGN1c3RvbQogICAgdWlfdHlwZTogT3B0aW9uYWxbVUlUeXBlXSA9IE5vbmUsCiAgICB1aV9oaWRkZW46IGJvb2wgPSBGYWxzZSwKICAgIHVpX29yZGVyOiBPcHRpb25hbFtpbnRdID0gTm9uZSwKKSAtPiBBbnk6CiAgICAiIiIKICAgIENyZWF0ZXMgYW4gb3V0cHV0IGZpZWxkIGZvciBhbiBpbnZvY2F0aW9uIG91dHB1dC4KCiAgICBUaGlzIGlzIGEgd3JhcHBlciBmb3IgUHlkYW50aWMncyBbRmllbGRdKGh0dHBzOi8vZG9jcy5weWRhbnRpYy5kZXYvMS4xMC91c2FnZS9zY2hlbWEvI2ZpZWxkLWN1c3RvbWl6YXRpb24pCiAgICB0aGF0IGFkZHMgYSBmZXcgZXh0cmEgcGFyYW1ldGVycyB0byBzdXBwb3J0IGdyYXBoIGV4ZWN1dGlvbiBhbmQgdGhlIG5vZGUgZWRpdG9yIFVJLgoKICAgIEFyZ3M6CiAgICAgICAgdWlfdHlwZTogT3B0aW9uYWxseSBwcm92aWRlcyBhbiBleHRyYSB0eXBlIGhpbnQgZm9yIHRoZSBVSS4gSW4gc29tZSBzaXR1YXRpb25zLCB0aGUgZmllbGQncyB0eXBlIGlzIG5vdCBlbm91Z2gKICAgICAgICB0byBpbmZlciB0aGUgY29ycmVjdCBVSSB0eXBlLiBGb3IgZXhhbXBsZSwgU2NoZWR1bGVyIGZpZWxkcyBhcmUgZW51bXMsIGJ1dCB3ZSB3YW50IHRvIHJlbmRlciBhIHNwZWNpYWwgc2NoZWR1bGVyCiAgICAgICAgZHJvcGRvd24gaW4gdGhlIFVJLiBVc2UgYFVJVHlwZS5TY2hlZHVsZXJgIHRvIGluZGljYXRlIHRoaXMuCgogICAgICAgIHVpX2hpZGRlbjogU3BlY2lmaWVzIHdoZXRoZXIgb3Igbm90IHRoaXMgZmllbGQgc2hvdWxkIGJlIGhpZGRlbiBpbiB0aGUgVUkuCgogICAgICAgIHVpX29yZGVyOiBTcGVjaWZpZXMgdGhlIG9yZGVyIGluIHdoaWNoIHRoaXMgZmllbGQgc2hvdWxkIGJlIHJlbmRlcmVkIGluIHRoZSBVSS4gSWYgb21pdHRlZCwgdGhlIGZpZWxkIHdpbGwgYmUKICAgICAgICByZW5kZXJlZCBhZnRlciBhbGwgZmllbGRzIHdpdGggYW4gZXhwbGljaXQgb3JkZXIsIGluIHRoZSBvcmRlciB0aGV5IGFyZSBkZWZpbmVkIGluIHRoZSBJbnZvY2F0aW9uIGNsYXNzLgogICAgIiIiCgogICAgcmV0dXJuIEZpZWxkKAogICAgICAgIGRlZmF1bHQ9ZGVmYXVsdCwKICAgICAgICB0aXRsZT10aXRsZSwKICAgICAgICBkZXNjcmlwdGlvbj1kZXNjcmlwdGlvbiwKICAgICAgICBwYXR0ZXJuPXBhdHRlcm4sCiAgICAgICAgc3RyaWN0PXN0cmljdCwKICAgICAgICBndD1ndCwKICAgICAgICBnZT1nZSwKICAgICAgICBsdD1sdCwKICAgICAgICBsZT1sZSwKICAgICAgICBtdWx0aXBsZV9vZj1tdWx0aXBsZV9vZiwKICAgICAgICBhbGxvd19pbmZfbmFuPWFsbG93X2luZl9uYW4sCiAgICAgICAgbWF4X2RpZ2l0cz1tYXhfZGlnaXRzLAogICAgICAgIGRlY2ltYWxfcGxhY2VzPWRlY2ltYWxfcGxhY2VzLAogICAgICAgIG1pbl9sZW5ndGg9bWluX2xlbmd0aCwKICAgICAgICBtYXhfbGVuZ3RoPW1heF9sZW5ndGgsCiAgICAgICAganNvbl9zY2hlbWFfZXh0cmE9T3V0cHV0RmllbGRKU09OU2NoZW1hRXh0cmEoCiAgICAgICAgICAgIHVpX2hpZGRlbj11aV9oaWRkZW4sCiAgICAgICAgICAgIHVpX29yZGVyPXVpX29yZGVyLAogICAgICAgICAgICB1aV90eXBlPXVpX3R5cGUsCiAgICAgICAgICAgIGZpZWxkX2tpbmQ9RmllbGRLaW5kLk91dHB1dCwKICAgICAgICApLm1vZGVsX2R1bXAoZXhjbHVkZV9ub25lPVRydWUpLAogICAgKQo="
    content = base64.b64decode(content_b64).decode('utf-8')
    write_file(r"d:\Cat_InvokeAI\invokeai\invokeai\app\invocations\fields.py", content)

if __name__ == "__main__":

    print("Starting repair...")
    repair_invocation_context()
    repair_invocation_services()
    repair_dependencies()
    print("Repair complete. Please verify with check_update.py or start the app.")
