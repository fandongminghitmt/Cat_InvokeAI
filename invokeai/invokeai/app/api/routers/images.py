
import io
import json
import traceback
from typing import ClassVar, Optional

from fastapi import BackgroundTasks, Body, HTTPException, Path, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.extract_metadata_from_image import extract_metadata_from_image
from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageNamesResult,
    ImageRecordChanges,
    ResourceOrigin,
)
from invokeai.app.services.images.images_common import (
    DeleteImagesResult,
    ImageDTO,
    ImageUrlsDTO,
    StarredImagesResult,
    UnstarredImagesResult,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.util.controlnet_utils import heuristic_resize_fast
from invokeai.backend.image_util.util import np_to_pil, pil_to_np

images_router = APIRouter(prefix="/v1/images", tags=["images"])


# images are immutable; set a high max-age
IMAGE_MAX_AGE = 31536000


class ResizeToDimensions(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    MAX_SIZE: ClassVar[int] = 4096 * 4096

    @model_validator(mode="after")
    def validate_total_output_size(self):
        if self.width * self.height > self.MAX_SIZE:
            raise ValueError(f"Max total output size for resizing is {self.MAX_SIZE} pixels")
        return self



def convert_video_to_image_dto(video_dto: VideoDTO) -> ImageDTO:
    base_url = "api/v1/media/video"
    return ImageDTO(
        image_name=video_dto.video_name,
        image_origin=video_dto.video_origin,
        image_category="general", 
        image_url=f"{base_url}/{video_dto.video_name}/full",
        thumbnail_url=f"{base_url}/{video_dto.video_name}/thumbnail",
        width=video_dto.width,
        height=video_dto.height,
        created_at=str(video_dto.created_at),
        updated_at=str(video_dto.updated_at),
        deleted_at=str(video_dto.deleted_at) if video_dto.deleted_at else None,
        is_intermediate=video_dto.is_intermediate,
        session_id=video_dto.session_id,
        node_id=video_dto.node_id,
        starred=video_dto.starred,
        has_workflow=video_dto.has_workflow,
        board_id=None
    )

    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(status_code=415, detail="Not an image")

    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=415, detail="Failed to read image")

    if crop_visible:
        try:
            bbox = pil_image.getbbox()
            pil_image = pil_image.crop(bbox)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to crop image")

    if resize_to:
        try:
            dims = json.loads(resize_to)
            resize_dims = ResizeToDimensions(**dims)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid resize_to format or size")

        try:
            # heuristic_resize_fast expects an RGB or RGBA image
            pil_rgba = pil_image.convert("RGBA")
            np_image = pil_to_np(pil_rgba)
            np_image = heuristic_resize_fast(np_image, (resize_dims.width, resize_dims.height))
            pil_image = np_to_pil(np_image)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to resize image")

    extracted_metadata = extract_metadata_from_image(
        pil_image=pil_image,
        invokeai_metadata_override=metadata,
        invokeai_workflow_override=None,
        invokeai_graph_override=None,
        logger=ApiDependencies.invoker.services.logger,
    )

    try:
        image_dto = ApiDependencies.invoker.services.images.create(
            image=pil_image,
            image_origin=ResourceOrigin.EXTERNAL,
            image_category=image_category,
            session_id=session_id,
            board_id=board_id,
            metadata=extracted_metadata.invokeai_metadata,
            workflow=extracted_metadata.invokeai_workflow,
            graph=extracted_metadata.invokeai_graph,
            is_intermediate=is_intermediate,
        )

        response.status_code = 201
        response.headers["Location"] = image_dto.image_url

        return image_dto
    except Exception:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to create image")


class ImageUploadEntry(BaseModel):
    image_dto: ImageDTO = Body(description="The image DTO")
    presigned_url: str = Body(description="The URL to get the presigned URL for the image upload")


@images_router.post("/", operation_id="create_image_upload_entry")
async def create_image_upload_entry(
    width: int = Body(description="The width of the image"),
    height: int = Body(description="The height of the image"),
    board_id: Optional[str] = Body(default=None, description="The board to add this image to, if any"),
) -> ImageUploadEntry:
    """Uploads an image from a URL, not implemented"""

    raise HTTPException(status_code=501, detail="Not implemented")


@images_router.delete("/i/{image_name}", operation_id="delete_image", response_model=DeleteImagesResult)
async def delete_image(
    image_name: str = Path(description="The name of the image to delete"),
) -> DeleteImagesResult:
    """Deletes an image"""

    deleted_images: set[str] = set()
    affected_boards: set[str] = set()

    try:
        image_dto = ApiDependencies.invoker.services.images.get_dto(image_name)
        board_id = image_dto.board_id or "none"
        ApiDependencies.invoker.services.images.delete(image_name)
        deleted_images.add(image_name)
        affected_boards.add(board_id)
    except Exception:
        # TODO: Does this need any exception handling at all?
        pass

    return DeleteImagesResult(
        deleted_images=list(deleted_images),
        affected_boards=list(affected_boards),
    )


@images_router.delete("/intermediates", operation_id="clear_intermediates")
async def clear_intermediates() -> int:
    """Clears all intermediates"""

    try:
        count_deleted = ApiDependencies.invoker.services.images.delete_intermediates()
        return count_deleted
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to clear intermediates")
        pass


@images_router.get("/intermediates", operation_id="get_intermediates_count")
async def get_intermediates_count() -> int:
    """Gets the count of intermediate images"""

    try:
        return ApiDependencies.invoker.services.images.get_intermediates_count()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to get intermediates")
        pass


@images_router.patch(
    "/i/{image_name}",
    operation_id="update_image",
    response_model=ImageDTO,
)
async def update_image(
    image_name: str = Path(description="The name of the image to update"),
    image_changes: ImageRecordChanges = Body(description="The changes to apply to the image"),
) -> ImageDTO:
    """Updates an image"""

    try:
        return ApiDependencies.invoker.services.images.update(image_name, image_changes)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to update image")


@images_router.get(
    "/i/{image_name}",
    operation_id="get_image_dto",
    response_model=ImageDTO,
)
async def get_image_dto(
    image_name: str = Path(description="The name of image to get"),
) -> ImageDTO:
    """Gets an image's DTO"""

    try:
        return ApiDependencies.invoker.services.images.get_dto(image_name)
    except Exception:
        raise HTTPException(status_code=404)


@images_router.get(
    "/i/{image_name}/metadata",
    operation_id="get_image_metadata",
    response_model=Optional[MetadataField],
)
async def get_image_metadata(
    image_name: str = Path(description="The name of image to get"),
) -> Optional[MetadataField]:
    """Gets an image's metadata"""

    try:
        return ApiDependencies.invoker.services.images.get_metadata(image_name)
    except Exception:
        raise HTTPException(status_code=404)


class WorkflowAndGraphResponse(BaseModel):
    workflow: Optional[str] = Field(description="The workflow used to generate the image, as stringified JSON")
    graph: Optional[str] = Field(description="The graph used to generate the image, as stringified JSON")


@images_router.get(
    "/i/{image_name}/workflow", operation_id="get_image_workflow", response_model=WorkflowAndGraphResponse
)
async def get_image_workflow(
    image_name: str = Path(description="The name of image whose workflow to get"),
) -> WorkflowAndGraphResponse:
    try:
        workflow = ApiDependencies.invoker.services.images.get_workflow(image_name)
        graph = ApiDependencies.invoker.services.images.get_graph(image_name)
        return WorkflowAndGraphResponse(workflow=workflow, graph=graph)
    except Exception:
        raise HTTPException(status_code=404)


@images_router.get(
    "/i/{image_name}/full",
    operation_id="get_image_full",
    response_class=Response,
    responses={
        200: {
            "description": "Return the full-resolution image",
            "content": {"image/png": {}},
        },
        404: {"description": "Image not found"},
    },
)
@images_router.head(
    "/i/{image_name}/full",
    operation_id="get_image_full_head",
    response_class=Response,
    responses={
        200: {
            "description": "Return the full-resolution image",
            "content": {"image/png": {}},
        },
        404: {"description": "Image not found"},
    },
)
async def get_image_full(
    image_name: str = Path(description="The name of full-resolution image file to get"),
) -> Response:
    """Gets a full-resolution image file"""

    try:
        path = ApiDependencies.invoker.services.images.get_path(image_name)
        with open(path, "rb") as f:
            content = f.read()
        import mimetypes
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "image/png"
        response = Response(content, media_type=mime_type)
