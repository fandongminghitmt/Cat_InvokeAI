import io
import traceback
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Path, Query, Request, Response, UploadFile
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.images.images_common import VideoDTO, AudioDTO

media_router = APIRouter(prefix="/v1/media", tags=["media"])

@media_router.post(
    "/video/upload",
    operation_id="upload_video",
    responses={
        201: {"description": "The video was uploaded successfully"},
        415: {"description": "Video upload failed"},
    },
    status_code=201,
    response_model=VideoDTO,
)
async def upload_video(
    file: UploadFile,
    request: Request,
    response: Response,
    video_category: ImageCategory = Query(description="The category of the video"),
    is_intermediate: bool = Query(description="Whether this is an intermediate video"),
    board_id: Optional[str] = Query(default=None, description="The board to add this video to, if any"),
    session_id: Optional[str] = Query(default=None, description="The session ID associated with this upload, if any"),
) -> VideoDTO:
    """Uploads a video"""
    # Force uploads to be USER category if they are passed as GENERAL
    if video_category == ImageCategory.GENERAL:
        video_category = ImageCategory.USER

    if not file.content_type or not file.content_type.startswith("video"):
        raise HTTPException(status_code=415, detail="Not a video")

    contents = await file.read()
    
    try:
        return ApiDependencies.invoker.services.videos.create(
            video_file=contents,
            video_origin=ResourceOrigin.EXTERNAL,
            video_category=video_category,
            session_id=session_id,
            board_id=board_id,
            is_intermediate=is_intermediate,
        )
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Video support not fully implemented")
    except Exception as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@media_router.get(
    "/video/{video_name}",
    operation_id="get_video_dto",
    response_model=VideoDTO,
)
async def get_video_dto(
    video_name: str = Path(description="The name of the video to get"),
) -> VideoDTO:
    """Gets a video DTO"""
    try:
        return ApiDependencies.invoker.services.videos.get_dto(video_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Video not found")



@media_router.get(
    "/video/{video_name}/full",
    operation_id="get_video_full",
    response_class=Response,
)
async def get_video_full(
    video_name: str = Path(description="The name of the video to get"),
) -> Response:
    """Gets a full-resolution video file"""
    try:
        # Access private member __files to get path. 
        # This is a hack because VideoService doesn't expose get_path yet.
        path = ApiDependencies.invoker.services.videos._VideoService__files.get_path(video_name)
        
        with open(path, "rb") as f:
            content = f.read()
            
        import mimetypes
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            mime_type = "video/mp4" # Default to mp4 if unknown
            
        response = Response(content, media_type=mime_type)
        # response.headers["Cache-Control"] = "max-age=31536000"
        response.headers["Content-Disposition"] = f'inline; filename="{video_name}"'
        return response
    except Exception as e:
        ApiDependencies.invoker.services.logger.error(f"Failed to get video: {str(e)}")
        raise HTTPException(status_code=404, detail="Video not found")



@media_router.get(
    "/video/{video_name}/thumbnail",
    operation_id="get_video_thumbnail",
    response_class=Response,
)
async def get_video_thumbnail(
    video_name: str = Path(description="The name of the video to get thumbnail for"),
) -> Response:
    """Gets a video thumbnail"""
    try:
        # Access private member __files to get path. 
        files_storage = ApiDependencies.invoker.services.videos._VideoService__files
        
        # Use the get_path method with thumbnail=True
        # Note: video_name passed here is the VIDEO name (e.g. xxxx.mp4)
        # get_path logic: filename = get_thumbnail_name(video_name) if thumbnail else video_name
        # So we pass video_name directly.
        
        path = files_storage.get_path(video_name, thumbnail=True)
        
        # DEBUG LOG
        print(f"DEBUG: Requesting thumbnail for {video_name} at {path}")
        
        if not path.exists():
             print(f"DEBUG: Thumbnail NOT FOUND at {path}")
             # Fallback: check if there is a .png or something else?
             # Or maybe video_name has extension but thumbnail doesn't?
             # Try stripping extension from video_name?
             # video_name = "abc.mp4" -> thumbnail = "abc.webp" ?
             # get_thumbnail_name implementation: os.path.splitext(image_name)[0] + ".webp"
             # So if video_name="abc.mp4", thumb="abc.webp".
             # This seems correct.
             
             raise Exception(f"Thumbnail file not found at {path}")

        with open(path, "rb") as f:
            content = f.read()
            
        return Response(content, media_type="image/webp")
    except Exception as e:
        ApiDependencies.invoker.services.logger.error(f"Failed to get video thumbnail: {str(e)}")
        raise HTTPException(status_code=404, detail="Thumbnail not found")


@media_router.post(
    "/audio/upload",
    operation_id="upload_audio",
    responses={
        201: {"description": "The audio was uploaded successfully"},
        415: {"description": "Audio upload failed"},
    },
    status_code=201,
    response_model=AudioDTO,
)
async def upload_audio(
    file: UploadFile,
    request: Request,
    response: Response,
    audio_category: ImageCategory = Query(description="The category of the audio"),
    is_intermediate: bool = Query(description="Whether this is an intermediate audio"),
    board_id: Optional[str] = Query(default=None, description="The board to add this audio to, if any"),
    session_id: Optional[str] = Query(default=None, description="The session ID associated with this upload, if any"),
) -> AudioDTO:
    """Uploads an audio"""
    if not file.content_type or not file.content_type.startswith("audio"):
        raise HTTPException(status_code=415, detail="Not an audio")

    contents = await file.read()
    
    try:
        return ApiDependencies.invoker.services.audios.create(
            audio_file=contents,
            audio_origin=ResourceOrigin.EXTERNAL,
            audio_category=audio_category,
            session_id=session_id,
            board_id=board_id,
            is_intermediate=is_intermediate,
        )
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Audio support not fully implemented")
    except Exception as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@media_router.get(
    "/audio/{audio_name}",
    operation_id="get_audio_dto",
    response_model=AudioDTO,
)
async def get_audio_dto(
    audio_name: str = Path(description="The name of the audio to get"),
) -> AudioDTO:
    """Gets an audio DTO"""
    try:
        return ApiDependencies.invoker.services.audios.get_dto(audio_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Audio not found")
