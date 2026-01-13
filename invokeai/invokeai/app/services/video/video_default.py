from typing import Optional
from PIL import Image
import io

from invokeai.app.services.video.video_base import VideoServiceABC
from invokeai.app.services.images.images_common import VideoDTO
from invokeai.app.services.image_records.image_records_common import ResourceOrigin, ImageCategory
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.image_records.video_records_sqlite import SqliteVideoRecordStorage
from invokeai.app.services.image_files.video_files_disk import DiskVideoFileStorage

class VideoService(VideoServiceABC):
    __invoker: Invoker
    __records: SqliteVideoRecordStorage
    __files: DiskVideoFileStorage

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        # Initialize storage services
        # Note: We are reusing the main DB connection from image_records for now
        self.__records = SqliteVideoRecordStorage(invoker.services.image_records._db)
        
        # Initialize file storage
        output_folder = invoker.services.configuration.outputs_path
        self.__files = DiskVideoFileStorage(f"{output_folder}/videos")

    def create(
        self,
        video_file: bytes,
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
    ) -> VideoDTO:
        
        # Generate a unique name
        video_name = self.__invoker.services.names.create_image_name().replace(".png", ".mp4") # Hacky extension swap
        
        # Placeholder for metadata extraction (FFmpeg would go here)
        width = 1920
        height = 1080
        duration = 10.0
        fps = 30.0
        
        # Save file to disk
        # Extract thumbnail from video
        try:
            import cv2
            import tempfile
            import os
            
            # Save bytes to temp file for OpenCV
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                tmp_vid.write(video_file)
                tmp_vid_path = tmp_vid.name
                
            cap = cv2.VideoCapture(tmp_vid_path)
            ret, frame = cap.read()
            cap.release()
            
            try:
                os.unlink(tmp_vid_path)
            except:
                pass
                
            if ret:
                print("DEBUG: Successfully extracted video frame.")
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thumb_img = Image.fromarray(frame_rgb)
                thumb_img.thumbnail((512, 512))
            else:
                # Fallback
                print("DEBUG: Failed to read video frame (ret=False). Using RED fallback.")
                thumb_img = Image.new('RGB', (256, 256), color = 'red')
                
        except Exception as e:
            print(f"DEBUG: Exception in video thumbnail: {e}")
            self.__invoker.services.logger.error(f"Failed to extract video thumbnail: {e}")
            thumb_img = Image.new('RGB', (256, 256), color = 'red')

        self.__files.save(video_name, video_file, thumbnail=thumb_img)
        
        # Save record to DB
        self.__records.save(
            video_name=video_name,
            video_origin=video_origin,
            video_category=video_category,
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            has_workflow=workflow is not None,
            is_intermediate=is_intermediate,
            node_id=node_id,
            session_id=session_id,
            metadata=metadata
        )

        return self.get_dto(video_name)

    def get_dto(self, video_name: str) -> VideoDTO:
        record = self.__records.get(video_name)
        
        # Construct DTO
        # Note: URLs need to be handled by a URL service extension or reusing image URLs logic
        base_url = "api/v1/media/video" # Placeholder base
        
        return VideoDTO(
            video_name=record.video_name,
            video_origin=record.video_origin,
            video_category=record.video_category,
            video_url=f"{base_url}/{video_name}", # Needs actual URL generation logic
            thumbnail_url=f"{base_url}/{video_name}/thumbnail",
            width=record.width,
            height=record.height,
            duration=record.duration,
            fps=record.fps,
            created_at=str(record.created_at),
            updated_at=str(record.updated_at),
            deleted_at=str(record.deleted_at) if record.deleted_at else None,
            is_intermediate=record.is_intermediate,
            session_id=record.session_id,
            node_id=record.node_id,
            starred=record.starred,
            has_workflow=record.has_workflow,
            board_id=None # Board support later
        )
