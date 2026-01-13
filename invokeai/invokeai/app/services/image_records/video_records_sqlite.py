import sqlite3
from datetime import datetime
from typing import Optional, Union, cast

from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
    VideoRecord,
    VideoRecordChanges,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ImageRecordDeleteException,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

class SqliteVideoRecordStorage:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def get(self, video_name: str) -> VideoRecord:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT * FROM videos
                    WHERE video_name = ?;
                    """,
                    (video_name,),
                )
                result = cast(Optional[sqlite3.Row], cursor.fetchone())
            except sqlite3.Error as e:
                raise ImageRecordNotFoundException from e

        if not result:
            raise ImageRecordNotFoundException

        return self._deserialize_video_record(dict(result))

    def save(
        self,
        video_name: str,
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        width: int,
        height: int,
        duration: float,
        fps: float,
        has_workflow: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> datetime:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO videos (
                        video_name,
                        video_origin,
                        video_category,
                        width,
                        height,
                        duration,
                        fps,
                        node_id,
                        session_id,
                        metadata,
                        is_intermediate,
                        starred,
                        has_workflow
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        video_name,
                        video_origin.value,
                        video_category.value,
                        width,
                        height,
                        duration,
                        fps,
                        node_id,
                        session_id,
                        metadata,
                        is_intermediate,
                        starred,
                        has_workflow,
                    ),
                )

                cursor.execute(
                    """--sql
                    SELECT created_at
                    FROM videos
                    WHERE video_name = ?;
                    """,
                    (video_name,),
                )

                created_at = datetime.fromisoformat(cursor.fetchone()[0])

            except sqlite3.Error as e:
                raise ImageRecordSaveException from e
        return created_at

    def _deserialize_video_record(self, video_dict: dict) -> VideoRecord:
        # Helper to convert dict to VideoRecord
        return VideoRecord(
            video_name=video_dict.get("video_name", "unknown"),
            video_origin=ResourceOrigin(video_dict.get("video_origin", ResourceOrigin.INTERNAL.value)),
            video_category=ImageCategory(video_dict.get("video_category", ImageCategory.GENERAL.value)),
            width=video_dict.get("width", 0),
            height=video_dict.get("height", 0),
            duration=video_dict.get("duration", 0.0),
            fps=video_dict.get("fps", 0.0),
            session_id=video_dict.get("session_id", None),
            node_id=video_dict.get("node_id", None),
            created_at=video_dict.get("created_at"),
            updated_at=video_dict.get("updated_at"),
            deleted_at=video_dict.get("deleted_at"),
            is_intermediate=video_dict.get("is_intermediate", False),
            starred=video_dict.get("starred", False),
            has_workflow=video_dict.get("has_workflow", False),
        )
