import sqlite3
from datetime import datetime
from typing import Optional, Union, cast

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
    AudioRecord,
    AudioRecordChanges,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ImageRecordDeleteException,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

class SqliteAudioRecordStorage:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db

    def get(self, audio_name: str) -> AudioRecord:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    SELECT * FROM audios
                    WHERE audio_name = ?;
                    """,
                    (audio_name,),
                )
                result = cast(Optional[sqlite3.Row], cursor.fetchone())
            except sqlite3.Error as e:
                raise ImageRecordNotFoundException from e

        if not result:
            raise ImageRecordNotFoundException

        return self._deserialize_audio_record(dict(result))

    def save(
        self,
        audio_name: str,
        audio_origin: ResourceOrigin,
        audio_category: ImageCategory,
        duration: float,
        has_workflow: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> datetime:
        with self._db.transaction() as cursor:
            try:
                cursor.execute(
                    """--sql
                    INSERT OR IGNORE INTO audios (
                        audio_name,
                        audio_origin,
                        audio_category,
                        duration,
                        node_id,
                        session_id,
                        is_intermediate,
                        starred,
                        has_workflow
                        )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        audio_name,
                        audio_origin.value,
                        audio_category.value,
                        duration,
                        node_id,
                        session_id,
                        is_intermediate,
                        starred,
                        has_workflow,
                    ),
                )

                cursor.execute(
                    """--sql
                    SELECT created_at
                    FROM audios
                    WHERE audio_name = ?;
                    """,
                    (audio_name,),
                )

                created_at = datetime.fromisoformat(cursor.fetchone()[0])

            except sqlite3.Error as e:
                raise ImageRecordSaveException from e
        return created_at

    def _deserialize_audio_record(self, audio_dict: dict) -> AudioRecord:
        return AudioRecord(
            audio_name=audio_dict.get("audio_name", "unknown"),
            audio_origin=ResourceOrigin(audio_dict.get("audio_origin", ResourceOrigin.INTERNAL.value)),
            audio_category=ImageCategory(audio_dict.get("audio_category", ImageCategory.GENERAL.value)),
            duration=audio_dict.get("duration", 0.0),
            session_id=audio_dict.get("session_id", None),
            node_id=audio_dict.get("node_id", None),
            created_at=audio_dict.get("created_at"),
            updated_at=audio_dict.get("updated_at"),
            deleted_at=audio_dict.get("deleted_at"),
            is_intermediate=audio_dict.get("is_intermediate", False),
            starred=audio_dict.get("starred", False),
            has_workflow=audio_dict.get("has_workflow", False),
        )
