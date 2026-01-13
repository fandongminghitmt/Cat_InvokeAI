import sqlite3
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

class Migration25Callback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        """Migration callback for database version 25."""
        self._create_videos_table(cursor)
        self._create_audios_table(cursor)

    def _create_videos_table(self, cursor: sqlite3.Cursor) -> None:
        """Creates the `videos` table, indices and triggers."""
        
        # Check if table exists first to be safe, although migrations should ensure order
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='videos';")
        if cursor.fetchone():
            return

        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS videos (
                video_name TEXT NOT NULL PRIMARY KEY,
                video_origin TEXT NOT NULL,
                video_category TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                duration REAL NOT NULL,
                fps REAL NOT NULL,
                session_id TEXT,
                node_id TEXT,
                metadata TEXT,
                is_intermediate BOOLEAN DEFAULT FALSE,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                deleted_at DATETIME,
                starred BOOLEAN DEFAULT FALSE,
                has_workflow BOOLEAN DEFAULT FALSE
            );
            """
        ]

        indices = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_video_name ON videos(video_name);",
            "CREATE INDEX IF NOT EXISTS idx_videos_video_origin ON videos(video_origin);",
            "CREATE INDEX IF NOT EXISTS idx_videos_video_category ON videos(video_category);",
            "CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_videos_starred ON videos(starred);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_videos_updated_at
            AFTER UPDATE
            ON videos FOR EACH ROW
            BEGIN
                UPDATE videos SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE video_name = old.video_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)

    def _create_audios_table(self, cursor: sqlite3.Cursor) -> None:
        """Creates the `audios` table, indices and triggers."""
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audios';")
        if cursor.fetchone():
            return

        tables = [
            """--sql
            CREATE TABLE IF NOT EXISTS audios (
                audio_name TEXT NOT NULL PRIMARY KEY,
                audio_origin TEXT NOT NULL,
                audio_category TEXT NOT NULL,
                duration REAL NOT NULL,
                session_id TEXT,
                node_id TEXT,
                is_intermediate BOOLEAN DEFAULT FALSE,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                deleted_at DATETIME,
                starred BOOLEAN DEFAULT FALSE,
                has_workflow BOOLEAN DEFAULT FALSE
            );
            """
        ]

        indices = [
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_audios_audio_name ON audios(audio_name);",
            "CREATE INDEX IF NOT EXISTS idx_audios_audio_origin ON audios(audio_origin);",
            "CREATE INDEX IF NOT EXISTS idx_audios_audio_category ON audios(audio_category);",
            "CREATE INDEX IF NOT EXISTS idx_audios_created_at ON audios(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_audios_starred ON audios(starred);",
        ]

        triggers = [
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_audios_updated_at
            AFTER UPDATE
            ON audios FOR EACH ROW
            BEGIN
                UPDATE audios SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE audio_name = old.audio_name;
            END;
            """
        ]

        for stmt in tables + indices + triggers:
            cursor.execute(stmt)


def build_migration_25() -> Migration:
    """
    Builds the migration from database version 24 to 25.
    
    - Create `videos` table
    - Create `audios` table
    """

    return Migration(
        from_version=24,
        to_version=25,
        callback=Migration25Callback(),
    )
