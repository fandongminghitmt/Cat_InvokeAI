from logging import Logger

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_1 import build_migration_1
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2 import build_migration_2
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_3 import build_migration_3
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_4 import build_migration_4
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_5 import build_migration_5
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_6 import build_migration_6
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_7 import build_migration_7
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_8 import build_migration_8
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_9 import build_migration_9
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_10 import build_migration_10
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_11 import build_migration_11
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_12 import build_migration_12
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_13 import build_migration_13
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_14 import build_migration_14
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_15 import build_migration_15
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_16 import build_migration_16
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_17 import build_migration_17
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_18 import build_migration_18
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_19 import build_migration_19
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_20 import build_migration_20
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_21 import build_migration_21
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_22 import build_migration_22
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_23 import build_migration_23
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_24 import build_migration_24
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_25 import build_migration_25
    migrator.run_migrations()

    return db
