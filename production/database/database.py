#!/usr/bin/env python3
"""
TGCF-IDS Database Connection & Session Management.
"""

import os
import logging
from typing import Optional
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from production.database.models import Base
from src.utils.paths import ProjectPaths

logger = logging.getLogger("production.database")


class DatabaseManager:
    """Manages SQLite / PostgreSQL database connection pooling and schema migrations."""

    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            # Default to local SQLite database in production/data/
            db_dir = Path("production/data")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{db_dir / 'production_nids.db'}"

        self.db_url = db_url
        is_sqlite = db_url.startswith("sqlite")

        connect_args = {"check_same_thread": False} if is_sqlite else {}
        self.engine = create_engine(self.db_url, connect_args=connect_args)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()

    def create_tables(self):
        """Creates all relational tables if they do not already exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("[+] Production database schema initialized and migrated.")

    def get_session(self) -> Session:
        return self.SessionLocal()
