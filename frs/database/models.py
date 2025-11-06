"""Database models for Face Recognition Service."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, Float, Integer, LargeBinary, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Identity(Base):
    """Identity table to store face embeddings and metadata."""

    __tablename__ = "identities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    identity_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Serialized numpy array
    image_path = Column(String(512), nullable=True)
    meta_data = Column(String(1024), nullable=True)  # JSON string for additional info
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Identity(id={self.id}, identity_id={self.identity_id}, name={self.name})>"


class RecognitionLog(Base):
    """Log table for recognition events."""

    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    identity_id = Column(String(255), nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    image_path = Column(String(512), nullable=True)
    bbox = Column(String(128), nullable=True)  # JSON string: [x1, y1, x2, y2]
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    processing_time_ms = Column(Float, nullable=True)

    def __repr__(self):
        return f"<RecognitionLog(id={self.id}, identity_id={self.identity_id}, confidence={self.confidence})>"


class Database:
    """Database manager class."""

    def __init__(self, db_url: str):
        """Initialize database connection.

        Args:
            db_url: Database URL (e.g., 'sqlite:///data/frs.db' or PostgreSQL URL)
        """
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self):
        """Get database session."""
        return self.SessionLocal()

    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(bind=self.engine)
