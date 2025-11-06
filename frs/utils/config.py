"""Configuration management for FRS."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel


class ServiceConfig(BaseModel):
    name: str
    version: str
    host: str
    port: int
    workers: int
    log_level: str


class DetectionConfig(BaseModel):
    model_type: str
    model_path: str
    confidence_threshold: float
    nms_threshold: float
    top_k: int
    keep_top_k: int
    input_size: list
    min_face_size: int
    max_face_size: int
    blur_threshold: float
    brightness_range: list


class AlignmentConfig(BaseModel):
    method: str
    output_size: list
    normalize: bool
    mean: list
    std: list


class EmbeddingConfig(BaseModel):
    model_type: str
    model_path: str
    embedding_size: int
    batch_size: int
    use_onnx: bool
    onnx_threads: int


class MatchingConfig(BaseModel):
    metric: str
    threshold: float
    top_k: int
    use_faiss: bool
    faiss_index_type: str
    min_confidence: float


class DatabaseConfig(BaseModel):
    type: str
    sqlite_path: str
    postgresql_url: str
    pool_size: int
    max_overflow: int


class Config:
    """Main configuration class."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @property
    def service(self) -> ServiceConfig:
        return ServiceConfig(**self._config["service"])

    @property
    def detection(self) -> DetectionConfig:
        return DetectionConfig(**self._config["detection"])

    @property
    def alignment(self) -> AlignmentConfig:
        return AlignmentConfig(**self._config["alignment"])

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig(**self._config["embedding"])

    @property
    def matching(self) -> MatchingConfig:
        return MatchingConfig(**self._config["matching"])

    @property
    def database(self) -> DatabaseConfig:
        return DatabaseConfig(**self._config["database"])


# Global configuration instance
config = Config()
