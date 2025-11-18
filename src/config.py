"""애플리케이션 설정 관리 모듈"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 전역 설정"""

    # ==========================================
    # 1. 프로젝트 기본 정보
    # ==========================================
    PROJECT_NAME: str = "Image Search with Quantization"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ==========================================
    # 2. 경로 설정
    # ==========================================
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    LOG_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    MODEL_CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")

    # ==========================================
    # 3. 디바이스 설정
    # ==========================================
    DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"

    @field_validator("DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """디바이스 유효성 검증"""
        try:
            import torch

            if v == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            elif v == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS is not available")
        except ImportError:
            # torch가 설치되어 있지 않은 경우, cpu만 허용
            if v != "cpu":
                raise ValueError("torch is not installed. Only 'cpu' device is available.")

        return v

    # ==========================================
    # 4. CLIP 모델 설정
    # ==========================================
    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_PRETRAINED: str = "openai"
    EMBEDDING_DIM: int = 512
    IMAGE_SIZE: int = 224

    # ==========================================
    # 5. Qdrant 설정
    # ==========================================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "image_embeddings"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_PREFER_GRPC: bool = False

    # ==========================================
    # 6. 양자화 설정
    # ==========================================
    QUANTIZATION_ENABLED: bool = True
    QUANTIZATION_METHOD: Literal["scalar", "product"] = "scalar"
    QUANTIZATION_BITS: int = 8

    @field_validator("QUANTIZATION_BITS")
    @classmethod
    def validate_quantization_bits(cls, v: int) -> int:
        """양자화 비트 검증"""
        if v not in [1, 2, 4, 8]:
            raise ValueError(f"QUANTIZATION_BITS must be one of [1, 2, 4, 8], got {v}")
        return v

    # ==========================================
    # 7. 검색 파라미터
    # ==========================================
    SEARCH_TOP_K: int = Field(default=10, ge=1, le=100)
    SEARCH_SCORE_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)

    # ==========================================
    # 8. 로깅 설정
    # ==========================================
    LOG_LEVEL: str = "INFO"
    LOG_ROTATION: str = "500 MB"
    LOG_RETENTION: str = "10 days"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

    # ==========================================
    # 9. API 설정
    # ==========================================
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = False

    # ==========================================
    # 10. Pydantic 설정
    # ==========================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def model_post_init(self, __context) -> None:
        """설정 초기화 후 디렉토리 생성"""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# 전역 설정 인스턴스
settings = Settings()
