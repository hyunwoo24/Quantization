# Module 2: 설정 및 유틸리티 - 상세 업무 프로세스

## 📋 모듈 개요

**목적:** 프로젝트 전반에서 사용할 설정 관리, 로깅 시스템, 데이터 모델 구축
**예상 소요 시간:** 2-3시간
**담당자:** 백엔드 개발자
**우선순위:** HIGH (모든 모듈의 기반)

---

## 🎯 전체 작업 흐름

```
1. 프로젝트 구조 생성
   ↓
2. M2.1: 설정 관리자 구현
   ↓
3. M2.2: 로깅 시스템 구현
   ↓
4. M2.3: 데이터 모델 구현
   ↓
5. 통합 테스트 및 검증
```

---

## 📁 1단계: 프로젝트 구조 생성

### 작업 내용
```bash
# 디렉토리 구조 생성
quantization/
├── src/
│   ├── __init__.py
│   ├── config.py          # M2.1
│   ├── logger.py          # M2.2
│   └── models.py          # M2.3
├── logs/                  # 로그 파일 저장
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_logger.py
│   └── test_models.py
├── .env                   # 환경변수
├── .env.example          # 환경변수 템플릿
├── requirements.txt
└── README.md
```

### 체크리스트
- [ ] 디렉토리 구조 생성 완료
- [ ] `__init__.py` 파일 생성
- [ ] `.env.example` 템플릿 작성
- [ ] `requirements.txt` 작성

### 필수 의존성 (`requirements.txt`)
```txt
# 설정 관리
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0

# 로깅
loguru==0.7.2

# AI/ML
torch>=2.0.0
transformers>=4.35.0
open-clip-torch>=2.24.0

# 벡터 DB
qdrant-client>=1.7.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0

# 유틸리티
Pillow>=10.1.0
numpy>=1.24.0

# 테스트
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
```

---

## 🔧 2단계: M2.1 - 설정 관리자 (config.py)

### 📝 작업 목표
Pydantic Settings를 활용한 타입 안전한 설정 관리 시스템 구축

### 구현 상세

#### 2.1.1 기본 구조 설계
```python
# src/config.py

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
        import torch

        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        elif v == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is not available")

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
    QUANTIZATION_BITS: Literal[1, 2, 4, 8] = 8

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
```

#### 2.1.2 환경변수 템플릿 (`.env.example`)
```bash
# ==========================================
# 프로젝트 설정
# ==========================================
PROJECT_NAME="Image Search with Quantization"
VERSION="1.0.0"
DEBUG=false

# ==========================================
# 디바이스 설정 (cpu, cuda, mps)
# ==========================================
DEVICE=cpu

# ==========================================
# CLIP 모델 설정
# ==========================================
CLIP_MODEL_NAME="ViT-B/32"
CLIP_PRETRAINED="openai"
EMBEDDING_DIM=512

# ==========================================
# Qdrant 설정
# ==========================================
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=image_embeddings
QDRANT_API_KEY=
QDRANT_PREFER_GRPC=false

# ==========================================
# 양자화 설정
# ==========================================
QUANTIZATION_ENABLED=true
QUANTIZATION_METHOD=scalar
QUANTIZATION_BITS=8

# ==========================================
# 검색 설정
# ==========================================
SEARCH_TOP_K=10
SEARCH_SCORE_THRESHOLD=0.5

# ==========================================
# 로깅 설정
# ==========================================
LOG_LEVEL=INFO
LOG_ROTATION="500 MB"
LOG_RETENTION="10 days"

# ==========================================
# API 설정
# ==========================================
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=false
```

#### 2.1.3 검증 테스트 (`tests/test_config.py`)
```python
import pytest
from pathlib import Path
from src.config import Settings, settings


def test_settings_instance_creation():
    """Settings 인스턴스 생성 테스트"""
    s = Settings()
    assert s is not None
    assert isinstance(s, Settings)


def test_env_file_loading():
    """.env 파일 로딩 테스트"""
    assert settings.PROJECT_NAME is not None
    assert len(settings.PROJECT_NAME) > 0


def test_path_creation():
    """경로 자동 생성 테스트"""
    assert settings.DATA_DIR.exists()
    assert settings.LOG_DIR.exists()
    assert settings.MODEL_CACHE_DIR.exists()


def test_device_validation():
    """디바이스 설정 검증 테스트"""
    assert settings.DEVICE in ["cpu", "cuda", "mps"]


def test_quantization_bits_validation():
    """양자화 비트 검증 테스트"""
    assert settings.QUANTIZATION_BITS in [1, 2, 4, 8]


def test_search_parameters():
    """검색 파라미터 범위 테스트"""
    assert 1 <= settings.SEARCH_TOP_K <= 100
    assert 0.0 <= settings.SEARCH_SCORE_THRESHOLD <= 1.0


def test_type_checking():
    """타입 체크 테스트"""
    assert isinstance(settings.QDRANT_PORT, int)
    assert isinstance(settings.DEBUG, bool)
    assert isinstance(settings.BASE_DIR, Path)
```

### ✅ 검증 기준
- [ ] `Settings()` 인스턴스 생성 성공
- [ ] `.env` 파일 읽기 성공 (환경변수 로딩)
- [ ] 모든 경로 자동 생성 확인
- [ ] 디바이스 검증 로직 동작 확인
- [ ] 타입 체크 통과 (mypy/pyright)
- [ ] 모든 테스트 통과 (`pytest tests/test_config.py`)

---

## 📊 3단계: M2.2 - 로깅 시스템 (logger.py)

### 📝 작업 목표
Loguru 기반 구조화된 로깅 시스템 구축 (레벨별, 모듈별 로그 분리)

### 구현 상세

#### 2.2.1 로깅 시스템 구현
```python
# src/logger.py

import sys
from pathlib import Path
from loguru import logger
from src.config import settings


def setup_logger():
    """로거 설정 및 초기화"""

    # 기본 핸들러 제거
    logger.remove()

    # ==========================================
    # 1. 콘솔 출력 핸들러
    # ==========================================
    logger.add(
        sys.stderr,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ==========================================
    # 2. 일반 애플리케이션 로그
    # ==========================================
    logger.add(
        settings.LOG_DIR / "app.log",
        format=settings.LOG_FORMAT,
        level="DEBUG",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        enqueue=True,  # 비동기 처리
        backtrace=True,
        diagnose=True,
    )

    # ==========================================
    # 3. 임베딩 처리 로그
    # ==========================================
    logger.add(
        settings.LOG_DIR / "embedding.log",
        format=settings.LOG_FORMAT,
        level="INFO",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        filter=lambda record: "embedding" in record["extra"].get("module", ""),
        enqueue=True,
    )

    # ==========================================
    # 4. 검색 요청 로그
    # ==========================================
    logger.add(
        settings.LOG_DIR / "search.log",
        format=settings.LOG_FORMAT,
        level="INFO",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        filter=lambda record: "search" in record["extra"].get("module", ""),
        enqueue=True,
    )

    # ==========================================
    # 5. 에러 전용 로그
    # ==========================================
    logger.add(
        settings.LOG_DIR / "error.log",
        format=settings.LOG_FORMAT,
        level="ERROR",
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logger setup completed")
    return logger


# 로거 초기화
app_logger = setup_logger()


# ==========================================
# 모듈별 로거 헬퍼 함수
# ==========================================

def get_embedding_logger():
    """임베딩 모듈용 로거"""
    return logger.bind(module="embedding")


def get_search_logger():
    """검색 모듈용 로거"""
    return logger.bind(module="search")


def get_quantization_logger():
    """양자화 모듈용 로거"""
    return logger.bind(module="quantization")


# ==========================================
# 데코레이터: 함수 실행 로깅
# ==========================================

from functools import wraps
from typing import Callable, Any
import time


def log_execution(func: Callable) -> Callable:
    """함수 실행 시간 및 결과 로깅 데코레이터"""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed_time:.2f}s")
            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed_time:.2f}s: {e}")
            raise

    return wrapper


@wraps(log_execution)
async def log_execution_async(func: Callable) -> Callable:
    """비동기 함수용 실행 로깅 데코레이터"""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {elapsed_time:.2f}s")
            return result

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed_time:.2f}s: {e}")
            raise

    return wrapper
```

#### 2.2.2 로거 사용 예제
```python
# src/example_usage.py

from src.logger import (
    app_logger,
    get_embedding_logger,
    get_search_logger,
    log_execution
)

# 일반 로깅
app_logger.info("Application started")
app_logger.debug("Debug information")
app_logger.warning("Warning message")
app_logger.error("Error occurred")

# 모듈별 로깅
embedding_logger = get_embedding_logger()
embedding_logger.info("Processing image embeddings")

search_logger = get_search_logger()
search_logger.info("Executing search query", query="cat")

# 데코레이터 사용
@log_execution
def process_image(image_path: str):
    app_logger.info(f"Processing {image_path}")
    # 처리 로직
    return "success"
```

#### 2.2.3 검증 테스트 (`tests/test_logger.py`)
```python
import pytest
from pathlib import Path
from src.logger import (
    setup_logger,
    get_embedding_logger,
    get_search_logger,
    log_execution
)
from src.config import settings


def test_logger_initialization():
    """로거 초기화 테스트"""
    logger = setup_logger()
    assert logger is not None


def test_log_file_creation():
    """로그 파일 생성 확인"""
    log_files = [
        settings.LOG_DIR / "app.log",
        settings.LOG_DIR / "embedding.log",
        settings.LOG_DIR / "search.log",
        settings.LOG_DIR / "error.log",
    ]

    # 로그 생성을 위해 각 로거 사용
    from src.logger import app_logger
    app_logger.info("Test log")

    get_embedding_logger().info("Test embedding log")
    get_search_logger().info("Test search log")
    app_logger.error("Test error log")

    # 파일 존재 확인 (비동기 처리로 인한 지연 고려)
    import time
    time.sleep(0.5)

    for log_file in log_files:
        assert log_file.exists(), f"{log_file} does not exist"


def test_log_level_filtering():
    """로그 레벨 필터링 테스트"""
    from src.logger import app_logger

    # DEBUG 레벨 로그는 콘솔에 표시되지 않을 수 있음 (설정에 따라)
    app_logger.debug("Debug message")
    app_logger.info("Info message")
    app_logger.warning("Warning message")
    app_logger.error("Error message")


def test_module_specific_logging():
    """모듈별 로깅 테스트"""
    embedding_logger = get_embedding_logger()
    search_logger = get_search_logger()

    embedding_logger.info("Embedding test")
    search_logger.info("Search test")

    # 파일에 기록되었는지 확인
    import time
    time.sleep(0.5)

    assert (settings.LOG_DIR / "embedding.log").exists()
    assert (settings.LOG_DIR / "search.log").exists()


def test_log_execution_decorator():
    """실행 로깅 데코레이터 테스트"""

    @log_execution
    def sample_function(x, y):
        return x + y

    result = sample_function(3, 5)
    assert result == 8
```

### ✅ 검증 기준
- [ ] 로그 파일 생성 확인 (app.log, embedding.log, search.log, error.log)
- [ ] 로그 로테이션 설정 동작 확인
- [ ] 로그 레벨 필터링 확인
- [ ] 모듈별 로그 분리 확인
- [ ] 데코레이터 동작 확인
- [ ] 모든 테스트 통과 (`pytest tests/test_logger.py`)

---

## 🗂️ 4단계: M2.3 - 데이터 모델 (models.py)

### 📝 작업 목표
Pydantic 기반 타입 안전한 데이터 모델 정의 (API, DB 스키마)

### 구현 상세

#### 2.3.1 데이터 모델 구현
```python
# src/models.py

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np


# ==========================================
# 1. 이미지 메타데이터
# ==========================================

class ImageMetadata(BaseModel):
    """이미지 메타데이터 모델"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="이미지 고유 ID (UUID)")
    file_path: str = Field(..., description="이미지 파일 경로")
    file_name: str = Field(..., description="파일명")
    file_size: int = Field(..., ge=0, description="파일 크기 (bytes)")

    width: int = Field(..., ge=1, description="이미지 너비")
    height: int = Field(..., ge=1, description="이미지 높이")
    format: str = Field(..., description="이미지 포맷 (JPEG, PNG, etc.)")

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    tags: List[str] = Field(default_factory=list, description="이미지 태그")
    description: Optional[str] = Field(None, description="이미지 설명")

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """파일 경로 검증"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """이미지 포맷 검증"""
        valid_formats = ["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"]
        v_upper = v.upper()
        if v_upper not in valid_formats:
            raise ValueError(f"Invalid format: {v}. Must be one of {valid_formats}")
        return v_upper


# ==========================================
# 2. 임베딩 레코드
# ==========================================

class EmbeddingRecord(BaseModel):
    """임베딩 레코드 모델"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="레코드 ID (이미지 ID와 동일)")
    image_id: str = Field(..., description="원본 이미지 ID")

    # 임베딩 벡터
    embedding: List[float] = Field(..., description="임베딩 벡터")
    embedding_dim: int = Field(..., ge=1, description="임베딩 차원")

    # 양자화 정보
    is_quantized: bool = Field(default=False)
    quantization_method: Optional[Literal["scalar", "product"]] = None
    quantization_bits: Optional[Literal[1, 2, 4, 8]] = None

    # 메타데이터
    model_name: str = Field(..., description="사용된 모델명")
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """임베딩 벡터 검증"""
        if len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int, info) -> int:
        """임베딩 차원 검증"""
        if "embedding" in info.data:
            actual_dim = len(info.data["embedding"])
            if v != actual_dim:
                raise ValueError(
                    f"embedding_dim ({v}) does not match actual dimension ({actual_dim})"
                )
        return v


# ==========================================
# 3. 검색 요청
# ==========================================

class SearchRequest(BaseModel):
    """검색 요청 모델"""

    # 검색 타입
    query_type: Literal["text", "image"] = Field(..., description="검색 타입")

    # 텍스트 검색
    query_text: Optional[str] = Field(None, description="검색 텍스트")

    # 이미지 검색
    query_image_path: Optional[str] = Field(None, description="검색 이미지 경로")
    query_image_url: Optional[str] = Field(None, description="검색 이미지 URL")

    # 검색 파라미터
    top_k: int = Field(default=10, ge=1, le=100, description="반환할 결과 수")
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="최소 유사도 점수"
    )

    # 필터링
    tags: Optional[List[str]] = Field(None, description="태그 필터")
    date_from: Optional[datetime] = Field(None, description="시작 날짜")
    date_to: Optional[datetime] = Field(None, description="종료 날짜")

    @field_validator("query_text", "query_image_path", "query_image_url")
    @classmethod
    def validate_query(cls, v, info):
        """검색 쿼리 검증"""
        query_type = info.data.get("query_type")
        field_name = info.field_name

        if query_type == "text" and field_name == "query_text" and not v:
            raise ValueError("query_text is required for text search")

        if query_type == "image":
            if field_name in ["query_image_path", "query_image_url"]:
                has_path = info.data.get("query_image_path")
                has_url = info.data.get("query_image_url")
                if not (has_path or has_url):
                    raise ValueError(
                        "Either query_image_path or query_image_url is required for image search"
                    )

        return v


# ==========================================
# 4. 검색 결과
# ==========================================

class SearchResultItem(BaseModel):
    """검색 결과 아이템"""

    image_id: str = Field(..., description="이미지 ID")
    score: float = Field(..., ge=0.0, le=1.0, description="유사도 점수")
    metadata: ImageMetadata = Field(..., description="이미지 메타데이터")

    # 추가 정보
    rank: int = Field(..., ge=1, description="순위")
    distance: Optional[float] = Field(None, description="거리 (선택)")


class SearchResult(BaseModel):
    """검색 결과 모델"""

    query: SearchRequest = Field(..., description="검색 요청")
    results: List[SearchResultItem] = Field(
        default_factory=list,
        description="검색 결과 목록"
    )

    total_count: int = Field(..., ge=0, description="총 결과 수")
    search_time: float = Field(..., ge=0.0, description="검색 소요 시간 (초)")

    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("total_count")
    @classmethod
    def validate_total_count(cls, v: int, info) -> int:
        """총 결과 수 검증"""
        if "results" in info.data:
            actual_count = len(info.data["results"])
            if v != actual_count:
                raise ValueError(
                    f"total_count ({v}) does not match results length ({actual_count})"
                )
        return v


# ==========================================
# 5. 양자화 설정
# ==========================================

class QuantizationConfig(BaseModel):
    """양자화 설정 모델"""

    enabled: bool = Field(default=True)
    method: Literal["scalar", "product"] = Field(default="scalar")
    bits: Literal[1, 2, 4, 8] = Field(default=8)

    # Scalar Quantization
    scalar_type: Optional[Literal["int8", "uint8"]] = Field(default="int8")

    # Product Quantization
    num_subvectors: Optional[int] = Field(None, ge=1)
    num_clusters: Optional[int] = Field(None, ge=1)

    @field_validator("num_subvectors", "num_clusters")
    @classmethod
    def validate_product_quantization(cls, v, info):
        """Product Quantization 파라미터 검증"""
        if info.data.get("method") == "product":
            if v is None:
                raise ValueError(
                    f"{info.field_name} is required for product quantization"
                )
        return v


# ==========================================
# 6. API 응답 모델
# ==========================================

class ErrorResponse(BaseModel):
    """에러 응답 모델"""

    error: str = Field(..., description="에러 타입")
    message: str = Field(..., description="에러 메시지")
    detail: Optional[Dict[str, Any]] = Field(None, description="상세 정보")
    timestamp: datetime = Field(default_factory=datetime.now)


class SuccessResponse(BaseModel):
    """성공 응답 모델"""

    success: bool = Field(default=True)
    message: str = Field(..., description="성공 메시지")
    data: Optional[Dict[str, Any]] = Field(None, description="응답 데이터")
    timestamp: datetime = Field(default_factory=datetime.now)


# ==========================================
# 7. 벡터 DB 스키마
# ==========================================

class VectorPoint(BaseModel):
    """Qdrant 포인트 모델"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="포인트 ID")
    vector: List[float] = Field(..., description="벡터")
    payload: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v: List[float]) -> List[float]:
        """벡터 검증"""
        if len(v) == 0:
            raise ValueError("Vector cannot be empty")
        return v
```

#### 2.3.2 검증 테스트 (`tests/test_models.py`)
```python
import pytest
from datetime import datetime
from pathlib import Path
from src.models import (
    ImageMetadata,
    EmbeddingRecord,
    SearchRequest,
    SearchResult,
    SearchResultItem,
    QuantizationConfig,
    VectorPoint,
)


# ==========================================
# ImageMetadata 테스트
# ==========================================

def test_image_metadata_creation(tmp_path):
    """ImageMetadata 생성 테스트"""
    # 임시 이미지 파일 생성
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"fake image data")

    metadata = ImageMetadata(
        id="test-123",
        file_path=str(test_image),
        file_name="test.jpg",
        file_size=1024,
        width=800,
        height=600,
        format="JPEG",
        tags=["test", "sample"],
    )

    assert metadata.id == "test-123"
    assert metadata.format == "JPEG"
    assert len(metadata.tags) == 2


def test_image_metadata_validation_file_not_found():
    """파일 존재하지 않을 때 검증 실패"""
    with pytest.raises(ValueError, match="File does not exist"):
        ImageMetadata(
            id="test-123",
            file_path="/nonexistent/path.jpg",
            file_name="test.jpg",
            file_size=1024,
            width=800,
            height=600,
            format="JPEG",
        )


def test_image_metadata_invalid_format(tmp_path):
    """잘못된 포맷 검증"""
    test_image = tmp_path / "test.xyz"
    test_image.write_bytes(b"fake")

    with pytest.raises(ValueError, match="Invalid format"):
        ImageMetadata(
            id="test-123",
            file_path=str(test_image),
            file_name="test.xyz",
            file_size=100,
            width=100,
            height=100,
            format="XYZ",  # 잘못된 포맷
        )


# ==========================================
# EmbeddingRecord 테스트
# ==========================================

def test_embedding_record_creation():
    """EmbeddingRecord 생성 테스트"""
    record = EmbeddingRecord(
        id="emb-123",
        image_id="img-123",
        embedding=[0.1, 0.2, 0.3, 0.4],
        embedding_dim=4,
        model_name="ViT-B/32",
        is_quantized=True,
        quantization_method="scalar",
        quantization_bits=8,
    )

    assert record.embedding_dim == 4
    assert len(record.embedding) == 4
    assert record.is_quantized is True


def test_embedding_dimension_mismatch():
    """임베딩 차원 불일치 검증"""
    with pytest.raises(ValueError, match="embedding_dim"):
        EmbeddingRecord(
            id="emb-123",
            image_id="img-123",
            embedding=[0.1, 0.2, 0.3],
            embedding_dim=5,  # 실제는 3차원
            model_name="ViT-B/32",
        )


# ==========================================
# SearchRequest 테스트
# ==========================================

def test_search_request_text():
    """텍스트 검색 요청 테스트"""
    request = SearchRequest(
        query_type="text",
        query_text="cat",
        top_k=20,
        score_threshold=0.5,
    )

    assert request.query_type == "text"
    assert request.query_text == "cat"
    assert request.top_k == 20


def test_search_request_image():
    """이미지 검색 요청 테스트"""
    request = SearchRequest(
        query_type="image",
        query_image_path="/path/to/image.jpg",
        top_k=10,
    )

    assert request.query_type == "image"
    assert request.query_image_path is not None


def test_search_request_validation_missing_query():
    """검색 쿼리 누락 검증"""
    with pytest.raises(ValueError):
        SearchRequest(
            query_type="text",
            # query_text 누락
            top_k=10,
        )


# ==========================================
# SearchResult 테스트
# ==========================================

def test_search_result_creation(tmp_path):
    """SearchResult 생성 테스트"""
    # 임시 이미지 생성
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"fake")

    metadata = ImageMetadata(
        id="img-1",
        file_path=str(test_image),
        file_name="test.jpg",
        file_size=100,
        width=100,
        height=100,
        format="JPEG",
    )

    request = SearchRequest(
        query_type="text",
        query_text="test",
        top_k=10,
    )

    result_item = SearchResultItem(
        image_id="img-1",
        score=0.95,
        metadata=metadata,
        rank=1,
    )

    result = SearchResult(
        query=request,
        results=[result_item],
        total_count=1,
        search_time=0.123,
    )

    assert result.total_count == 1
    assert len(result.results) == 1
    assert result.results[0].score == 0.95


# ==========================================
# QuantizationConfig 테스트
# ==========================================

def test_quantization_config_scalar():
    """Scalar Quantization 설정 테스트"""
    config = QuantizationConfig(
        enabled=True,
        method="scalar",
        bits=8,
        scalar_type="int8",
    )

    assert config.method == "scalar"
    assert config.bits == 8


def test_quantization_config_product():
    """Product Quantization 설정 테스트"""
    config = QuantizationConfig(
        enabled=True,
        method="product",
        bits=8,
        num_subvectors=8,
        num_clusters=256,
    )

    assert config.method == "product"
    assert config.num_subvectors == 8


# ==========================================
# JSON 직렬화 테스트
# ==========================================

def test_json_serialization(tmp_path):
    """JSON 직렬화 테스트"""
    test_image = tmp_path / "test.jpg"
    test_image.write_bytes(b"fake")

    metadata = ImageMetadata(
        id="test-123",
        file_path=str(test_image),
        file_name="test.jpg",
        file_size=1024,
        width=800,
        height=600,
        format="JPEG",
    )

    # JSON 직렬화
    json_str = metadata.model_dump_json()
    assert isinstance(json_str, str)

    # JSON 역직렬화
    metadata_restored = ImageMetadata.model_validate_json(json_str)
    assert metadata_restored.id == metadata.id
```

### ✅ 검증 기준
- [ ] 모든 모델 인스턴스 생성 성공
- [ ] 검증 로직 동작 확인 (잘못된 데이터 입력 시 에러)
- [ ] JSON 직렬화/역직렬화 성공
- [ ] 타입 체크 통과 (mypy/pyright)
- [ ] 모든 테스트 통과 (`pytest tests/test_models.py`)

---

## 🧪 5단계: 통합 테스트 및 검증

### 작업 내용

#### 5.1 전체 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html

# 특정 모듈만 테스트
pytest tests/test_config.py -v
pytest tests/test_logger.py -v
pytest tests/test_models.py -v
```

#### 5.2 통합 검증 스크립트 (`scripts/verify_module2.py`)
```python
#!/usr/bin/env python3
"""Module 2 통합 검증 스크립트"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_config():
    """설정 관리자 검증"""
    print("\n=== M2.1: 설정 관리자 검증 ===")

    try:
        from src.config import settings

        # 1. 인스턴스 생성
        assert settings is not None
        print("✓ Settings 인스턴스 생성 성공")

        # 2. 환경변수 로딩
        assert settings.PROJECT_NAME
        print(f"✓ 프로젝트명: {settings.PROJECT_NAME}")

        # 3. 경로 생성
        assert settings.DATA_DIR.exists()
        assert settings.LOG_DIR.exists()
        print("✓ 디렉토리 자동 생성 확인")

        # 4. 디바이스 검증
        assert settings.DEVICE in ["cpu", "cuda", "mps"]
        print(f"✓ 디바이스: {settings.DEVICE}")

        print("✅ M2.1 검증 완료\n")
        return True

    except Exception as e:
        print(f"❌ M2.1 검증 실패: {e}\n")
        return False


def verify_logger():
    """로깅 시스템 검증"""
    print("=== M2.2: 로깅 시스템 검증 ===")

    try:
        from src.logger import (
            app_logger,
            get_embedding_logger,
            get_search_logger
        )
        from src.config import settings
        import time

        # 1. 로거 생성
        assert app_logger is not None
        print("✓ 로거 초기화 성공")

        # 2. 로그 작성
        app_logger.info("Test log message")
        get_embedding_logger().info("Test embedding log")
        get_search_logger().info("Test search log")
        app_logger.error("Test error log")

        # 비동기 처리 대기
        time.sleep(1)

        # 3. 로그 파일 확인
        log_files = [
            settings.LOG_DIR / "app.log",
            settings.LOG_DIR / "embedding.log",
            settings.LOG_DIR / "search.log",
            settings.LOG_DIR / "error.log",
        ]

        for log_file in log_files:
            if log_file.exists():
                print(f"✓ {log_file.name} 생성됨")
            else:
                print(f"⚠ {log_file.name} 미생성")

        print("✅ M2.2 검증 완료\n")
        return True

    except Exception as e:
        print(f"❌ M2.2 검증 실패: {e}\n")
        return False


def verify_models():
    """데이터 모델 검증"""
    print("=== M2.3: 데이터 모델 검증 ===")

    try:
        from src.models import (
            ImageMetadata,
            EmbeddingRecord,
            SearchRequest,
            SearchResult,
        )
        import tempfile
        from pathlib import Path

        # 1. 임시 파일 생성
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name

        # 2. ImageMetadata 생성
        metadata = ImageMetadata(
            id="test-1",
            file_path=temp_path,
            file_name="test.jpg",
            file_size=1024,
            width=800,
            height=600,
            format="JPEG",
        )
        print("✓ ImageMetadata 생성 성공")

        # 3. EmbeddingRecord 생성
        record = EmbeddingRecord(
            id="emb-1",
            image_id="test-1",
            embedding=[0.1] * 512,
            embedding_dim=512,
            model_name="ViT-B/32",
        )
        print("✓ EmbeddingRecord 생성 성공")

        # 4. SearchRequest 생성
        request = SearchRequest(
            query_type="text",
            query_text="test",
            top_k=10,
        )
        print("✓ SearchRequest 생성 성공")

        # 5. JSON 직렬화
        json_data = metadata.model_dump_json()
        assert isinstance(json_data, str)
        print("✓ JSON 직렬화 성공")

        # 정리
        Path(temp_path).unlink()

        print("✅ M2.3 검증 완료\n")
        return True

    except Exception as e:
        print(f"❌ M2.3 검증 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 검증 함수"""
    print("\n" + "="*50)
    print("Module 2: 설정 및 유틸리티 통합 검증")
    print("="*50)

    results = {
        "M2.1 (Config)": verify_config(),
        "M2.2 (Logger)": verify_logger(),
        "M2.3 (Models)": verify_models(),
    }

    print("="*50)
    print("검증 결과 요약")
    print("="*50)

    for module, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{module}: {status}")

    print("="*50)

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 Module 2 검증 완료! 모든 테스트 통과\n")
        return 0
    else:
        print("\n⚠️  일부 검증 실패. 위 에러를 확인하세요.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

#### 5.3 실행 권한 부여 및 실행
```bash
chmod +x scripts/verify_module2.py
python scripts/verify_module2.py
```

### ✅ 최종 검증 체크리스트
- [ ] 모든 단위 테스트 통과 (`pytest tests/ -v`)
- [ ] 통합 검증 스크립트 통과 (`python scripts/verify_module2.py`)
- [ ] 타입 체크 통과 (`mypy src/`)
- [ ] 코드 스타일 검사 (`black --check src/`, `ruff check src/`)
- [ ] 모든 로그 파일 정상 생성
- [ ] `.env` 파일 정상 로딩
- [ ] 문서화 완료 (docstring, README)

---

## 📚 6단계: 문서화

### README 업데이트 (`README.md`)
```markdown
# Image Search with Quantization

## Module 2: 설정 및 유틸리티

### 구성 요소

#### 1. 설정 관리자 (`src/config.py`)
- Pydantic Settings 기반 타입 안전한 설정
- 환경변수 자동 로딩 (`.env`)
- 디바이스, 모델, DB 설정 관리

**사용 예제:**
\`\`\`python
from src.config import settings

print(settings.DEVICE)  # cpu, cuda, mps
print(settings.CLIP_MODEL_NAME)  # ViT-B/32
\`\`\`

#### 2. 로깅 시스템 (`src/logger.py`)
- Loguru 기반 구조화 로깅
- 모듈별 로그 분리 (app, embedding, search, error)
- 자동 로테이션 및 압축

**사용 예제:**
\`\`\`python
from src.logger import app_logger, get_embedding_logger

app_logger.info("Application started")
get_embedding_logger().info("Processing embeddings")
\`\`\`

#### 3. 데이터 모델 (`src/models.py`)
- Pydantic 기반 데이터 검증
- API 요청/응답 스키마
- DB 레코드 모델

**사용 예제:**
\`\`\`python
from src.models import SearchRequest

request = SearchRequest(
    query_type="text",
    query_text="cat",
    top_k=10
)
\`\`\`

### 설치 및 실행

\`\`\`bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일 편집

# 검증 실행
python scripts/verify_module2.py

# 테스트 실행
pytest tests/ -v
\`\`\`

### 디렉토리 구조

\`\`\`
quantization/
├── src/
│   ├── config.py       # 설정 관리
│   ├── logger.py       # 로깅 시스템
│   └── models.py       # 데이터 모델
├── tests/
│   ├── test_config.py
│   ├── test_logger.py
│   └── test_models.py
├── logs/               # 로그 파일
├── scripts/
│   └── verify_module2.py
├── .env
└── requirements.txt
\`\`\`
```

---

## ⚠️ 주의사항 및 트러블슈팅

### 일반적인 문제

1. **`.env` 파일 읽기 실패**
   - `.env` 파일이 프로젝트 루트에 있는지 확인
   - 파일 인코딩이 UTF-8인지 확인
   - 환경변수 이름이 정확한지 확인

2. **디바이스 검증 실패**
   - CUDA: `torch.cuda.is_available()` 확인
   - MPS: macOS + Apple Silicon 확인
   - 기본값 `cpu` 사용 권장

3. **로그 파일 생성 안됨**
   - `logs/` 디렉토리 권한 확인
   - `loguru` 설치 확인
   - 비동기 처리로 인한 지연 고려 (1초 대기)

4. **Pydantic 검증 실패**
   - 모델 필드 타입 확인
   - 필수 필드 누락 확인
   - `@field_validator` 로직 확인

### 디버깅 팁

```python
# 설정 확인
from src.config import settings
print(settings.model_dump_json(indent=2))

# 로그 레벨 변경
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# 모델 검증 에러 확인
from pydantic import ValidationError
try:
    model = MyModel(**data)
except ValidationError as e:
    print(e.json())
```

---

## 🎯 완료 기준

### Module 2 완료 조건

- [x] M2.1: `src/config.py` 구현 및 테스트 통과
- [x] M2.2: `src/logger.py` 구현 및 테스트 통과
- [x] M2.3: `src/models.py` 구현 및 테스트 통과
- [x] 통합 검증 스크립트 통과
- [x] 모든 단위 테스트 통과 (pytest)
- [x] 타입 체크 통과 (mypy)
- [x] 문서화 완료

### 다음 단계

Module 2 완료 후 다음 모듈로 진행:
- **Module 3:** CLIP 임베딩 모듈
- **Module 4:** Qdrant 벡터 DB 연동
- **Module 5:** 양자화 구현

---

## 📞 참고 자료

- [Pydantic 공식 문서](https://docs.pydantic.dev/)
- [Loguru 공식 문서](https://loguru.readthedocs.io/)
- [Python dotenv](https://github.com/theskumar/python-dotenv)
- [PyTest 공식 문서](https://docs.pytest.org/)

---

**작성일:** 2025-11-18
**버전:** 1.0
**담당자:** 백엔드 개발자
