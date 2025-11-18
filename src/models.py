# src/models.py

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


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

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

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

    @model_validator(mode='after')
    def validate_query(self):
        """검색 쿼리 검증"""
        if self.query_type == "text":
            if not self.query_text:
                raise ValueError("query_text is required for text search")

        if self.query_type == "image":
            if not self.query_image_path and not self.query_image_url:
                raise ValueError(
                    "Either query_image_path or query_image_url is required for image search"
                )

        return self


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
