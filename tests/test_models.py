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
