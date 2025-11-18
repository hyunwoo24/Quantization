"""
FastAPI Backend Tests
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
from io import BytesIO

from src.api.main import app


@pytest.fixture
def client():
    """TestClient fixture"""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """샘플 이미지 파일 생성"""
    # 간단한 1x1 픽셀 PNG 이미지 (바이트 데이터)
    png_data = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01'
        b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return BytesIO(png_data)


# ==========================================
# 헬스 체크 테스트
# ==========================================

def test_root_endpoint(client):
    """루트 엔드포인트 테스트"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_check(client):
    """헬스 체크 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "device" in data
    assert "quantization_enabled" in data


def test_config_endpoint(client):
    """설정 조회 테스트"""
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "project_name" in data
    assert "version" in data
    assert "device" in data
    assert "clip_model" in data
    assert "quantization" in data
    assert "qdrant" in data


# ==========================================
# 검색 엔드포인트 테스트
# ==========================================

def test_search_by_text(client):
    """텍스트 검색 테스트"""
    response = client.post(
        "/search/text",
        data={
            "query_text": "a cat",
            "top_k": 10,
            "score_threshold": 0.0,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert "total_count" in data
    assert "search_time" in data
    assert data["query"]["query_type"] == "text"
    assert data["query"]["query_text"] == "a cat"


def test_search_by_text_missing_query(client):
    """텍스트 검색 - 쿼리 누락 테스트"""
    response = client.post(
        "/search/text",
        data={
            "top_k": 10,
        }
    )
    # FastAPI는 필수 필드 누락 시 422 Unprocessable Entity 반환
    assert response.status_code == 422


def test_search_by_image(client, sample_image):
    """이미지 검색 테스트"""
    response = client.post(
        "/search/image",
        files={"image": ("test.png", sample_image, "image/png")},
        data={
            "top_k": 10,
            "score_threshold": 0.0,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert data["query"]["query_type"] == "image"


def test_search_by_image_missing_file(client):
    """이미지 검색 - 파일 누락 테스트"""
    response = client.post(
        "/search/image",
        data={
            "top_k": 10,
        }
    )
    assert response.status_code == 422


# ==========================================
# 이미지 업로드 테스트
# ==========================================

def test_upload_image(client, sample_image):
    """이미지 업로드 테스트"""
    response = client.post(
        "/images/upload",
        files={"image": ("test.png", sample_image, "image/png")},
        data={
            "tags": "test, sample",
            "description": "Test image",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "message" in data
    assert "data" in data
    assert data["data"]["filename"] == "test.png"


def test_upload_image_without_metadata(client, sample_image):
    """이미지 업로드 - 메타데이터 없이"""
    response = client.post(
        "/images/upload",
        files={"image": ("test2.png", sample_image, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_upload_image_missing_file(client):
    """이미지 업로드 - 파일 누락 테스트"""
    response = client.post(
        "/images/upload",
        data={
            "tags": "test",
        }
    )
    assert response.status_code == 422


# ==========================================
# 파라미터 검증 테스트
# ==========================================

def test_search_text_with_custom_params(client):
    """텍스트 검색 - 커스텀 파라미터 테스트"""
    response = client.post(
        "/search/text",
        data={
            "query_text": "dog",
            "top_k": 20,
            "score_threshold": 0.5,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"]["top_k"] == 20
    assert data["query"]["score_threshold"] == 0.5


def test_search_image_with_custom_params(client, sample_image):
    """이미지 검색 - 커스텀 파라미터 테스트"""
    response = client.post(
        "/search/image",
        files={"image": ("test.png", sample_image, "image/png")},
        data={
            "top_k": 5,
            "score_threshold": 0.7,
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"]["top_k"] == 5
    assert data["query"]["score_threshold"] == 0.7
