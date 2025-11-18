"""설정 관리자 테스트 모듈"""

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
