"""로깅 시스템 테스트 모듈"""

import pytest
from pathlib import Path
from src.logger import (
    setup_logger,
    get_embedding_logger,
    get_search_logger,
    get_quantization_logger,
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
    quantization_logger = get_quantization_logger()

    embedding_logger.info("Embedding test")
    search_logger.info("Search test")
    quantization_logger.info("Quantization test")

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


def test_log_execution_decorator_with_exception():
    """예외 발생 시 데코레이터 동작 테스트"""

    @log_execution
    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        failing_function()


def test_logger_bindings():
    """로거 바인딩 테스트"""
    embedding_logger = get_embedding_logger()
    search_logger = get_search_logger()

    # 바인딩된 컨텍스트가 있는지 확인
    assert embedding_logger is not None
    assert search_logger is not None


def test_log_directory_exists():
    """로그 디렉토리 존재 확인"""
    assert settings.LOG_DIR.exists()
    assert settings.LOG_DIR.is_dir()


@pytest.mark.asyncio
async def test_async_log_execution_decorator():
    """비동기 함수용 실행 로깅 데코레이터 테스트"""
    from src.logger import log_execution_async

    @log_execution_async
    async def async_sample_function(x, y):
        return x * y

    result = await async_sample_function(4, 5)
    assert result == 20
