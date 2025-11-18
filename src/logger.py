"""로깅 시스템 모듈

Loguru 기반의 구조화된 로깅 시스템
- 레벨별, 모듈별 로그 분리
- 자동 로테이션 및 압축
- 비동기 처리
"""

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


def log_execution_async(func: Callable) -> Callable:
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
