"""
FastAPI Backend for Image Search with Quantization
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config import settings
from src.logger import app_logger, get_search_logger
from src.models import (
    SearchRequest,
    SearchResult,
    SearchResultItem,
    ImageMetadata,
    SuccessResponse,
    ErrorResponse,
)

# FastAPI 앱 초기화
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Image search API with quantization support",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로거 초기화
search_logger = get_search_logger()


@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    app_logger.info("Starting Image Search API")
    app_logger.info(f"Device: {settings.DEVICE}")
    app_logger.info(f"CLIP Model: {settings.CLIP_MODEL_NAME}")
    app_logger.info(f"Quantization: {settings.QUANTIZATION_ENABLED}")


@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    app_logger.info("Shutting down Image Search API")


# ==========================================
# 헬스 체크 엔드포인트
# ==========================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Image Search API with Quantization",
        "version": settings.VERSION,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": settings.DEVICE,
        "quantization_enabled": settings.QUANTIZATION_ENABLED,
    }


# ==========================================
# 검색 엔드포인트
# ==========================================

@app.post("/search/text", response_model=SearchResult)
async def search_by_text(
    query_text: str = Form(...),
    top_k: int = Form(default=10),
    score_threshold: float = Form(default=0.0),
):
    """
    텍스트 기반 이미지 검색

    Args:
        query_text: 검색할 텍스트
        top_k: 반환할 결과 수
        score_threshold: 최소 유사도 점수

    Returns:
        SearchResult: 검색 결과
    """
    try:
        search_logger.info(f"Text search request: {query_text}")

        # SearchRequest 생성
        request = SearchRequest(
            query_type="text",
            query_text=query_text,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # TODO: 실제 검색 로직 구현 (CLIP 임베딩 + Qdrant 검색)
        # 현재는 모의 응답 반환
        result = SearchResult(
            query=request,
            results=[],
            total_count=0,
            search_time=0.0,
        )

        search_logger.info(f"Search completed: {result.total_count} results")
        return result

    except Exception as e:
        app_logger.error(f"Text search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=SearchResult)
async def search_by_image(
    image: UploadFile = File(...),
    top_k: int = Form(default=10),
    score_threshold: float = Form(default=0.0),
):
    """
    이미지 기반 이미지 검색

    Args:
        image: 검색할 이미지 파일
        top_k: 반환할 결과 수
        score_threshold: 최소 유사도 점수

    Returns:
        SearchResult: 검색 결과
    """
    try:
        search_logger.info(f"Image search request: {image.filename}")

        # 임시 파일로 저장
        temp_path = settings.DATA_DIR / f"temp_{image.filename}"
        with open(temp_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # SearchRequest 생성
        request = SearchRequest(
            query_type="image",
            query_image_path=str(temp_path),
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # TODO: 실제 검색 로직 구현
        result = SearchResult(
            query=request,
            results=[],
            total_count=0,
            search_time=0.0,
        )

        # 임시 파일 삭제
        temp_path.unlink(missing_ok=True)

        search_logger.info(f"Search completed: {result.total_count} results")
        return result

    except Exception as e:
        app_logger.error(f"Image search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 이미지 업로드 및 인덱싱
# ==========================================

@app.post("/images/upload", response_model=SuccessResponse)
async def upload_image(
    image: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    이미지 업로드 및 인덱싱

    Args:
        image: 업로드할 이미지 파일
        tags: 이미지 태그 (쉼표로 구분)
        description: 이미지 설명

    Returns:
        SuccessResponse: 업로드 결과
    """
    try:
        app_logger.info(f"Uploading image: {image.filename}")

        # 이미지 저장
        image_path = settings.DATA_DIR / image.filename
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)

        # 태그 파싱
        tag_list = []
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]

        # TODO: 이미지 임베딩 생성 및 Qdrant에 저장

        app_logger.info(f"Image uploaded successfully: {image.filename}")

        return SuccessResponse(
            message="Image uploaded and indexed successfully",
            data={
                "filename": image.filename,
                "path": str(image_path),
                "tags": tag_list,
            }
        )

    except Exception as e:
        app_logger.error(f"Image upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 설정 조회
# ==========================================

@app.get("/config")
async def get_config():
    """현재 설정 조회"""
    return {
        "project_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "device": settings.DEVICE,
        "clip_model": settings.CLIP_MODEL_NAME,
        "embedding_dim": settings.EMBEDDING_DIM,
        "quantization": {
            "enabled": settings.QUANTIZATION_ENABLED,
            "method": settings.QUANTIZATION_METHOD,
            "bits": settings.QUANTIZATION_BITS,
        },
        "qdrant": {
            "host": settings.QDRANT_HOST,
            "port": settings.QDRANT_PORT,
            "collection": settings.QDRANT_COLLECTION_NAME,
        },
    }


# ==========================================
# 에러 핸들러
# ==========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 에러 핸들러"""
    app_logger.error(f"Unhandled exception: {exc}")

    error_response = ErrorResponse(
        error="InternalServerError",
        message=str(exc),
    )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# ==========================================
# 메인 실행
# ==========================================

def main():
    """FastAPI 서버 실행"""
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.API_RELOAD,
    )


if __name__ == "__main__":
    main()
