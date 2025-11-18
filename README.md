# Visual Search MVP with Quantization

Progressive Quantization을 활용한 이미지 시각 검색 시스템

## 프로젝트 개요

이 프로젝트는 CLIP 모델과 양자화 기술을 활용하여 효율적인 이미지 검색 시스템을 구축합니다.

### 주요 기능

- 🔍 이미지 유사도 검색
- ⚡ Progressive Quantization을 통한 성능 최적화
- 🎨 Streamlit 기반 웹 UI
- 🚀 FastAPI REST API
- 📦 Qdrant Vector Database

## 시스템 아키텍처

```
Layer 1: 사용자 인터페이스
  └─ 웹 UI (Streamlit) + REST API (FastAPI)

Layer 2: 비즈니스 로직
  └─ 이미지 처리 파이프라인 + 검색 로직

Layer 3: AI/ML 엔진
  └─ 임베딩 생성 (CLIP) + 양자화 엔진

Layer 4: 데이터 저장소
  └─ Vector Database (Qdrant) + 파일 시스템

Layer 5: 인프라
  └─ uv + Docker + 설정 관리
```

## 프로젝트 구조

```
visual-search-mvp/
├── app/          # 메인 애플리케이션
├── scripts/      # 유틸리티 스크립트
├── tests/        # 테스트 코드
├── data/         # 데이터 저장소
└── logs/         # 로그 파일
```

## 개발 환경 설정

### 필수 요구사항

- Python 3.10+
- uv (패키지 관리자)
- Docker (선택사항)

### 설치 방법

```bash
# 1. 저장소 클론
git clone <repository-url>
cd Quantization

# 2. 의존성 설치
uv sync

# 3. 개발 도구 설치
uv sync --extra dev
```

## 사용 기술

### Core
- **CLIP**: Open-source CLIP 모델 (open-clip-torch)
- **Qdrant**: Vector Database
- **PyTorch**: 딥러닝 프레임워크

### API & UI
- **FastAPI**: REST API 서버
- **Streamlit**: 웹 UI

### Development
- **uv**: 빠른 패키지 관리
- **pytest**: 테스트 프레임워크
- **black/ruff**: 코드 포맷팅 및 린팅

## 라이선스

MIT License
