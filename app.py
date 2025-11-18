"""
Streamlit Frontend for Image Search with Quantization
"""

import streamlit as st
import requests
from pathlib import Path
from typing import Optional
import json

# 페이지 설정
st.set_page_config(
    page_title="Image Search with Quantization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API 설정
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """API 헬스 체크"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def search_by_text(query_text: str, top_k: int, score_threshold: float):
    """텍스트 기반 이미지 검색"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search/text",
            data={
                "query_text": query_text,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"검색 실패: {str(e)}")
        return None


def search_by_image(image_file, top_k: int, score_threshold: float):
    """이미지 기반 이미지 검색"""
    try:
        files = {"image": image_file}
        data = {
            "top_k": top_k,
            "score_threshold": score_threshold,
        }
        response = requests.post(
            f"{API_BASE_URL}/search/image",
            files=files,
            data=data,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"검색 실패: {str(e)}")
        return None


def upload_image(image_file, tags: Optional[str], description: Optional[str]):
    """이미지 업로드 및 인덱싱"""
    try:
        files = {"image": image_file}
        data = {}
        if tags:
            data["tags"] = tags
        if description:
            data["description"] = description

        response = requests.post(
            f"{API_BASE_URL}/images/upload",
            files=files,
            data=data,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"업로드 실패: {str(e)}")
        return None


def get_config():
    """API 설정 조회"""
    try:
        response = requests.get(f"{API_BASE_URL}/config", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"설정 조회 실패: {str(e)}")
        return None


# ==========================================
# 메인 UI
# ==========================================

def main():
    st.title("🔍 Image Search with Quantization")
    st.markdown("CLIP 임베딩과 양자화를 활용한 이미지 검색 시스템")

    # API 상태 확인
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API 연결됨")
    else:
        st.sidebar.error("❌ API 연결 실패")
        st.error(
            "백엔드 API가 실행되지 않았습니다. "
            "터미널에서 다음 명령을 실행하세요:\n\n"
            "```bash\npython -m src.api.main\n```"
        )
        return

    # 사이드바: 설정
    st.sidebar.header("⚙️ 설정")

    # API 설정 표시
    config = get_config()
    if config:
        with st.sidebar.expander("🔧 API 설정", expanded=False):
            st.json(config)

    # 검색 파라미터
    st.sidebar.subheader("검색 파라미터")
    top_k = st.sidebar.slider("결과 개수 (Top K)", 1, 50, 10)
    score_threshold = st.sidebar.slider("최소 유사도", 0.0, 1.0, 0.0, 0.05)

    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["📝 텍스트 검색", "🖼️ 이미지 검색", "⬆️ 이미지 업로드"])

    # ==========================================
    # 탭 1: 텍스트 검색
    # ==========================================
    with tab1:
        st.header("📝 텍스트로 이미지 검색")
        st.markdown("텍스트 설명을 입력하면 유사한 이미지를 찾아드립니다.")

        query_text = st.text_input(
            "검색어를 입력하세요",
            placeholder="예: a cat sitting on a sofa",
        )

        if st.button("🔍 검색", key="text_search", type="primary"):
            if not query_text:
                st.warning("검색어를 입력해주세요.")
            else:
                with st.spinner("검색 중..."):
                    result = search_by_text(query_text, top_k, score_threshold)

                if result:
                    st.success(
                        f"검색 완료! {result['total_count']}개 결과 "
                        f"({result['search_time']:.3f}초)"
                    )

                    # 검색 결과 표시
                    if result['total_count'] > 0:
                        st.subheader("검색 결과")
                        cols = st.columns(3)
                        for idx, item in enumerate(result['results']):
                            col = cols[idx % 3]
                            with col:
                                st.image(
                                    item['metadata']['file_path'],
                                    caption=f"Score: {item['score']:.3f}",
                                    use_container_width=True,
                                )
                                st.text(f"Rank: {item['rank']}")
                    else:
                        st.info("검색 결과가 없습니다.")

    # ==========================================
    # 탭 2: 이미지 검색
    # ==========================================
    with tab2:
        st.header("🖼️ 이미지로 유사 이미지 검색")
        st.markdown("이미지를 업로드하면 유사한 이미지를 찾아드립니다.")

        uploaded_image = st.file_uploader(
            "이미지를 업로드하세요",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_search_upload",
        )

        if uploaded_image:
            st.image(
                uploaded_image,
                caption="검색 이미지",
                width=300,
            )

            if st.button("🔍 유사 이미지 검색", key="image_search", type="primary"):
                with st.spinner("검색 중..."):
                    # 파일 포인터를 처음으로 되돌림
                    uploaded_image.seek(0)
                    result = search_by_image(
                        uploaded_image,
                        top_k,
                        score_threshold,
                    )

                if result:
                    st.success(
                        f"검색 완료! {result['total_count']}개 결과 "
                        f"({result['search_time']:.3f}초)"
                    )

                    # 검색 결과 표시
                    if result['total_count'] > 0:
                        st.subheader("유사 이미지")
                        cols = st.columns(3)
                        for idx, item in enumerate(result['results']):
                            col = cols[idx % 3]
                            with col:
                                st.image(
                                    item['metadata']['file_path'],
                                    caption=f"Score: {item['score']:.3f}",
                                    use_container_width=True,
                                )
                                st.text(f"Rank: {item['rank']}")
                    else:
                        st.info("유사한 이미지가 없습니다.")

    # ==========================================
    # 탭 3: 이미지 업로드
    # ==========================================
    with tab3:
        st.header("⬆️ 이미지 업로드 및 인덱싱")
        st.markdown("이미지를 업로드하여 검색 데이터베이스에 추가합니다.")

        upload_file = st.file_uploader(
            "업로드할 이미지를 선택하세요",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_upload",
        )

        if upload_file:
            st.image(
                upload_file,
                caption="업로드할 이미지",
                width=300,
            )

            # 메타데이터 입력
            col1, col2 = st.columns(2)
            with col1:
                tags = st.text_input(
                    "태그 (쉼표로 구분)",
                    placeholder="예: cat, animal, pet",
                )
            with col2:
                description = st.text_area(
                    "설명",
                    placeholder="이미지에 대한 설명을 입력하세요",
                )

            if st.button("⬆️ 업로드 및 인덱싱", key="upload_button", type="primary"):
                with st.spinner("업로드 중..."):
                    # 파일 포인터를 처음으로 되돌림
                    upload_file.seek(0)
                    result = upload_image(
                        upload_file,
                        tags if tags else None,
                        description if description else None,
                    )

                if result and result.get("success"):
                    st.success("✅ 이미지가 성공적으로 업로드되고 인덱싱되었습니다!")
                    st.json(result["data"])

    # ==========================================
    # 푸터
    # ==========================================
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### 📚 사용 방법
        1. **텍스트 검색**: 텍스트 설명으로 이미지 검색
        2. **이미지 검색**: 이미지로 유사 이미지 찾기
        3. **이미지 업로드**: 새 이미지를 DB에 추가

        ### 🔧 기술 스택
        - Frontend: Streamlit
        - Backend: FastAPI
        - Embedding: CLIP (OpenAI)
        - Vector DB: Qdrant
        - Quantization: Scalar/Product
        """
    )


if __name__ == "__main__":
    main()
