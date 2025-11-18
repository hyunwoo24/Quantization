#!/usr/bin/env python3
"""
Frontend and Backend Implementation Verification Script
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_requirements():
    """requirements.txt 검증"""
    print("\n=== Requirements 검증 ===")

    req_file = Path(__file__).parent.parent / "requirements.txt"
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False

    content = req_file.read_text()

    # 필수 패키지 확인
    required_packages = {
        "fastapi": "FastAPI (Backend)",
        "uvicorn": "Uvicorn (ASGI Server)",
        "streamlit": "Streamlit (Frontend)",
    }

    all_found = True
    for package, description in required_packages.items():
        if package in content.lower():
            print(f"✓ {description} 포함됨")
        else:
            print(f"❌ {description} 누락됨")
            all_found = False

    return all_found


def verify_backend():
    """백엔드 구현 검증"""
    print("\n=== 백엔드 (FastAPI) 검증 ===")

    try:
        from src.api.main import app
        print("✓ FastAPI 앱 import 성공")

        # 라우트 확인
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/",
            "/health",
            "/config",
            "/search/text",
            "/search/image",
            "/images/upload",
        ]

        all_routes_found = True
        for route in expected_routes:
            if route in routes:
                print(f"✓ 라우트 '{route}' 존재")
            else:
                print(f"❌ 라우트 '{route}' 누락")
                all_routes_found = False

        return all_routes_found

    except Exception as e:
        print(f"❌ 백엔드 검증 실패: {e}")
        return False


def verify_frontend():
    """프론트엔드 구현 검증"""
    print("\n=== 프론트엔드 (Streamlit) 검증 ===")

    app_file = Path(__file__).parent.parent / "app.py"

    if not app_file.exists():
        print("❌ app.py not found")
        return False

    print("✓ app.py 존재")

    content = app_file.read_text()

    # 주요 기능 확인
    features = {
        "import streamlit": "Streamlit import",
        "search_by_text": "텍스트 검색 함수",
        "search_by_image": "이미지 검색 함수",
        "upload_image": "이미지 업로드 함수",
        "st.tabs": "탭 UI",
        "API_BASE_URL": "API 연결 설정",
    }

    all_features_found = True
    for feature, description in features.items():
        if feature in content:
            print(f"✓ {description} 구현됨")
        else:
            print(f"❌ {description} 누락됨")
            all_features_found = False

    return all_features_found


def verify_tests():
    """테스트 검증"""
    print("\n=== 테스트 검증 ===")

    test_file = Path(__file__).parent.parent / "tests" / "test_api.py"

    if not test_file.exists():
        print("❌ test_api.py not found")
        return False

    print("✓ test_api.py 존재")

    content = test_file.read_text()

    # 테스트 함수 확인
    test_functions = [
        "test_root_endpoint",
        "test_health_check",
        "test_search_by_text",
        "test_search_by_image",
        "test_upload_image",
    ]

    all_tests_found = True
    for test_func in test_functions:
        if test_func in content:
            print(f"✓ {test_func} 구현됨")
        else:
            print(f"❌ {test_func} 누락됨")
            all_tests_found = False

    return all_tests_found


def main():
    """메인 검증 함수"""
    print("\n" + "="*50)
    print("Frontend & Backend Implementation Verification")
    print("="*50)

    results = {
        "Requirements": verify_requirements(),
        "Backend (FastAPI)": verify_backend(),
        "Frontend (Streamlit)": verify_frontend(),
        "Tests": verify_tests(),
    }

    print("\n" + "="*50)
    print("검증 결과 요약")
    print("="*50)

    for module, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{module}: {status}")

    print("="*50)

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 모든 검증 통과! Frontend와 Backend가 성공적으로 구현되었습니다.\n")
        print("📝 다음 단계:")
        print("1. 백엔드 실행: python -m src.api.main")
        print("2. 프론트엔드 실행: streamlit run app.py")
        print()
        return 0
    else:
        print("\n⚠️  일부 검증 실패. 위 에러를 확인하세요.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
