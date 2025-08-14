# debug_vectorstore.py (프로젝트 루트에 생성)
"""
벡터스토어 상태 진단 스크립트
"""

import os
from pathlib import Path

def check_vectorstore_status():
    """벡터스토어 파일 존재 여부 및 상태 체크"""
    
    print("🔍 벡터스토어 상태 진단 시작")
    print("=" * 50)
    
    # 프로젝트 루트 확인
    root_dir = Path(__file__).parent
    vectorstore_dir = root_dir / "vectorstores"
    
    print(f"📁 프로젝트 루트: {root_dir}")
    print(f"📁 벡터스토어 디렉토리: {vectorstore_dir}")
    print(f"📁 벡터스토어 존재 여부: {vectorstore_dir.exists()}")
    
    if not vectorstore_dir.exists():
        print("❌ vectorstores 디렉토리가 없습니다!")
        print("💡 해결방법: make build-index-all 실행")
        return
    
    # 도메인별 벡터스토어 체크
    domains = ["satisfaction", "general", "publish", "cyber", "menu", "notice"]
    
    for domain in domains:
        print(f"\n🎯 {domain.upper()} 도메인 체크:")
        
        if domain == "satisfaction":
            vectorstore_path = vectorstore_dir / "vectorstore_unified_satisfaction"
        elif domain == "publish":
            vectorstore_path = vectorstore_dir / "vectorstore_unified_publish"
        else:
            vectorstore_path = vectorstore_dir / f"vectorstore_{domain}"
        
        print(f"  📂 경로: {vectorstore_path}")
        print(f"  📂 존재: {vectorstore_path.exists()}")
        
        if vectorstore_path.exists():
            faiss_file = vectorstore_path / f"{domain}_index.faiss"
            pkl_file = vectorstore_path / f"{domain}_index.pkl"
            bm25_file = vectorstore_path / f"{domain}_index.bm25"
            
            print(f"  📄 FAISS: {faiss_file.exists()} ({faiss_file.stat().st_size // 1024}KB)" if faiss_file.exists() else "  📄 FAISS: ❌")
            print(f"  📄 PKL: {pkl_file.exists()} ({pkl_file.stat().st_size // 1024}KB)" if pkl_file.exists() else "  📄 PKL: ❌")
            print(f"  📄 BM25: {bm25_file.exists()} ({bm25_file.stat().st_size // 1024}KB)" if bm25_file.exists() else "  📄 BM25: ❌")
        else:
            print("  ❌ 벡터스토어 디렉토리 없음")
    
    print("\n" + "=" * 50)
    print("🚀 수정 방법:")
    print("1. 인덱스 빌드: make build-index-all")
    print("2. 개별 도메인 빌드: make build-satisfaction")
    print("3. 데이터 확인: ls -la data/")

if __name__ == "__main__":
    check_vectorstore_status()
